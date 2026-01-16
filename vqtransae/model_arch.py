"""
VQTransAE model architecture.
Vector Quantized Transformer Autoencoder for road surface anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from .quantum import QuantumKernelMapper


# ——————————— Relative Position Bias ——————————— #
class RelPosBias(nn.Module):
    """Relative position bias for attention."""
    def __init__(self, n_heads: int, max_dist: int = 512):
        super().__init__()
        self.n_heads = n_heads
        self.max_dist = max_dist
        self.rel_bias = nn.Parameter(torch.zeros(2*max_dist + 1, n_heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

    def forward(self, q_len: int, k_len: int, device=None):
        device = device or self.rel_bias.device
        qs = torch.arange(q_len, device=device)[:, None]
        ks = torch.arange(k_len, device=device)[None, :]
        rel = (ks - qs).clamp(-self.max_dist, self.max_dist) + self.max_dist
        bias = self.rel_bias[rel]
        bias = bias.permute(2, 0, 1)
        return bias


# ——————————— Transformer Block ——————————— #
class TSBlock(nn.Module):
    """Transformer block with relative position bias."""
    def __init__(self, d_model=64, n_heads=4, ff_mult=4, drop=0.1, max_dist=512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_dist = max_dist

        self.qkv_proj = nn.Linear(d_model, d_model*3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.bias = RelPosBias(n_heads, max_dist)
        self.dropout = nn.Dropout(drop)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*ff_mult),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_model*ff_mult, d_model),
            nn.Dropout(drop)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, D//self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, D//self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, D//self.n_heads).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / sqrt(D//self.n_heads)
        bias = self.bias(T, T, x.device).unsqueeze(0)
        scores = scores + bias
        attn = F.softmax(scores, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        out = self.dropout(self.out_proj(out))
        x2 = self.norm1(x + out)
        x3 = self.norm2(x2 + self.ff(x2))

        return x3, attn


# ——————————— Sequence Encoder ——————————— #
class SeqEncoder(nn.Module):
    """Bidirectional LSTM encoder."""
    def __init__(self, in_dim=6, hidden=64, latent_dim=32, n_layers=2, bidirectional=True, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0
        )
        lstm_output_dim = hidden * 2 if bidirectional else hidden
        self.proj = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, latent_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        z_e = self.proj(out)
        return z_e


# ——————————— Vector Quantizer with EMA ——————————— #
class VectorQuantizerEMA(nn.Module):
    """Vector quantizer with exponential moving average codebook updates."""
    def __init__(self, K=512, D=32, decay=0.99, eps=1e-5, commitment_cost=0.25,
                 commitment_loss_fn: nn.Module = None, use_quantum_ema: bool = False,
                 quantum_ema_topk: int = 4, quantum_ema_tau: float = 0.5):
        super().__init__()
        self.K, self.D, self.decay, self.eps = K, D, decay, eps
        self.commitment_cost = commitment_cost
        self.commitment_loss_fn = commitment_loss_fn
        self.ema_update = True
        self.use_quantum_ema = use_quantum_ema
        self.quantum_ema_topk = quantum_ema_topk
        self.quantum_ema_tau = quantum_ema_tau

        self.embed = nn.Embedding(K, D)
        self.register_buffer('ema_count', torch.zeros(K))
        self.register_buffer('ema_weight', self.embed.weight.data.clone())
        nn.init.kaiming_uniform_(self.embed.weight)

    def set_ema_update(self, enabled: bool) -> None:
        """Enable or disable EMA updates (useful for warm-up stages)."""
        self.ema_update = enabled

    def forward(self, z_e):
        flat = z_e.reshape(-1, self.D)

        flat_norm = torch.sum(flat ** 2, dim=1, keepdim=True)
        embed_norm = torch.sum(self.embed.weight ** 2, dim=1)
        dist = flat_norm + embed_norm - 2 * torch.matmul(flat, self.embed.weight.t())

        encoding_indices = torch.argmin(dist, dim=1)
        encodings = F.one_hot(encoding_indices, self.K).float()
        quantized = self.embed(encoding_indices).view_as(z_e)

        if self.training and self.ema_update:
            if self.use_quantum_ema:
                if not hasattr(self.commitment_loss_fn, 'fidelity'):
                    raise RuntimeError("Quantum EMA requires a commitment_loss_fn with fidelity().")
                topk = min(self.quantum_ema_topk, self.K)
                with torch.no_grad():
                    _, topk_indices = torch.topk(dist, k=topk, dim=1, largest=False)
                    topk_embeddings = self.embed(topk_indices)

                    fidelity_scores = []
                    for k in range(topk):
                        fidelity_scores.append(
                            self.commitment_loss_fn.fidelity(flat, topk_embeddings[:, k, :])
                        )
                    fidelities = torch.stack(fidelity_scores, dim=1)
                    weights = torch.softmax(fidelities / self.quantum_ema_tau, dim=1)

                    ema_count_update = torch.zeros(self.K, device=flat.device)
                    ema_weight_update = torch.zeros(self.K, self.D, device=flat.device)

                    for k in range(topk):
                        indices_k = topk_indices[:, k]
                        weights_k = weights[:, k]
                        ema_count_update.scatter_add_(0, indices_k, weights_k)
                        ema_weight_update.index_add_(0, indices_k, weights_k.unsqueeze(1) * flat)

                    self.ema_count = self.decay * self.ema_count + (1 - self.decay) * ema_count_update
                    n = torch.sum(self.ema_count)
                    weight_norm = ((self.ema_count + self.eps) / (n + self.K * self.eps)) * n
                    self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * ema_weight_update
                    self.embed.weight.data = self.ema_weight / weight_norm.unsqueeze(1)
            else:
                self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
                n = torch.sum(self.ema_count)
                weights = ((self.ema_count + self.eps) / (n + self.K * self.eps)) * n
                dw = torch.matmul(encodings.t(), flat)
                self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
                self.embed.weight.data = self.ema_weight / weights.unsqueeze(1)

        e_latent_loss = F.mse_loss(quantized.detach(), z_e)
        if self.commitment_loss_fn is None:
            q_latent_loss = F.mse_loss(quantized, z_e.detach())
            loss = e_latent_loss + self.commitment_cost * q_latent_loss
        else:
            q_latent_loss = self.commitment_loss_fn(z_e, quantized)
            loss = e_latent_loss + q_latent_loss

        quantized = z_e + (quantized - z_e).detach()
        return quantized, loss, encoding_indices


# ——————————— Temporal Conv Block ——————————— #
class TemporalConvBlock(nn.Module):
    """Single temporal convolution block with residual."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        residual = x
        out = self.dropout(self.relu(self.norm1(self.conv1(x))))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)


# ——————————— Temporal Convolutional Network ——————————— #
class TCN(nn.Module):
    """Stacked temporal conv blocks with exponentially increasing dilation."""
    def __init__(self, in_channels, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else channels[i - 1]
            layers.append(TemporalConvBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        return x.transpose(1, 2)


# ——————————— VQTransAE Model ——————————— #
class VQTransAE(nn.Module):
    """
    Vector Quantized Transformer Autoencoder.

    Combines bidirectional LSTM encoding, vector quantization with EMA,
    temporal convolutions, and transformer layers for anomaly detection.
    """
    def __init__(self, win_size, in_dim=6,
                 hidden=64, latent=32,
                 codebook=512, d_model=64,
                 heads=4, layers=3, dropout=0.1,
                 commitment_loss_fn: nn.Module = None,
                 use_quantum_ema: bool = False,
                 quantum_ema_topk: int = 4,
                 quantum_ema_tau: float = 0.5,
                 use_qkernel: bool = False,
                 qkernel_num_qubits: int = 8,
                 qkernel_anchors: int = 16):
        super().__init__()

        self.encoder = SeqEncoder(in_dim, hidden, latent, n_layers=2, bidirectional=True, dropout=dropout)
        self.use_qkernel = use_qkernel
        self.qkernel_anchors = qkernel_anchors
        if self.use_qkernel:
            self.qkernel_mapper = QuantumKernelMapper(latent_dim=latent, num_qubits=qkernel_num_qubits)
            self.qkernel_proj = nn.Linear(qkernel_anchors, latent)
            self.register_buffer("qkernel_anchor_bank", torch.zeros(qkernel_anchors, latent))
            self.qkernel_initialized = False
        self.quant = VectorQuantizerEMA(
            codebook,
            latent,
            decay=0.99,
            eps=1e-5,
            commitment_cost=0.25,
            commitment_loss_fn=commitment_loss_fn,
            use_quantum_ema=use_quantum_ema,
            quantum_ema_topk=quantum_ema_topk,
            quantum_ema_tau=quantum_ema_tau,
        )
        self.embed = nn.Linear(latent, d_model)
        self.tcn = TCN(d_model, [d_model, d_model, d_model], kernel_size=3, dropout=dropout)
        self.tf_layers = nn.ModuleList([
            TSBlock(d_model, heads, ff_mult=4, drop=dropout, max_dist=win_size) for _ in range(layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, in_dim)
        )
        self.use_skip = True
        if self.use_skip:
            self.skip_proj = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        if self.use_qkernel:
            if not self.qkernel_initialized:
                raise RuntimeError("QKernel anchors are not initialized.")
            kernel_features = self.qkernel_mapper(z_e, self.qkernel_anchor_bank)
            z_e = self.qkernel_proj(kernel_features)
        z_q, loss_vq, indices = self.quant(z_e)
        h = self.embed(z_q)
        h = self.tcn(h)

        attn_list = []
        for blk in self.tf_layers:
            h, attn = blk(h)
            attn_list.append(attn)

        recon = self.decoder(h)
        if self.use_skip:
            recon = recon + self.skip_proj(x)

        return recon, attn_list, loss_vq, indices
