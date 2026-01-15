"""
Quantum commitment loss powered by PennyLane + JAX.

Implements angle encoding with RY rotations, a ring CNOT entangler, and
trainable single-qubit rotations. The fidelity between two encoded states
is used as a commitment loss.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


@lru_cache(maxsize=None)
def _build_state_circuit(num_qubits: int, num_layers: int):
    dev = qml.device("default.qubit", wires=num_qubits)

    def circuit(angles, weights):
        for i in range(num_qubits):
            qml.RY(angles[i], wires=i)
        for layer in range(num_layers):
            for i in range(num_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(num_qubits):
                qml.CNOT(wires=[i, (i + 1) % num_qubits])
        return qml.state()

    return qml.QNode(circuit, dev, interface="jax", diff_method="best")


def _fidelity_batch(
    z_e: jnp.ndarray,
    z_q: jnp.ndarray,
    proj_w: jnp.ndarray,
    proj_b: jnp.ndarray,
    weights: jnp.ndarray,
    num_qubits: int,
    num_layers: int,
) -> jnp.ndarray:
    qnode = _build_state_circuit(num_qubits, num_layers)
    angles_e = jnp.dot(z_e, proj_w.T) + proj_b
    angles_q = jnp.dot(z_q, proj_w.T) + proj_b

    def single_fidelity(ae, aq):
        psi_e = qnode(ae, weights)
        psi_q = qnode(aq, weights)
        overlap = jnp.vdot(psi_e, psi_q)
        return jnp.abs(overlap) ** 2

    fidelities = jax.vmap(single_fidelity)(angles_e, angles_q)
    return jnp.mean(fidelities)


class _QuantumCommitmentFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_e, z_q, proj_w, proj_b, weights, weight, num_qubits: int, num_layers: int):
        z_e_np = np.asarray(z_e.detach().cpu())
        z_q_np = np.asarray(z_q.detach().cpu())
        proj_w_np = np.asarray(proj_w.detach().cpu())
        proj_b_np = np.asarray(proj_b.detach().cpu())
        weights_np = np.asarray(weights.detach().cpu())
        weight_np = float(weight.detach().cpu())

        z_e_j = jnp.array(z_e_np)
        z_q_j = jnp.array(z_q_np)
        proj_w_j = jnp.array(proj_w_np)
        proj_b_j = jnp.array(proj_b_np)
        weights_j = jnp.array(weights_np)

        def loss_fn(z_e_in, z_q_in, proj_w_in, proj_b_in, weights_in):
            fidelity = _fidelity_batch(
                z_e_in,
                z_q_in,
                proj_w_in,
                proj_b_in,
                weights_in,
                num_qubits,
                num_layers,
            )
            return weight_np * (1.0 - fidelity)

        loss_value, vjp_fn = jax.vjp(loss_fn, z_e_j, z_q_j, proj_w_j, proj_b_j, weights_j)
        ctx.vjp_fn = vjp_fn
        ctx.num_qubits = num_qubits
        ctx.num_layers = num_layers
        ctx.device = z_e.device

        loss_tensor = torch.tensor(np.asarray(loss_value), dtype=z_e.dtype, device=z_e.device)
        return loss_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_np = np.asarray(grad_output.detach().cpu())
        (dz_e, dz_q, dproj_w, dproj_b, dweights) = ctx.vjp_fn(jnp.array(grad_output_np))

        def to_torch(array):
            return torch.tensor(np.asarray(array), device=ctx.device)

        return (
            to_torch(dz_e),
            to_torch(dz_q),
            to_torch(dproj_w),
            to_torch(dproj_b),
            to_torch(dweights),
            None,
            None,
            None,
        )


class QuantumCommitmentLoss(nn.Module):
    """PennyLane + JAX fidelity commitment loss with trainable projection."""

    def __init__(
        self,
        latent_dim: int,
        num_qubits: int = 8,
        num_layers: int = 1,
        weight: float = 1.0,
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.weight_value = weight
        self.proj_weight = nn.Parameter(torch.empty(num_qubits, latent_dim))
        self.proj_bias = nn.Parameter(torch.zeros(num_qubits))
        self.vqc_weights = nn.Parameter(torch.empty(num_layers, num_qubits, 2))
        nn.init.xavier_uniform_(self.proj_weight)
        nn.init.zeros_(self.proj_bias)
        nn.init.xavier_uniform_(self.vqc_weights)

    def set_weight(self, weight: float) -> None:
        """Update the weighting factor for the commitment loss."""
        self.weight_value = weight

    def forward(self, z_e: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        z_e_flat = z_e.reshape(-1, z_e.shape[-1])
        z_q_flat = z_q.reshape(-1, z_q.shape[-1])
        weight = torch.tensor(self.weight_value, device=z_e.device, dtype=z_e.dtype)
        return _QuantumCommitmentFunction.apply(
            z_e_flat,
            z_q_flat,
            self.proj_weight,
            self.proj_bias,
            self.vqc_weights,
            weight,
            self.num_qubits,
            self.num_layers,
        )
