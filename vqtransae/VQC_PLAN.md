# VQC Integration Plan (Current Progress Summary)

## Phase 1 Completion (Integration Milestone)

Phase 1 is complete: the quantum commitment loss is now integrated into the
training pipeline with a PennyLane + JAX implementation, configuration toggles,
warm-up scheduling, and evaluation that tolerates optional quantum parameters.
The next phase focuses on experimentation, tuning, and stability analysis.

## Phase 2 Draft: Quantum-EMA Codebook Update (Top-4 Soft Update)

Goal: keep **classical L2 token selection** for stability, but update the codebook
using **quantum similarity-weighted EMA** so the representation learning is biased
toward quantum space.

### Draft Update Flow (per batch)

1. **Classical top-K selection (K=4)**
   - For each `z_e`, compute L2 distance to all codebook entries.
   - Keep the top-4 nearest tokens (indices).
2. **Quantum similarity for top-4 only**
   - For each `(z_e, E_k)` pair in top-4, compute fidelity via the existing VQC.
   - This reuses the current VQC circuit (same as commitment loss), but treats
     fidelity as a **soft similarity score**.
3. **Soft weights from similarity**
   - Convert the 4 fidelity scores into weights using softmax with temperature `τ`:
     `w_k = softmax(fidelity / τ)`.
4. **Quantum-weighted EMA update**
   - Replace one-hot EMA updates with weighted updates:
     - `ema_count[k] = decay * ema_count[k] + (1 - decay) * Σ_batch w_k`
     - `ema_weight[k] = decay * ema_weight[k] + (1 - decay) * Σ_batch (w_k * z_e)`
   - Update codebook vectors as usual:
     `E_k = ema_weight[k] / (ema_count[k] + eps)`

### Expected Cost

- Each fidelity uses 2 VQC forwards (one for `z_e`, one for `E_k`).
- Per batch cost ≈ `B * 4 * 2` VQC forward calls.

### Notes / Open Items

- Keep the quantum commitment loss optional; this phase only targets codebook updates.
- The temperature `τ` controls how “soft” the top-4 update is.
- We can add a hybrid fallback (one-hot + quantum weights) if stability is an issue.

## Phase 3 Draft: QKernel Feature Mapping Before VQ

Goal: insert a fixed (non-trainable) qkernel feature mapping between the encoder
output `z_e` and the VQ module to improve representation quality before
quantization. Codebook updates remain unchanged in this phase.

### Draft Update Flow (per batch)

1. **Encoder output**: produce `z_e` with shape `(B, T, LATENT_DIM=32)`.
2. **QKernel mapping**:
   - Use **A = 32** anchors derived from K-Means on `z_e`.
   - Compute kernel features `k(z_e, anchor_i)` for each anchor.
   - Output shape becomes `(B, T, 32)` (no compression).
3. **Direct feed into VQ**:
   - No linear projection; `(B, T, 32)` is passed directly to VQ.
4. **Anchor refresh cadence**:
   - Recompute K-Means anchors **every 5 epochs**.

### Implementation Decisions (Confirmed)

- QKernel is **non-trainable** (fixed feature map) with **CNOT entanglement enabled**.
- `A = 32` anchors; `K = A` in K-Means.
- Anchors are updated every 5 epochs (not every epoch) to control cost.

## Phase 4 Draft: Trainable Quantum Encoder Bottleneck

Goal: replace qkernel with a **trainable VQC bottleneck** before VQ, while
keeping the VQ and loss **classical** to focus compute on the quantum encoder.

### Draft Update Flow (per batch)

1. **Encoder output**: produce `z_e` with shape `(B, T, 32)`.
2. **Quantum bottleneck (trainable)**:
   - Linear map `32 → 8` angles.
   - VQC with `RY + ring CNOT` layers.
   - Measure 8 qubits (Pauli-Z expectations) → `(B, T, 8)`.
3. **Linear expansion**:
   - Project `(B, T, 8)` back to `(B, T, 32)` before VQ.
4. **Classical VQ + loss**:
   - Use standard VQ EMA update and classical commitment loss.

### Implementation Decisions (Confirmed)

- Use `QENCODER_NUM_QUBITS = 8`, `QENCODER_NUM_LAYERS = 1`.
- Disable quantum commitment loss and quantum EMA during this phase.

## Phase 5 Draft: Anomaly Window Clustering (HDBSCAN)

Goal: after evaluation, cluster **anomalous windows** using similarity in
`(E_norm, D_norm, A_norm)` space to discover anomaly groups without defining
explicit pothole categories.

### Draft Update Flow

1. **Evaluate** to compute `E_norm`, `D_norm`, `A_norm`, and composite score `S`.
2. **Select anomalies** on the **test set** via the existing percentile threshold on `S`.
3. **Cluster anomalies** with HDBSCAN using 3D features:
   - feature vector per anomaly window: `[E_norm, D_norm, A_norm]`.

### Required Outputs (All)

**Cluster summary (per cluster)**
- `cluster_id`
- `size` and `proportion`
- `mean_E_norm`, `mean_D_norm`, `mean_A_norm`
- `representative_indices` (closest samples to cluster center)
- `stability` (HDBSCAN cluster stability score)

**Cluster members (per window)**
- `window_index` (to map back to time/location)
- `cluster_id` (or `-1` for noise)
- `E_norm`, `D_norm`, `A_norm`
- `membership_probability`
- `is_noise`

**Global metrics**
- `silhouette_score` (on clustered points)
- `noise_ratio`
- HDBSCAN parameters (`min_cluster_size`, `min_samples`)

### Notes

- Store outputs to JSON/CSV **and** print key metrics to stdout.
- Use PCA/UMAP plots optionally for qualitative checks (not required).

### Implementation Decisions (Confirmed)

- Use **HDBSCAN** with `min_cluster_size >= 2`; tune `min_samples` by sensitivity.
- Features are **only** `[E_norm, D_norm, A_norm]`.
- Cluster only **test-set anomalies** after thresholding.
- Output files:
  - `cluster_summary.(json|csv)`
  - `cluster_members.(json|csv)`
- Print summary metrics with a separator between Phase 3 and Phase 5 outputs.

## Phase 6 Draft: VQ Code Histogram Clustering (HDBSCAN + JS Distance)

Goal: cluster anomaly windows using **VQ codebook usage histograms** (shape
signatures) instead of scalar E/D/A scores, to better capture anomaly types.

### Draft Update Flow

1. **Collect anomaly windows** on the test set (same thresholding as Phase 5).
2. **Build VQ code histograms** per window:
   - Histogram dimension = `codebook_size` (e.g., 1024).
   - Use L1 normalization.
3. **Distance metric**:
   - Jensen–Shannon distance (with small epsilon smoothing).
4. **Clustering**:
   - HDBSCAN on histogram features.
5. **Representatives**:
   - Select representative windows by highest membership probability.

### Implementation Decisions (Confirmed)

- Feature: **code histogram only** (no hybrid stats).
- Histogram dimension: **full codebook size (1024)**.
- Distance: **Jensen–Shannon**.
- Clustering: **HDBSCAN**.
- Representative selection: **highest membership probability**.

### Hypothesis: Why Quantum-EMA Could Outperform Classic EMA

- **Richer assignment signal:** EMA uses hard one-hot assignments, while quantum
  similarity provides a graded signal that can update multiple nearby tokens,
  potentially improving codebook coverage and representation smoothness.
- **Alignment with VQC embedding:** If the VQC captures useful similarity structure,
  codebook updates that follow quantum similarity may better match downstream
  anomaly scoring.
- **Controlled softness:** The temperature `τ` allows interpolation between
  hard (EMA-like) and soft updates without changing token selection logic.

### Evidence Plan (to validate superiority vs EMA)

1. **Ablation vs classic EMA**
   - Compare baseline EMA vs quantum-EMA (Top-4) with identical training budgets.
2. **Core metrics**
   - Reconstruction loss trajectory.
   - Active token count + perplexity (codebook health).
   - Test-set anomaly metrics (Precision/Recall/F1).
3. **Stability checks**
   - Monitor quantum similarity distribution (mean/min/max) to detect collapse.
   - Track whether top-4 weights become too peaky (τ too low) or too flat (τ too high).

This document captures the current agreed direction for integrating a trainable VQC into the VQ-VAE pipeline, based on the latest discussion. The goal is to use a quantum circuit to produce a similarity/distance score that replaces the classical commitment distance between encoder output and codebook selection.

## Objective

Use a **trainable VQC** to produce a similarity score between `z_e` (encoder output) and `z_q` (selected codebook vector), then convert that score into a classical scalar loss term. This loss will **replace the original L2 commitment distance**.

## Scope Clarification (Agreed)

- **Token selection remains classical**: `z_q` is still selected by the current nearest-neighbor codebook lookup.
- **VQC is used only for the distance/similarity score** between `z_e` and `z_q`.
- **The VQC is trainable**, and its output is used as a loss term that replaces the classical commitment term.

## Minimal Viable Combination (Agreed)

1. **Similarity metric:** Fidelity via kernel/inner product estimation.
2. **Input handling:** Separate angle encoding of `z_e` and `z_q` into the same VQC, followed by overlap computation.
3. **Loss strategy:** Replace the original L2/commitment loss with the quantum similarity-based loss.
4. **Training schedule:** Two-stage training
   - Stage 1: Freeze encoder + codebook (+ decoder), train only projection + VQC.
   - Stage 2: Joint fine-tuning of all modules.

## Conceptual Flow

```
Encoder(x) -> z_e
Codebook lookup -> z_q

z_e -> VQC -> |ψ_e⟩
z_q -> VQC -> |ψ_q⟩

Similarity = Fidelity(|ψ_e⟩, |ψ_q⟩)
Quantum loss = λ * (1 - Similarity)
```

- `λ` is a tunable scale factor to align the magnitude of the quantum loss with the rest of the training losses.
- The VQC uses the **same circuit architecture** for both inputs; only parameters differ based on the input vector.

## Quantum Loss Definition (Replacement)

The classical commitment term typically looks like:

```
L_commit = ||z_e - z_q||^2
```

This is replaced by:

```
L_commit_quantum = λ * (1 - Fidelity(|ψ_e⟩, |ψ_q⟩))
```

Notes:
- Fidelity is bounded in `[0, 1]`, so `L_commit_quantum` is bounded in `[0, λ]`.
- We do **not** need to match the original L2 scale exactly; `λ` is tuned to balance the overall loss.

## Training Schedule Details

**Stage 1: VQC warm-up (frozen encoder + codebook)**
- Encoder, codebook, and decoder are frozen to stabilize the input embedding geometry.
- Train the learned projection + VQC parameters to approximate a useful similarity signal.
- Initial run target: 25 epochs.

**Stage 2: Joint fine-tuning**
- Unfreeze encoder and codebook.
- Jointly optimize reconstruction + VQ loss + quantum commitment loss.

## Implementation Notes (PennyLane + JAX)

- Use a **PennyLane `default.qubit` device** with `interface="jax"` and `diff_method="best"` for the VQC.
- Wrap the VQC in a **JAX-jitted forward function** and compute gradients via **`jax.vjp`** for efficient vector-Jacobian products.
- For a PyTorch pipeline, build a **custom autograd bridge** that:
  - Converts tensors to JAX arrays (`jnp.array`), runs the JIT-compiled forward, and returns results as Torch tensors.
  - Uses the saved inputs/weights in `backward` to call the JIT-compiled VJP function and convert gradients back to Torch.
- Keep an explicit **mini-batch loop** around the VQC call to limit JAX memory usage.
- Default to **CPU backend** for JAX during initial integration to simplify environment setup and debugging.

## Success Criteria (Initial Experiments)

We will consider the integration viable if:
- Reconstruction loss does not degrade significantly vs. baseline.
- Codebook usage remains healthy (no collapse).
- The quantum loss provides a stable gradient signal without exploding/vanishing.

## Finalized Design Decisions

1. **Input encoding**
   - Angle encoding with **RY rotations** (Hadamard only if needed for initialization, not for data encoding).
2. **Feature map / circuit architecture**
   - Hardware-efficient ansatz with **ring CNOT entanglement**, **1 layer** to start.
3. **Similarity / measurement strategy**
   - Fidelity via **kernel/inner-product estimation**.
   - *Note:* Keep a follow-up experiment note to try **classical overlap** estimation in simulator.
4. **VQC output form**
   - **Scalar similarity** score (not an embedding).
5. **z-dimension to qubit mapping**
   - **Learned linear projection** from `z` to qubit angles.
   - **8 qubits** (optionally 9 if needed).
6. **Quantum loss weighting**
   - **Warm-up schedule** for `λ` (small early, then increase).
7. **Stage 1 training plan**
   - Freeze encoder + codebook + decoder.
   - Train projection + VQC only.
   - **25 epochs** as the initial target.

## Notes

- Output scale does **not** need to match classical L2 exactly; it can be aligned by tuning `λ`.
- Fidelity provides a bounded similarity score in `[0, 1]`, which stabilizes the loss.

## Observability (Outputs to Track)

We will track **three outputs** to observe the impact of the VQC integration:

1. **Fidelity / similarity value**
   - Log the batch mean (and optionally min/max) of the fidelity score.
2. **Reconstruction outputs**
   - Save reconstructed samples periodically to visually compare quality.
3. **Loss curves**
   - Track `recon_loss`, `quantum_commit_loss`, and `total_loss` over time.
   - Use the same run to compare these curves; a separate baseline run is optional.

## Fidelity (Explanation)

Fidelity measures **how similar two quantum states are**. For states |ψ_e⟩ and |ψ_q⟩,

```
Fidelity = |⟨ψ_e | ψ_q⟩|^2
```

- **Range:** `[0, 1]` (1 means identical, 0 means orthogonal).
- We use **`1 - Fidelity`** as a distance-like term in the quantum commitment loss.
