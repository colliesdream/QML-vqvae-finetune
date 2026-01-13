# VQC Integration Plan (Current Progress Summary)

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
