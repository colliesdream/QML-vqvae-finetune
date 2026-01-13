# VQC Integration Plan (Current Progress Summary)

This document captures the current agreed direction for integrating a trainable VQC into the VQ-VAE pipeline, based on the latest discussion. The goal is to use a quantum circuit to produce a similarity/distance score that replaces the classical commitment distance between encoder output and codebook selection.

## Objective

Use a **trainable VQC** to produce a similarity score between `z_e` (encoder output) and `z_q` (selected codebook vector), then convert that score into a classical scalar loss term. This loss will **replace the original L2 commitment distance**.

## Scope Clarification (Agreed)

- **Token selection remains classical**: `z_q` is still selected by the current nearest-neighbor codebook lookup.
- **VQC is used only for the distance/similarity score** between `z_e` and `z_q`.
- **The VQC is trainable**, and its output is used as a loss term that replaces the classical commitment term.

## Minimal Viable Combination (Agreed)

1. **Similarity metric:** Fidelity (quantum state overlap).
2. **Input handling:** Separate encoding of `z_e` and `z_q`, followed by overlap computation.
3. **Loss strategy:** Replace the original L2/commitment loss with the quantum similarity-based loss.
4. **Training schedule:** Two-stage training
   - Stage 1: Freeze encoder + codebook, train only VQC.
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
- Encoder and codebook are frozen to stabilize the input embedding geometry.
- Train the VQC parameters to approximate a useful similarity signal.

**Stage 2: Joint fine-tuning**
- Unfreeze encoder and codebook.
- Jointly optimize reconstruction + VQ loss + quantum commitment loss.

## Success Criteria (Initial Experiments)

We will consider the integration viable if:
- Reconstruction loss does not degrade significantly vs. baseline.
- Codebook usage remains healthy (no collapse).
- The quantum loss provides a stable gradient signal without exploding/vanishing.

## Open Design Decisions (To Finalize)

These are the next points to pin down before implementation:

1. **Feature map / circuit architecture**
   - Hardware-efficient ansatz vs. Pauli feature map.
2. **Measurement strategy**
   - Fidelity via overlap or kernel evaluation method.
3. **Parameterization of inputs**
   - Angle encoding vs. other encodings.
4. **Integration location**
   - Exact codepath where `|z_e - z_q|^2` is replaced.
5. **Training hyperparameters**
   - `λ` scaling, VQC depth, number of qubits, etc.

## Notes

- Output scale does **not** need to match classical L2 exactly; it can be aligned by tuning `λ`.
- Fidelity provides a bounded similarity score in `[0, 1]`, which stabilizes the loss.
