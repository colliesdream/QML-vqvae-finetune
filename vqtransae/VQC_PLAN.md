# VQC Integration Plan (Current Progress Summary)

This document captures the current agreed direction for integrating a trainable VQC into the VQ-VAE pipeline, based on the latest discussion. The goal is to use a quantum circuit to produce a similarity/distance score that replaces the classical commitment distance between encoder output and codebook selection.

## Objective

Use a **trainable VQC** to produce a similarity score between `z_e` (encoder output) and `z_q` (selected codebook vector), then convert that score into a classical scalar loss term. This loss will **replace the original L2 commitment distance**.

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

