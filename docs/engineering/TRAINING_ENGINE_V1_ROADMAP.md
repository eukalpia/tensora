# Training Engine V1 — implementation sequence

The implementation proceeds as validated vertical slices. A later slice must not weaken an earlier correctness or lifetime contract.

1. Complete tensor view and alias semantics.
2. Move all new training, optimizer, checkpoint, finite-difference, and view proofs into the ordinary cross-platform native test matrix.
3. Generalize optimizers from a single Linear module to arbitrary parameter collections.
4. Add module composition and parameter registration required for multi-layer models.
5. Expand Tensor Core operations, broadcasting, indexing, reductions, and scalar semantics.
6. Add native dtype storage and promotion rules, then integer class targets for cross-entropy.
7. Add normalization, embedding, dropout, softmax/log-softmax, GELU/SiLU/SwiGLU, and transformer primitives.
8. Add Dataset/DataLoader/Trainer infrastructure with deterministic checkpoint/resume.
9. Train a small GPT end-to-end without LibTorch as the primary Training Engine V1 product proof.
10. Introduce graph capture and Tensora IR only after eager correctness is established.
11. Add compiled memory planning, fusion, and backend lowering.
12. Qualify CUDA on physical hardware, then add mixed precision and distributed primitives.

Large-model distributed training is a later acceptance level. It must build on the same tensor, autograd, mutation, checkpoint, and numerical contracts rather than creating an independent execution path.
