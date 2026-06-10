# KairoSIMD

KairoSIMD is the scalar-first SIMD kernel layer for Kairo ML.

Phase 1 scope:

- stable vector-kernel API,
- scalar fallback implementations,
- CPU feature detection surface,
- kernels needed by tensor training loops.

Later phases should add NEON/AVX dispatch tables and backend-specific optimized
files without changing public tensor code.
