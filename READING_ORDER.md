# KairoSIMD — linked source learning order

Start with portable scalar arithmetic and vector API before reading the platform-dispatched Apple Silicon NEON implementations. SIMD speed is not guaranteed: compare semantics, alignment, vector widths and a scalar fallback on the exact same inputs.

## APIs and neighbors

Public vector/AXPY/dot/activation operations must produce numerically comparable outputs between scalar and SIMD kernels. Track feature selection, memory alignment, aliasing, bounds and precision/error tests. Consumers include Kairo ML and math libraries.

## Read these source files (4)

1. [`src/KairoSIMD.cppm`](src/KairoSIMD.cppm) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.
2. [`src/KairoSIMDPlatform.cpp`](src/KairoSIMDPlatform.cpp) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.
3. [`tests/smoke.cpp`](tests/smoke.cpp) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.
4. [`tests/benchmark.cpp`](tests/benchmark.cpp) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.

Read [README.md](README.md) for the toolchain and [KairoGameEngine](https://github.com/swayam8624/KairoGameEngine) for exact-head ecosystem acceptance. Original source is [MIT licensed](LICENSE) for learning, modification and continued development, without relicensing dependencies/fixtures. This guide is a documented navigation plan, not comprehensive in-file comment coverage or proof of correctness.
