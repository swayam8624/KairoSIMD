# KairoSIMD

KairoSIMD is the CPU vector-kernel layer for Kairo ML. It defines the stable API
that tensor kernels call for hot inner loops, with scalar reference
implementations and Apple Silicon NEON fast paths for core float kernels.

## Problem

ML performance depends on simple operations repeated billions of times:

- elementwise add/sub/mul,
- scale and AXPY,
- dot products and reductions,
- activation functions,
- softmax and normalization kernels.

These kernels need a scalar correctness path, but the public tensor API should
not change when optimized CPU implementations arrive.

## Solution

KairoSIMD provides:

- Compile-time and runtime CPU feature reporting through `BaselineFeature()`
  and `DetectedFeature()`.
- Scalar fallback kernels for every exposed operation.
- NEON float fast paths on ARM for `Add`, `Sub`, `Mul`, `Scale`, `AXPY`,
  `Dot`, `Sum`, and `ReLU`.
- Runtime-dispatched AVX2 and AVX-512 Float32 add/dot kernels on x86, compiled
  as targeted functions so the same binary retains scalar fallback.
- Span-based APIs that compose naturally with tensor views and scheduled ranges.
- A future dispatch point for platform-specific implementation files.

Current kernels:

- `Fill`, `Copy`
- `Add`, `Sub`, `Mul`
- `Scale`, `Clamp`, `Axpy`
- `Dot`, `Sum`, `L2Norm`
- `ReLU`, `Sigmoid`, `Softmax`
- fused `BiasReLU` and stateful `AdamW` update

`KairoSIMDBenchmark` writes `kairo.simd.benchmark.v1` JSON containing the
detected backend, fixed problem size, warmups, measured iterations, median
latency, element throughput, and scalar numerical error. It is a bounded CTest
gate and a machine-readable input for hardware/compiler-specific histories.

Separate ASan and TSan build configurations are available through
`KAIRO_SIMD_ENABLE_ASAN` and `KAIRO_SIMD_ENABLE_TSAN`.
`KAIRO_SIMD_MIN_ADD_ELEMENTS_PER_SECOND` sets a hardware-specific CTest
throughput floor. CI must calibrate it per runner/compiler; leaving it at zero
records evidence without making a non-portable speed claim.

## Where It Connects

- `KairoMath::Tensor`: delegates contiguous/ranged math to KairoSIMD.
- `KairoScheduler`: splits work, then each range calls KairoSIMD kernels.
- `MLLibrary`: uses tensor kernels for training, inference, and metrics.
- `KairoGPU`: remains a separate backend; KairoSIMD is the CPU backend.

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
cmake --build build
ctest --test-dir build --output-on-failure
./build/KairoSIMDSmoke
./build/KairoSIMDBenchmark build/kairo-simd-benchmark.json
```

## Roadmap

- Expand AVX2/AVX-512 coverage beyond add and dot.
- General runtime dispatch tables.
- Alignment-aware kernels.
- Fused operations such as bias+activation and scale+add.
- Microbenchmarks and regression thresholds.
