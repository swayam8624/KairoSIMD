# KairoSIMD

KairoSIMD is the CPU vector-kernel layer for Kairo ML. It defines the stable API
that tensor kernels call for hot inner loops, starting with scalar reference
implementations and leaving room for NEON, AVX2, and AVX-512 backends.

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

- CPU feature detection surface through `BaselineFeature()`.
- Scalar fallback kernels for every exposed operation.
- Span-based APIs that compose naturally with tensor views and scheduled ranges.
- A future dispatch point for platform-specific implementation files.

Current kernels:

- `Fill`, `Copy`
- `Add`, `Sub`, `Mul`
- `Scale`, `Clamp`, `Axpy`
- `Dot`, `Sum`, `L2Norm`
- `ReLU`, `Sigmoid`, `Softmax`

## Where It Connects

- `KairoMath::Tensor`: delegates contiguous/ranged math to KairoSIMD.
- `KairoScheduler`: splits work, then each range calls KairoSIMD kernels.
- `MLLibrary`: uses tensor kernels for training, inference, and metrics.
- `KairoGPU`: remains a separate backend; KairoSIMD is the CPU backend.

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
cmake --build build
./build/KairoSIMDSmoke
```

## Roadmap

- NEON implementation for Apple Silicon and ARM.
- AVX2/AVX-512 implementation for x86 servers.
- Runtime dispatch tables.
- Alignment-aware kernels.
- Fused operations such as bias+activation and scale+add.
- Microbenchmarks and regression thresholds.
