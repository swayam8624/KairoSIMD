# KairoSIMD — what SIMD accelerates and what it must preserve

Read [`src/KairoSIMD.cppm`](../src/KairoSIMD.cppm) beside
[`src/KairoSIMDPlatform.cpp`](../src/KairoSIMDPlatform.cpp).
The first declares portable numerical operations and scalar reference
implementations; the second specializes selected Float32 functions with
native CPU vector instructions.

## 1. CPU vectorization is not CPU threading

A SIMD instruction processes multiple contiguous scalar lanes within
one CPU core at once. `std::span<T>` expresses the address and element
count of a contiguous range; it is not an owning allocation. A
`KairoScheduler::ParallelFor` can run separate span subranges on
different threads, but KairoSIMD by itself only executes on the
calling thread. KairoGPU is a different device and memory system.

Vectorization does not make noncontiguous KairoMath Tensor views
contiguous or fix aliasing hazards. The caller must check the memory
layout and whether output and input spans may overlap.

## 2. The feature-dispatch boundary

`BaselineFeature()` describes the instruction set selected at
compile time. `DetectedFeature()` describes a runtime decision.
Portable binaries must not execute an AVX-512 instruction merely
because the compiler knows how to generate it. CPU capability
discovery and OS support for extended register state must agree.

On ARM, NEON lanes and scalar tails use a different native path.
On x86, AVX2/AVX512 branches use the correct width plus a scalar
remainder. Loop end bounds are correctness-critical, especially
for inputs whose length is not a multiple of the vector width.

## 3. Arithmetic and numerical parity

The exported API includes `Fill`, `Copy`, `Add`, `Sub`,
`Mul`, `Scale`, `Clamp`, `Axpy`, `Dot`, `Sum`,
`L2Norm`, `ReLU`, `BiasReLU`, `LayerNormRows`,
`AdamW`, `Sigmoid`, and `Softmax`.

There are different accuracy contracts for each:

- Elementwise arithmetic usually permits close scalar parity, but
  exceptions include subnormals, infinities, NaNs and aliasing.
- Dot/sum reductions can change the order of floating-point
  additions and therefore differ slightly by ISA.
- Layer normalization needs numerically stable statistics and
  explicit behavior for near-constant rows.
- Softmax needs subtract-max stabilization to avoid exponential
  overflow, even though the result is a probability vector.
- AdamW must preserve moment estimates, bias correction and
  decoupled weight decay over a complete optimizer step.

**Exercise:** run lengths 0, 1, 7, 8, 9, 15, 16, 17 and large odd
lengths; compare scalar and native paths with documented tolerance,
including negative, extreme and nonfinite inputs.

## 4. API and memory safety

A span does not ensure enough destination capacity unless the kernel
checks it. `ValidateEqualSizes` guards matching lengths in the
public implementation. Vector intrinsics must never access a lane
outside the last element. An unaligned load is valid only for the
range's actual readable bytes; unaligned does not mean unbounded.

When adding a kernel, keep one scalar oracle, specify in-place alias
behavior, add platform-specific dispatch, and test at boundary sizes
before benchmarking.

## 5. Performance studies

Read [`tests/smoke.cpp`](../tests/smoke.cpp) and
[`tests/benchmark.cpp`](../tests/benchmark.cpp). Compare the same
workload at the same compiler optimization level, target and host,
with warmups and repeated trials. A speedup at one vector length
is not a blanket claim for every cache size, core or CPU generation.

Include accuracy error, ISA actually selected, array length, alignment,
backend thread count and memory bandwidth in evidence. Never claim
universal SIMD acceleration if the fallback scalar path is active.

*Evidence note: source-based teaching guide, no new benchmark or
CPU-feature certification performed in this pass.*
