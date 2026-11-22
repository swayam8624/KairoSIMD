# Advanced laboratory: vector kernels, CPU dispatch, and numerical contracts

Read [the C++23 public module](../src/KairoSIMD.cppm) first, then [the platform-specific implementation](../src/KairoSIMDPlatform.cpp), [smoke tests](../tests/smoke.cpp), and [benchmark](../tests/benchmark.cpp). This lesson complements the [existing scalar-parity walkthrough](SIMD_SCALAR_PARITY_GUIDE.md). It documents the *current code*, not an idealized SIMD library.

## 1. Exactly where the public API ends

`import Kairo.SIMD;` exposes `kairo::simd` functions over `std::span`. The generic `template<typename T>` versions in `KairoSIMD.cppm` are straightforward scalar reference operations. The Float32 *non-template* overload declarations for `Add`, `Sub`, `Mul`, `Scale`, `Axpy`, `Dot`, `Sum`, and `ReLU` have their definitions in `KairoSIMDPlatform.cpp`.

That distinction matters: calling `Add(std::span<float>, std::span<const float>, std::span<const float>)` selects the Float32 overload, while `Add` on another suitable element type selects the generic template. A C++23 module **interface unit** exports the API and scalar templates; a **module implementation unit** hides architecture-specific intrinsics from importers.

`std::span<T>` is a *non-owning*, contiguous view. It does not allocate, extend lifetime, or validate that the backing allocation remains alive. The tensor layer must translate its own shape/stride/ownership constraints into valid spans.

## 2. ISA dispatch is a two-part contract

`BaselineFeature()` reports architecture macros fixed at compile time. `DetectedFeature()` reports a process-usable feature according to the runtime checks implemented for that platform. They answer different questions.

The current ARM path uses NEON when compiled with ARM NEON support. On supported Clang/GCC x86 builds, `Add` and `Dot` dispatch to target-specific AVX2 or AVX-512 helpers, with scalar tails. Other Float32 operations shown in the platform file use NEON when enabled or scalar code; they do **not** automatically gain AVX2 merely because `DetectedFeature()` says AVX2.

A vector operation processes a fixed lane count (four Float32 lanes for the NEON shown here, eight for AVX2 add, sixteen for AVX-512 add), then processes the leftover suffix in a scalar loop. The x86 helpers use unaligned `loadu/storeu` intrinsics. This permits ordinary valid Float32 pointers without assuming 32- or 64-byte alignment, but does not fix overlapping input/output views.

**Code-review hazard:** x86 `DotAVX2` is compiled with `target("avx2,fma")`, and `DotAVX512` with `target("avx512f,fma")`; the runtime dispatch currently asks for `avx2` or `avx512f` without an explicit `fma` check. Before claiming universal CPU compatibility, establish that the selected host also supports FMA or strengthen feature gating and test on representative hardware. An unsupported instruction is not an exception that C++ can safely recover from.

## 3. Overlap, sizes, and caller-provided invariants

`ValidateEqualSizes` guards the three spans in `Add/Sub/Mul`, and the Float32 `Scale/Axpy/Dot/ReLU` overloads check sizes. But the **generic** `Copy` and `Clamp` implementations as currently written do *not* check that `out.size()` is bounded by input size. Treat equal-size spans as a caller prerequisite for these APIs until their validation is hardened.

A pointwise operation is often safe when an output is exactly the same underlying span as an input and each index is read before being overwritten. **Partial overlap with an offset** is a different question: writing `out[i]` can corrupt a future `input[i+1]`. Generic `Copy` is not a `memmove` replacement. Tests must cover exact in-place reuse separately from partially overlapping spans.

```cpp
import Kairo.SIMD;
#include <array>
#include <span>

std::array<float, 4> a{1.f, 2.f, 3.f, 4.f};
std::array<float, 4> b{4.f, 3.f, 2.f, 1.f};
std::array<float, 4> out{};

kairo::simd::Add(
    std::span<float>{out},
    std::span<const float>{a},
    std::span<const float>{b});
// Every output element should equal 5.0f.
// This demonstrates the public Float32 overload, not performance.
```

## 4. Numerical equivalence is not bitwise identity

`Dot` sums products. The scalar implementation adds one product at a time; NEON and x86 helpers accumulate products across SIMD lanes and then horizontally reduce. Some paths explicitly use fused multiply-add. Reassociation and fused rounding can change low bits even when both results are mathematically correct.

For `Softmax`, the generic implementation first subtracts the largest logit before exponentiating. This improves stability compared with exponentiating raw large values, but NaN, positive infinity, negative infinity, and output/input aliasing still require a declared numerical policy. Test: finite uniform logits, an extreme finite spread, empty vectors, and non-finite values separately. Do not treat a normal finite-input test as proof of all input categories.

`LayerNormRows` validates nonzero shape, multiplication overflow, contiguous sizes, affine vector widths, and positive epsilon. Its row-wise mean and variance depend on summation order and the representable range of `T`. `AdamW` mutates parameters and both moment arrays, with bias-correction factors supplied by the caller; the function currently validates span sizes, **not** the optimizer's scalar hyperparameter ranges.

For numerical-parity testing, use both absolute and relative tolerances, such as `abs(actual - reference) <= atol + rtol * abs(reference)`, and handle NaNs deliberately outside that comparison.

## 5. SIMD and threading are orthogonal axes

A KairoMath operation can ask KairoScheduler to partition output indices into nonoverlapping ranges, then call a KairoSIMD function on a local span **within each range**. The scheduler decides which CPU thread executes a chunk; SIMD decides how one thread processes the elements in that chunk.

This composition requires disjoint writes, valid source lifetimes, compatible span lengths, and a sane chunk size. Thread count times vector width is not a speedup formula: memory bandwidth, cache traffic, temporary allocations, and small-task overhead often dominate. GPU kernels are a separate command-and-memory model owned by KairoGPU, not a SIMD mode toggle.

## 6. Verification sequence

1. Read `KairoSIMD.cppm` and label each API as generic template, Float32 platform overload, or metadata/validation function.
2. Trace `Add<float>` and Float32 `Dot` to the actual platform implementation. Count vector-width iterations and scalar-tail elements for sizes 0, 1, 3, 4, 7, 8, 15, 16, and 17.
3. Add negative tests for input/output length mismatches, aliasing, and shape overflow. Specifically investigate the unchecked generic `Copy` and `Clamp` boundaries rather than assuming they reject mismatches.
4. Audit `fma` feature availability against the x86 Dot target attributes.
5. Test finite and non-finite inputs for reductions, activation functions, softmax, LayerNormRows and AdamW; document the accepted domain and tolerance.
6. Run host-specific CMake/CTest, ASan or TSan as appropriate, and the benchmark. Capture compiler/CPU/OS, exact commit, scalar error, problem size, warmup, repetitions, median and distribution; do not present a single `DetectedFeature()` value as measured speedup.

**Evidence status:** source-backed teaching and review findings only; no compilation, host feature validation, or benchmark has been executed in this documentation pass.
