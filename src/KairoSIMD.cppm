module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

export module Kairo.SIMD;

export namespace kairo::simd
{
    enum class CpuFeature
    {
        Scalar,
        NEON,
        AVX2,
        AVX512
    };

    [[nodiscard]]
    constexpr CpuFeature BaselineFeature() noexcept
    {

    // Float32 overloads have out-of-line definitions in a private module
    // implementation unit. Keeping architecture intrinsics out of the exported
    // interface prevents Clang module BMIs from exposing TU-local NEON/AVX
    // entities while preserving the same public API and scalar tail semantics.
    void Add(std::span<float> out, std::span<const float> a, std::span<const float> b);
    void Sub(std::span<float> out, std::span<const float> a, std::span<const float> b);
    void Mul(std::span<float> out, std::span<const float> a, std::span<const float> b);
    void Scale(std::span<float> out, std::span<const float> input, float scalar);
    void Axpy(std::span<float> y, float alpha, std::span<const float> x);
    [[nodiscard]] float Dot(std::span<const float> a, std::span<const float> b);
    [[nodiscard]] float Sum(std::span<const float> input);
    void ReLU(std::span<float> out, std::span<const float> input);

}
