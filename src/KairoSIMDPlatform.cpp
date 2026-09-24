module;

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

module Kairo.SIMD;

namespace kairo::simd
{
#if (defined(__clang__) || defined(__GNUC__)) && (defined(__x86_64__) || defined(_M_X64))
    namespace x86_detail
    {
        __attribute__((target("avx2")))
        std::size_t AddAVX2(
            float* out, const float* first, const float* second, std::size_t count)
        {
            std::size_t index = 0;
            for (; index + 8 <= count; index += 8)
                _mm256_storeu_ps(
                    out + index,
                    _mm256_add_ps(
                        _mm256_loadu_ps(first + index),
                        _mm256_loadu_ps(second + index)));
            return index;
        }

        __attribute__((target("avx512f")))
        std::size_t AddAVX512(
            float* out, const float* first, const float* second, std::size_t count)
        {
            std::size_t index = 0;
            for (; index + 16 <= count; index += 16)
                _mm512_storeu_ps(
                    out + index,
                    _mm512_add_ps(
                        _mm512_loadu_ps(first + index),
                        _mm512_loadu_ps(second + index)));
            return index;
        }

        __attribute__((target("avx2,fma")))
        std::pair<float, std::size_t> DotAVX2(
            const float* first, const float* second, std::size_t count)
        {
            __m256 sum = _mm256_setzero_ps();
            std::size_t index = 0;
            for (; index + 8 <= count; index += 8)
                sum = _mm256_fmadd_ps(
                    _mm256_loadu_ps(first + index),
                    _mm256_loadu_ps(second + index), sum);
            alignas(32) float lanes[8];
            _mm256_store_ps(lanes, sum);
            float reduced = 0.0f;
            for (float lane : lanes) reduced += lane;
            return { reduced, index };
        }

        __attribute__((target("avx512f,fma")))
        std::pair<float, std::size_t> DotAVX512(
            const float* first, const float* second, std::size_t count)
        {
            __m512 sum = _mm512_setzero_ps();
            std::size_t index = 0;
            for (; index + 16 <= count; index += 16)
                sum = _mm512_fmadd_ps(
                    _mm512_loadu_ps(first + index),
                    _mm512_loadu_ps(second + index), sum);
            alignas(64) float lanes[16];
            _mm512_store_ps(lanes, sum);
            float reduced = 0.0f;
            for (float lane : lanes) reduced += lane;
            return { reduced, index };
        }
    }
#endif

    void Add(std::span<float> out, std::span<const float> a, std::span<const float> b)
    {
        ValidateEqualSizes(out.size(), a.size(), b.size());
        std::size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        for (; i + 4 <= out.size(); i += 4)
            vst1q_f32(out.data() + i,
                vaddq_f32(vld1q_f32(a.data() + i), vld1q_f32(b.data() + i)));
#elif (defined(__clang__) || defined(__GNUC__)) && (defined(__x86_64__) || defined(_M_X64))
        const CpuFeature feature = DetectedFeature();
        if (feature == CpuFeature::AVX512)
            i = x86_detail::AddAVX512(out.data(), a.data(), b.data(), out.size());
        else if (feature == CpuFeature::AVX2)
            i = x86_detail::AddAVX2(out.data(), a.data(), b.data(), out.size());
#endif
        for (; i < out.size(); ++i) out[i] = a[i] + b[i];
    }

    void Sub(std::span<float> out, std::span<const float> a, std::span<const float> b)
    {
        ValidateEqualSizes(out.size(), a.size(), b.size());
        std::size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        for (; i + 4 <= out.size(); i += 4)
            vst1q_f32(out.data() + i,
                vsubq_f32(vld1q_f32(a.data() + i), vld1q_f32(b.data() + i)));
#endif
        for (; i < out.size(); ++i) out[i] = a[i] - b[i];
    }

    void Mul(std::span<float> out, std::span<const float> a, std::span<const float> b)
    {
        ValidateEqualSizes(out.size(), a.size(), b.size());
        std::size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        for (; i + 4 <= out.size(); i += 4)
            vst1q_f32(out.data() + i,
                vmulq_f32(vld1q_f32(a.data() + i), vld1q_f32(b.data() + i)));
#endif
        for (; i < out.size(); ++i) out[i] = a[i] * b[i];
    }

    void Scale(std::span<float> out, std::span<const float> input, float scalar)
    {
        if (out.size() != input.size())
            throw std::invalid_argument("SIMD kernel spans must have equal sizes.");
        std::size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        const float32x4_t scale = vdupq_n_f32(scalar);
        for (; i + 4 <= out.size(); i += 4)
            vst1q_f32(out.data() + i,
                vmulq_f32(vld1q_f32(input.data() + i), scale));
#endif
        for (; i < out.size(); ++i) out[i] = input[i] * scalar;
    }

    void Axpy(std::span<float> y, float alpha, std::span<const float> x)
    {
        if (y.size() != x.size())
            throw std::invalid_argument("SIMD kernel spans must have equal sizes.");
        std::size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        const float32x4_t scale = vdupq_n_f32(alpha);
        for (; i + 4 <= y.size(); i += 4)
        {
            const float32x4_t value =
                vfmaq_f32(vld1q_f32(y.data() + i), scale, vld1q_f32(x.data() + i));
            vst1q_f32(y.data() + i, value);
        }
#endif
        for (; i < y.size(); ++i) y[i] += alpha * x[i];
    }

    float Dot(std::span<const float> a, std::span<const float> b)
    {
        if (a.size() != b.size())
            throw std::invalid_argument("SIMD kernel spans must have equal sizes.");
        std::size_t i = 0;
        float result = 0.0f;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        float32x4_t sum = vdupq_n_f32(0.0f);
        for (; i + 4 <= a.size(); i += 4)
            sum = vfmaq_f32(sum, vld1q_f32(a.data() + i), vld1q_f32(b.data() + i));
        result = vaddvq_f32(sum);
#elif (defined(__clang__) || defined(__GNUC__)) && (defined(__x86_64__) || defined(_M_X64))
        std::pair<float, std::size_t> partial{ 0.0f, 0u };
        const CpuFeature feature = DetectedFeature();
        if (feature == CpuFeature::AVX512)
            partial = x86_detail::DotAVX512(a.data(), b.data(), a.size());
        else if (feature == CpuFeature::AVX2)
            partial = x86_detail::DotAVX2(a.data(), b.data(), a.size());
        result = partial.first;
        i = partial.second;
#endif
        for (; i < a.size(); ++i) result += a[i] * b[i];
        return result;
    }

    float Sum(std::span<const float> input)
    {
        std::size_t i = 0;
        float result = 0.0f;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        float32x4_t sum = vdupq_n_f32(0.0f);
        for (; i + 4 <= input.size(); i += 4)
            sum = vaddq_f32(sum, vld1q_f32(input.data() + i));
        result = vaddvq_f32(sum);
#endif
        for (; i < input.size(); ++i) result += input[i];
        return result;
    }

    void ReLU(std::span<float> out, std::span<const float> input)
    {
        if (out.size() != input.size())
            throw std::invalid_argument("SIMD kernel spans must have equal sizes.");
        std::size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        const float32x4_t zero = vdupq_n_f32(0.0f);
        for (; i + 4 <= out.size(); i += 4)
            vst1q_f32(out.data() + i,
                vmaxq_f32(vld1q_f32(input.data() + i), zero));
#endif
        for (; i < out.size(); ++i) out[i] = std::max(0.0f, input[i]);
    }
}
