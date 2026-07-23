module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

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
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        return CpuFeature::NEON;
#elif defined(__AVX512F__)
        return CpuFeature::AVX512;
#elif defined(__AVX2__)
        return CpuFeature::AVX2;
#else
        return CpuFeature::Scalar;
#endif
    }

    /// Highest feature set usable by this process. On Apple Silicon NEON is a
    /// mandatory architectural baseline. x86 builds use compiler runtime CPU
    /// queries and retain scalar fallback when optimized objects are absent.
    [[nodiscard]]
    inline CpuFeature DetectedFeature() noexcept
    {
#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
        return CpuFeature::NEON;
#elif defined(__x86_64__) || defined(_M_X64)
#if defined(__clang__) || defined(__GNUC__)
        if (__builtin_cpu_supports("avx512f")) return CpuFeature::AVX512;
        if (__builtin_cpu_supports("avx2")) return CpuFeature::AVX2;
#endif
        return CpuFeature::Scalar;
#else
        return CpuFeature::Scalar;
#endif
    }

    [[nodiscard]]
    constexpr std::string_view FeatureName(CpuFeature feature) noexcept
    {
        switch (feature)
        {
        case CpuFeature::Scalar: return "scalar";
        case CpuFeature::NEON: return "neon";
        case CpuFeature::AVX2: return "avx2";
        case CpuFeature::AVX512: return "avx512";
        }
        return "unknown";
    }

    inline void ValidateEqualSizes(std::size_t outputSize, std::size_t firstSize, std::size_t secondSize)
    {
        if (outputSize != firstSize || outputSize != secondSize)
        {
            throw std::invalid_argument("SIMD kernel spans must have equal sizes.");
        }
    }

    template<typename T>
    void Fill(std::span<T> out, T value)
    {
        for (T& item : out)
        {
            item = value;
        }
    }

    template<typename T>
    void Copy(std::span<T> out, std::span<const T> input)
    {
        for (std::size_t i = 0; i < out.size(); ++i)
        {
            out[i] = input[i];
        }
    }

    template<typename T>
    void Add(std::span<T> out, std::span<const T> a, std::span<const T> b)
    {
        ValidateEqualSizes(out.size(), a.size(), b.size());
        const std::size_t count = out.size();
        for (std::size_t i = 0; i < count; ++i)
        {
            out[i] = a[i] + b[i];
        }
    }

    template<typename T>
    void Sub(std::span<T> out, std::span<const T> a, std::span<const T> b)
    {
        ValidateEqualSizes(out.size(), a.size(), b.size());
        const std::size_t count = out.size();
        for (std::size_t i = 0; i < count; ++i)
        {
            out[i] = a[i] - b[i];
        }
    }

    template<typename T>
    void Mul(std::span<T> out, std::span<const T> a, std::span<const T> b)
    {
        ValidateEqualSizes(out.size(), a.size(), b.size());
        const std::size_t count = out.size();
        for (std::size_t i = 0; i < count; ++i)
        {
            out[i] = a[i] * b[i];
        }
    }

    template<typename T>
    void Scale(std::span<T> out, std::span<const T> input, T scalar)
    {
        if (out.size() != input.size()) throw std::invalid_argument("SIMD kernel spans must have equal sizes.");
        for (std::size_t i = 0; i < out.size(); ++i)
        {
            out[i] = input[i] * scalar;
        }
    }

    template<typename T>
    void Clamp(std::span<T> out, std::span<const T> input, T low, T high)
    {
        for (std::size_t i = 0; i < out.size(); ++i)
        {
            out[i] = std::clamp(input[i], low, high);
        }
    }

    template<typename T>
    void Axpy(std::span<T> y, T alpha, std::span<const T> x)
    {
        if (y.size() != x.size()) throw std::invalid_argument("SIMD kernel spans must have equal sizes.");
        const std::size_t count = y.size();
        for (std::size_t i = 0; i < count; ++i)
        {
            y[i] += alpha * x[i];
        }
    }

    template<typename T>
    [[nodiscard]]
    T Dot(std::span<const T> a, std::span<const T> b)
    {
        if (a.size() != b.size()) throw std::invalid_argument("SIMD kernel spans must have equal sizes.");
        T sum = T(0);
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }

    template<typename T>
    [[nodiscard]]
    T Sum(std::span<const T> input)
    {
        T sum = T(0);
        for (T value : input)
        {
            sum += value;
        }
        return sum;
    }

    template<typename T>
    [[nodiscard]]
    T L2Norm(std::span<const T> input)
    {
        return std::sqrt(Dot(input, input));
    }

    template<typename T>
    void ReLU(std::span<T> out, std::span<const T> input)
    {
        if (out.size() != input.size()) throw std::invalid_argument("SIMD kernel spans must have equal sizes.");
        for (std::size_t i = 0; i < out.size(); ++i)
        {
            out[i] = std::max(T(0), input[i]);
        }
    }

    template<typename T>
    void BiasReLU(std::span<T> out, std::span<const T> input, std::span<const T> bias)
    {
        if (out.size() != input.size() || bias.empty() || out.size() % bias.size() != 0)
            throw std::invalid_argument("BiasReLU requires equal tensors and a row-width bias.");
        for (std::size_t index = 0; index < out.size(); ++index)
            out[index] = std::max(T(0), input[index] + bias[index % bias.size()]);
    }

    /// Fused row-wise affine LayerNorm over contiguous [rows,width] storage.
    template<typename T>
    void LayerNormRows(
        std::span<T> out,
        std::span<const T> input,
        std::span<const T> scale,
        std::span<const T> bias,
        std::size_t rows,
        std::size_t width,
        T epsilon)
    {
        if (rows == 0 || width == 0 || out.size() != rows * width
            || input.size() != out.size() || scale.size() != width
            || bias.size() != width || !(epsilon > T(0)))
            throw std::invalid_argument(
                "LayerNormRows requires contiguous [rows,width] and affine vectors.");
        const T inverseWidth = T(1) / static_cast<T>(width);
        for (std::size_t row = 0; row < rows; ++row)
        {
            T mean = T(0);
            for (std::size_t column = 0; column < width; ++column)
                mean += input[row * width + column];
            mean *= inverseWidth;
            T variance = T(0);
            for (std::size_t column = 0; column < width; ++column)
            {
                const T delta = input[row * width + column] - mean;
                variance += delta * delta;
            }
            const T inverseStandardDeviation =
                T(1) / std::sqrt(variance * inverseWidth + epsilon);
            for (std::size_t column = 0; column < width; ++column)
                out[row * width + column] =
                    (input[row * width + column] - mean)
                    * inverseStandardDeviation * scale[column] + bias[column];
        }
    }

    template<typename T>
    void AdamW(
        std::span<T> parameters,
        std::span<T> firstMoment,
        std::span<T> secondMoment,
        std::span<const T> gradients,
        T learningRate,
        T beta1,
        T beta2,
        T inverseBias1,
        T inverseBias2,
        T epsilon,
        T weightDecay)
    {
        if (parameters.size() != gradients.size()
            || parameters.size() != firstMoment.size()
            || parameters.size() != secondMoment.size())
            throw std::invalid_argument("AdamW spans must have equal sizes.");
        for (std::size_t index = 0; index < parameters.size(); ++index)
        {
            const T gradient = gradients[index];
            firstMoment[index] = beta1 * firstMoment[index] + (T(1) - beta1) * gradient;
            secondMoment[index] =
                beta2 * secondMoment[index] + (T(1) - beta2) * gradient * gradient;
            parameters[index] -= learningRate * (
                firstMoment[index] * inverseBias1
                    / (std::sqrt(secondMoment[index] * inverseBias2) + epsilon)
                + weightDecay * parameters[index]);
        }
    }

    template<typename T>
    void Sigmoid(std::span<T> out, std::span<const T> input)
    {
        for (std::size_t i = 0; i < out.size(); ++i)
        {
            out[i] = T(1) / (T(1) + std::exp(-input[i]));
        }
    }

    template<typename T>
    void Softmax(std::span<T> out, std::span<const T> logits)
    {
        if (out.size() != logits.size()) throw std::invalid_argument("SIMD kernel spans must have equal sizes.");
        if (logits.empty())
        {
            return;
        }

        T maxValue = logits[0];
        for (T value : logits)
        {
            maxValue = std::max(maxValue, value);
        }

        T sum = T(0);
        for (std::size_t i = 0; i < logits.size(); ++i)
        {
            out[i] = std::exp(logits[i] - maxValue);
            sum += out[i];
        }

        const T invSum = sum == T(0) ? T(0) : T(1) / sum;
        for (T& value : out)
        {
            value *= invSum;
        }
    }

#if defined(__x86_64__) || defined(_M_X64)
    namespace x86_detail
    {
        __attribute__((target("avx2")))
        inline std::size_t AddAVX2(
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
        inline std::size_t AddAVX512(
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
        inline std::pair<float, std::size_t> DotAVX2(
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
        inline std::pair<float, std::size_t> DotAVX512(
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

    /// Runtime-dispatched x86 Float32 addition with scalar tail handling.
    inline void Add(
        std::span<float> out,
        std::span<const float> first,
        std::span<const float> second)
    {
        ValidateEqualSizes(out.size(), first.size(), second.size());
        std::size_t index = 0;
        const CpuFeature feature = DetectedFeature();
        if (feature == CpuFeature::AVX512)
            index = x86_detail::AddAVX512(
                out.data(), first.data(), second.data(), out.size());
        else if (feature == CpuFeature::AVX2)
            index = x86_detail::AddAVX2(
                out.data(), first.data(), second.data(), out.size());
        for (; index < out.size(); ++index)
            out[index] = first[index] + second[index];
    }

    /// Runtime-dispatched x86 Float32 dot product with Float32 accumulation.
    [[nodiscard]] inline float Dot(
        std::span<const float> first, std::span<const float> second)
    {
        if (first.size() != second.size())
            throw std::invalid_argument("SIMD kernel spans must have equal sizes.");
        std::pair<float, std::size_t> partial{ 0.0f, 0 };
        const CpuFeature feature = DetectedFeature();
        if (feature == CpuFeature::AVX512)
            partial = x86_detail::DotAVX512(
                first.data(), second.data(), first.size());
        else if (feature == CpuFeature::AVX2)
            partial = x86_detail::DotAVX2(
                first.data(), second.data(), first.size());
        for (std::size_t index = partial.second; index < first.size(); ++index)
            partial.first += first[index] * second[index];
        return partial.first;
    }
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    /// Apple Silicon NEON specializations. They preserve the scalar API and
    /// tail semantics, allowing callers to dispatch by type without platform
    /// conditionals.
    inline void Add(std::span<float> out, std::span<const float> a, std::span<const float> b)
    {
        ValidateEqualSizes(out.size(), a.size(), b.size());
        std::size_t i = 0;
        for (; i + 4 <= out.size(); i += 4)
        {
            vst1q_f32(out.data() + i, vaddq_f32(vld1q_f32(a.data() + i), vld1q_f32(b.data() + i)));
        }
        for (; i < out.size(); ++i) out[i] = a[i] + b[i];
    }

    inline void Sub(std::span<float> out, std::span<const float> a, std::span<const float> b)
    {
        ValidateEqualSizes(out.size(), a.size(), b.size());
        std::size_t i = 0;
        for (; i + 4 <= out.size(); i += 4)
            vst1q_f32(out.data() + i, vsubq_f32(vld1q_f32(a.data() + i), vld1q_f32(b.data() + i)));
        for (; i < out.size(); ++i) out[i] = a[i] - b[i];
    }

    inline void Mul(std::span<float> out, std::span<const float> a, std::span<const float> b)
    {
        ValidateEqualSizes(out.size(), a.size(), b.size());
        std::size_t i = 0;
        for (; i + 4 <= out.size(); i += 4)
            vst1q_f32(out.data() + i, vmulq_f32(vld1q_f32(a.data() + i), vld1q_f32(b.data() + i)));
        for (; i < out.size(); ++i) out[i] = a[i] * b[i];
    }

    inline void Scale(std::span<float> out, std::span<const float> input, float scalar)
    {
        if (out.size() != input.size()) throw std::invalid_argument("SIMD kernel spans must have equal sizes.");
        const float32x4_t scale = vdupq_n_f32(scalar);
        std::size_t i = 0;
        for (; i + 4 <= out.size(); i += 4)
        {
            vst1q_f32(out.data() + i, vmulq_f32(vld1q_f32(input.data() + i), scale));
        }
        for (; i < out.size(); ++i) out[i] = input[i] * scalar;
    }

    inline void Axpy(std::span<float> y, float alpha, std::span<const float> x)
    {
        if (y.size() != x.size()) throw std::invalid_argument("SIMD kernel spans must have equal sizes.");
        const float32x4_t scale = vdupq_n_f32(alpha);
        std::size_t i = 0;
        for (; i + 4 <= y.size(); i += 4)
        {
            const float32x4_t value = vfmaq_f32(vld1q_f32(y.data() + i), scale, vld1q_f32(x.data() + i));
            vst1q_f32(y.data() + i, value);
        }
        for (; i < y.size(); ++i) y[i] += alpha * x[i];
    }

    [[nodiscard]]
    inline float Dot(std::span<const float> a, std::span<const float> b)
    {
        if (a.size() != b.size()) throw std::invalid_argument("SIMD kernel spans must have equal sizes.");
        float32x4_t sum = vdupq_n_f32(0.0f);
        std::size_t i = 0;
        for (; i + 4 <= a.size(); i += 4)
        {
            sum = vfmaq_f32(sum, vld1q_f32(a.data() + i), vld1q_f32(b.data() + i));
        }
        float result = vaddvq_f32(sum);
        for (; i < a.size(); ++i) result += a[i] * b[i];
        return result;
    }

    [[nodiscard]]
    inline float Sum(std::span<const float> input)
    {
        float32x4_t sum = vdupq_n_f32(0.0f);
        std::size_t i = 0;
        for (; i + 4 <= input.size(); i += 4)
            sum = vaddq_f32(sum, vld1q_f32(input.data() + i));
        float result = vaddvq_f32(sum);
        for (; i < input.size(); ++i) result += input[i];
        return result;
    }

    inline void ReLU(std::span<float> out, std::span<const float> input)
    {
        if (out.size() != input.size()) throw std::invalid_argument("SIMD kernel spans must have equal sizes.");
        const float32x4_t zero = vdupq_n_f32(0.0f);
        std::size_t i = 0;
        for (; i + 4 <= out.size(); i += 4)
        {
            vst1q_f32(out.data() + i, vmaxq_f32(vld1q_f32(input.data() + i), zero));
        }
        for (; i < out.size(); ++i) out[i] = std::max(0.0f, input[i]);
    }
#endif
}
