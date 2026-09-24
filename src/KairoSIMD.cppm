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

    // Float32 overloads are defined in a private module implementation unit.
    // Architecture intrinsic headers and bodies stay outside the exported BMI,
    // avoiding TU-local intrinsic exposure under strict Clang module diagnostics.
    void Add(std::span<float> out, std::span<const float> a, std::span<const float> b);
    void Sub(std::span<float> out, std::span<const float> a, std::span<const float> b);
    void Mul(std::span<float> out, std::span<const float> a, std::span<const float> b);
    void Scale(std::span<float> out, std::span<const float> input, float scalar);
    void Axpy(std::span<float> y, float alpha, std::span<const float> x);
    [[nodiscard]] float Dot(std::span<const float> a, std::span<const float> b);
    [[nodiscard]] float Sum(std::span<const float> input);
    void ReLU(std::span<float> out, std::span<const float> input);

}