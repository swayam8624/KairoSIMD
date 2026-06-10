module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>

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
        const std::size_t count = out.size();
        for (std::size_t i = 0; i < count; ++i)
        {
            out[i] = a[i] + b[i];
        }
    }

    template<typename T>
    void Sub(std::span<T> out, std::span<const T> a, std::span<const T> b)
    {
        const std::size_t count = out.size();
        for (std::size_t i = 0; i < count; ++i)
        {
            out[i] = a[i] - b[i];
        }
    }

    template<typename T>
    void Mul(std::span<T> out, std::span<const T> a, std::span<const T> b)
    {
        const std::size_t count = out.size();
        for (std::size_t i = 0; i < count; ++i)
        {
            out[i] = a[i] * b[i];
        }
    }

    template<typename T>
    void Scale(std::span<T> out, std::span<const T> input, T scalar)
    {
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
        for (std::size_t i = 0; i < out.size(); ++i)
        {
            out[i] = std::max(T(0), input[i]);
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
}
