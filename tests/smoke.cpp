import Kairo.SIMD;

#include <array>
#include <cassert>
#include <span>
#include <string_view>

int main()
{
    std::array<float, 3> a{ 1.0f, 2.0f, 3.0f };
    std::array<float, 3> b{ 4.0f, 5.0f, 6.0f };
    std::array<float, 3> out{};
    kairo::simd::Add<float>(out, a, b);
    assert(out[0] == 5.0f && out[2] == 9.0f);
    assert(kairo::simd::Dot<float>(a, b) == 32.0f);
    kairo::simd::Scale<float>(out, a, 2.0f);
    assert(out[1] == 4.0f);
    kairo::simd::Softmax<float>(out, a);
    assert(out[0] < out[1] && out[1] < out[2]);

    std::array<float, 9> vectorA{ -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f };
    std::array<float, 9> vectorB{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    std::array<float, 9> vectorOut{};
    kairo::simd::Add(vectorOut, vectorA, vectorB);
    assert(vectorOut[0] == -3.0f && vectorOut[8] == 13.0f);
    kairo::simd::Scale(vectorOut, vectorB, 0.5f);
    assert(vectorOut[0] == 0.5f && vectorOut[8] == 4.5f);
    kairo::simd::ReLU(vectorOut, vectorA);
    assert(vectorOut[0] == 0.0f && vectorOut[8] == 4.0f);
    assert(kairo::simd::Dot(vectorA, vectorB) == 60.0f);
    assert(kairo::simd::Sum(std::span<const float>(vectorB)) == 45.0f);

    std::array<float, 3> bias{ 1.0f, -1.0f, 0.5f };
    kairo::simd::BiasReLU<float>(vectorOut, vectorA, bias);
    assert(vectorOut[0] == 0.0f && vectorOut[5] == 1.5f && vectorOut[8] == 4.5f);

    std::array<float, 3> parameters{ 1.0f, 2.0f, 3.0f };
    std::array<float, 3> first{};
    std::array<float, 3> second{};
    std::array<float, 3> gradients{ 0.1f, -0.2f, 0.3f };
    kairo::simd::AdamW<float>(
        parameters, first, second, gradients,
        0.01f, 0.9f, 0.999f, 10.0f, 1000.0f, 1e-8f, 0.01f);
    assert(parameters[0] < 1.0f && parameters[1] > 2.0f && parameters[2] < 3.0f);
    assert(kairo::simd::FeatureName(kairo::simd::DetectedFeature()) != "unknown");
    return 0;
}
