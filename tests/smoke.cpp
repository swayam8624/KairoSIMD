import Kairo.SIMD;

#include <array>
#include <cassert>
#include <span>

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
    kairo::simd::Add<float>(vectorOut, vectorA, vectorB);
    assert(vectorOut[0] == -3.0f && vectorOut[8] == 13.0f);
    kairo::simd::Scale<float>(vectorOut, vectorB, 0.5f);
    assert(vectorOut[0] == 0.5f && vectorOut[8] == 4.5f);
    kairo::simd::ReLU<float>(vectorOut, vectorA);
    assert(vectorOut[0] == 0.0f && vectorOut[8] == 4.0f);
    assert(kairo::simd::Dot<float>(vectorA, vectorB) == 60.0f);
    return 0;
}
