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
    return 0;
}
