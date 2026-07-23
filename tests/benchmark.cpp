import Kairo.SIMD;

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv)
{
    constexpr std::size_t elements = 1u << 20;
    constexpr std::size_t warmups = 3;
    constexpr std::size_t iterations = 15;
    std::vector<float> first(elements);
    std::vector<float> second(elements);
    std::vector<float> output(elements);
    for (std::size_t index = 0; index < elements; ++index)
    {
        first[index] = static_cast<float>(index % 127) * 0.01f;
        second[index] = static_cast<float>(index % 61) * -0.02f;
    }
    for (std::size_t iteration = 0; iteration < warmups; ++iteration)
        kairo::simd::Add<float>(output, first, second);

    std::vector<double> milliseconds;
    milliseconds.reserve(iterations);
    for (std::size_t iteration = 0; iteration < iterations; ++iteration)
    {
        const auto begin = std::chrono::steady_clock::now();
        kairo::simd::Add<float>(output, first, second);
        const auto end = std::chrono::steady_clock::now();
        milliseconds.push_back(
            std::chrono::duration<double, std::milli>(end - begin).count());
    }
    std::sort(milliseconds.begin(), milliseconds.end());
    double maximumError = 0.0;
    for (std::size_t index = 0; index < elements; ++index)
        maximumError = std::max(maximumError, std::abs(
            static_cast<double>(output[index])
            - static_cast<double>(first[index] + second[index])));
    const double medianMs = milliseconds[milliseconds.size() / 2];
    const double elementsPerSecond =
        static_cast<double>(elements) / (medianMs / 1000.0);
    if (!std::isfinite(elementsPerSecond) || elementsPerSecond <= 0.0
        || maximumError != 0.0)
        return 1;
    const double minimumThroughput =
        argc > 2 ? std::stod(argv[2]) : 0.0;
    if (minimumThroughput < 0.0
        || (minimumThroughput > 0.0 && elementsPerSecond < minimumThroughput))
        return 3;

    const std::string json =
        "{\n"
        "  \"schema\": \"kairo.simd.benchmark.v1\",\n"
        "  \"operation\": \"add_f32\",\n"
        "  \"backend\": \"" +
        std::string(kairo::simd::FeatureName(kairo::simd::DetectedFeature())) +
        "\",\n"
        "  \"elements\": " + std::to_string(elements) + ",\n"
        "  \"warmups\": " + std::to_string(warmups) + ",\n"
        "  \"iterations\": " + std::to_string(iterations) + ",\n"
        "  \"median_ms\": " + std::to_string(medianMs) + ",\n"
        "  \"elements_per_second\": " + std::to_string(elementsPerSecond) + ",\n"
        "  \"minimum_elements_per_second\": "
            + std::to_string(minimumThroughput) + ",\n"
        "  \"maximum_absolute_error\": " + std::to_string(maximumError) + "\n"
        "}\n";
    if (argc > 1)
    {
        std::ofstream outputFile(argv[1], std::ios::binary | std::ios::trunc);
        if (!outputFile) return 2;
        outputFile << json;
        if (!outputFile) return 2;
    }
    std::cout << json;
    return 0;
}
