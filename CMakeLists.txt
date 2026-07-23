cmake_minimum_required(VERSION 3.28)
project(KairoSIMD LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KAIRO_SIMD_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(KAIRO_SIMD_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
if(KAIRO_SIMD_ENABLE_ASAN AND KAIRO_SIMD_ENABLE_TSAN)
    message(FATAL_ERROR "ASan and TSan must use separate build directories")
endif()
if(KAIRO_SIMD_ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()
if(KAIRO_SIMD_ENABLE_TSAN)
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
    add_link_options(-fsanitize=thread)
endif()

add_library(KairoSIMD)
target_sources(KairoSIMD
    PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
        FILES src/KairoSIMD.cppm
)

add_executable(KairoSIMDSmoke tests/smoke.cpp)
target_link_libraries(KairoSIMDSmoke PRIVATE KairoSIMD)
add_executable(KairoSIMDBenchmark tests/benchmark.cpp)
target_link_libraries(KairoSIMDBenchmark PRIVATE KairoSIMD)

enable_testing()
add_test(NAME KairoSIMDSmoke COMMAND KairoSIMDSmoke)
add_test(
    NAME KairoSIMDBenchmark
    COMMAND KairoSIMDBenchmark ${CMAKE_CURRENT_BINARY_DIR}/kairo-simd-benchmark.json
)
