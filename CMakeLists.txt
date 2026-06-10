cmake_minimum_required(VERSION 3.28)
project(KairoSIMD LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(KairoSIMD)
target_sources(KairoSIMD
    PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
        FILES src/KairoSIMD.cppm
)

add_executable(KairoSIMDSmoke tests/smoke.cpp)
target_link_libraries(KairoSIMDSmoke PRIVATE KairoSIMD)
