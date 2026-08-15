cmake_minimum_required(VERSION 3.20)
project(MoE-Nexus-CPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MOE_NEXUS_BUILD_TESTS "Build C++ tests" ON)
option(MOE_NEXUS_BUILD_BENCHMARK "Build C++ benchmarks" ON)
option(MOE_NEXUS_BUILD_PYTHON "Build Python bindings" ON)

# Find pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, disabling Python bindings")
    set(MOE_NEXUS_BUILD_PYTHON OFF)
endif()

# Core library
add_library(moe_nexus_core STATIC
    src/tokenizer.cpp
    src/router.cpp
    src/load_balancer.cpp
    src/model.cpp
    src/engine.cpp
)

target_include_directories(moe_nexus_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Python bindings
if(MOE_NEXUS_BUILD_PYTHON AND pybind11_FOUND)
    pybind11_add_module(moe_nexus_core_bindings src/bindings.cpp)
    target_link_libraries(moe_nexus_core_bindings PRIVATE moe_nexus_core)
    set_target_properties(moe_nexus_core_bindings PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/moe_nexus/cpp
    )
endif()

# Benchmark
if(MOE_NEXUS_BUILD_BENCHMARK)
    add_executable(moe_nexus_benchmark benchmark/main.cpp)
    target_link_libraries(moe_nexus_benchmark PRIVATE moe_nexus_core)
endif()

# Tests
if(MOE_NEXUS_BUILD_TESTS)
    enable_testing()
    add_executable(moe_nexus_tests test/tests.cpp)
    target_link_libraries(moe_nexus_tests PRIVATE moe_nexus_core)
    add_test(NAME moe_nexus_core_tests COMMAND moe_nexus_tests)
endif()
