cmake_minimum_required(VERSION 3.25...3.31)
project(libmoran
        VERSION 0.1.0
        DESCRIPTION "Moran process fixation probability library"
        LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(MORAN_BUILD_TESTS "Build C++ unit tests" ON)
option(MORAN_BUILD_PYTHON "Build Python bindings" OFF)
option(MORAN_USE_OPENMP "Enable OpenMP" ON)
option(MORAN_USE_SIMD "Enable SIMD (march=native)" ON)

# ---------------------------------------------------------------------------
# Build type defaults
# ---------------------------------------------------------------------------
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif ()

# ---------------------------------------------------------------------------
# Compiler warnings
# ---------------------------------------------------------------------------
add_library(moran_compiler_flags INTERFACE)
target_compile_options(moran_compiler_flags INTERFACE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
        -Wno-sign-conversion
        >
        $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<BOOL:${MORAN_USE_SIMD}>>:
        -march=native
        >
)

# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------
if (MORAN_USE_OPENMP)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: version ${OpenMP_CXX_VERSION}")
    else ()
        message(WARNING "OpenMP not found, parallel MC disabled")
    endif ()
endif ()

# ---------------------------------------------------------------------------
# Main library (header-only, INTERFACE)
# ---------------------------------------------------------------------------
add_library(moran INTERFACE)
add_library(moran::moran ALIAS moran)

target_compile_features(moran INTERFACE cxx_std_23)
set(CMAKE_CXX_EXTENSIONS OFF)

target_include_directories(moran INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

file(GLOB_RECURSE MORAN_HEADERS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/include/*.hpp"
)
target_sources(moran INTERFACE
        FILE_SET HEADERS
        BASE_DIRS "${PROJECT_SOURCE_DIR}/include"
        FILES ${MORAN_HEADERS}
)

add_custom_target(moran_headers SOURCES ${MORAN_HEADERS})
target_link_libraries(moran INTERFACE
        moran_compiler_flags
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

# ---------------------------------------------------------------------------
# Tests (Google Test)
# ---------------------------------------------------------------------------
if (MORAN_BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if (NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG v1.15.2
        )
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif ()

    add_executable(moran_tests
            tests/test_numeric.cpp
            tests/test_wellmixed.cpp
            tests/test_graph.cpp
            tests/test_graph_algorithms.cpp
    )
    target_link_libraries(moran_tests PRIVATE
            moran::moran
            GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(moran_tests)
endif ()

# ---------------------------------------------------------------------------
# Python bindings (pybind11)
# ---------------------------------------------------------------------------
if (MORAN_BUILD_PYTHON)
    if (NOT Python3_EXECUTABLE AND EXISTS "${PROJECT_SOURCE_DIR}/.venv/bin/python")
        set(Python3_EXECUTABLE "${PROJECT_SOURCE_DIR}/.venv/bin/python"
                CACHE FILEPATH "Python interpreter (auto-detected from .venv)" FORCE)
        message(STATUS "Auto-detected venv Python: ${Python3_EXECUTABLE}")
    endif ()

    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

    if (NOT pybind11_DIR)
        execute_process(
                COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
                OUTPUT_VARIABLE _pybind11_cmake_dir
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _pybind11_ret
        )
        if (_pybind11_ret EQUAL 0 AND EXISTS "${_pybind11_cmake_dir}")
            set(pybind11_DIR "${_pybind11_cmake_dir}" CACHE PATH "pybind11 cmake dir" FORCE)
            message(STATUS "Auto-detected pybind11: ${pybind11_DIR}")
        endif ()
    endif ()

    find_package(pybind11 REQUIRED)

    pybind11_add_module(_moran
            python/pymoran.cpp
            python/bind_graph.cpp
            python/bind_algorithms.cpp
            python/helpers.hpp
    )
    target_link_libraries(_moran PRIVATE moran::moran)

    install(TARGETS _moran DESTINATION moran)

    if (NOT SKBUILD)
        add_custom_command(TARGET _moran POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                $<TARGET_FILE:_moran>
                ${PROJECT_SOURCE_DIR}/python/moran/$<TARGET_FILE_NAME:_moran>
                COMMENT "Copying _moran extension into python/moran/"
        )
    endif ()
endif ()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS moran moran_compiler_flags
        EXPORT moranTargets
        FILE_SET HEADERS DESTINATION include
)
install(EXPORT moranTargets
        FILE moranTargets.cmake
        NAMESPACE moran::
        DESTINATION lib/cmake/moran
)
