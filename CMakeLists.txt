cmake_minimum_required(VERSION 3.30)

# === Required CMake Options ===
option(BUILD_PYBIND "Build Python bindings using pybind11" OFF)
option(BUILD_EXAMPLES "Build examples/tdcr_example applications" ON)

# === Compiler Settings ===
# set(CMAKE_C_COMPILER clang)
# set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

project(Project VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# === Output Directories ===
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# === RPATH Settings to avoid library conflicts===
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# === Subdirectories ===
add_subdirectory(lib_tdcr)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(BUILD_PYBIND)
    add_subdirectory(pybindings)
endif()

