cmake_minimum_required(VERSION 3.30)

# set(CMAKE_C_COMPILER clang)
# set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

project(Project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -DNDEBUG -fcatch-undefined-behavior")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set RPATH to avoid library conflicts
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# Add executable
add_executable(tdcr_sim main.cpp)

target_include_directories(tdcr_sim PRIVATE ${PROJECT_SOURCE_DIR}/include)

# BLAS
set(BLA_VENDOR OpenBLAS)
set(BLA_STATIC ON)
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    # message(STATUS "BLAS found")#include <iostream>
    include_directories(${BLAS_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "BLAS not found")
endif()

# LAPACK
find_package(LAPACK REQUIRED)
if(LAPACK_FOUND)
    # message(STATUS "LAPACK found")
    include_directories(${LAPACK_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "LAPACK not found")
endif()

# Blaze
find_package(Blaze REQUIRED)
if(Blaze_FOUND)
    # message(STATUS "Blaze found")
    # target_link_libraries(${PROJECT_NAME} INTERFACE blaze::blaze)
else()
    message(FATAL_ERROR "Blaze not found")
endif()

target_link_libraries(tdcr_sim 
    PUBLIC 
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        blaze::blaze
)

# # Sets and enables OpenMP parallel optimization in case OpenMP is available
# find_package(OpenMP)
# if(OpenMP_FOUND)
#     # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
# else()
#     message(STATUS "OpenMP NOT found")
# endif()

# ----------------------- pybind11 Python Module -----------------------
find_package(pybind11 REQUIRED)
if(pybind11_FOUND)
    message(STATUS "pybind11 found")
    # Add Python module (header-only binding logic in pybindings.hpp)
    pybind11_add_module(tdcr_physics MODULE ${PROJECT_SOURCE_DIR}/src/pybindings.cpp)
    target_include_directories(tdcr_physics PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/blaze
    )
    set_target_properties(tdcr_physics PROPERTIES PREFIX "" SUFFIX ".so")

else()
    message(FATAL_ERROR "pybind11 not found")
endif()


find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(Python3_FOUND)
    # Install the Python module to the site-packages directory of the current Python environment
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Python module will be installed to: ${PYTHON_SITE_PACKAGES}")
    install(TARGETS tdcr_physics LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}")
    install(CODE "message(STATUS \"Installed tdcr_physics module to: ${PYTHON_SITE_PACKAGES}/tdcr_physics.so\")")
else()
    message(FATAL_ERROR "Python3 not found")
endif()