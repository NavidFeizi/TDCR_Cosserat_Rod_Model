project(TDCRLib LANGUAGES CXX)

add_library(TDCRLib INTERFACE)

target_include_directories(TDCRLib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

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
else()
    message(FATAL_ERROR "Blaze not found")
endif()

target_link_libraries(${PROJECT_NAME} 
    INTERFACE 
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
