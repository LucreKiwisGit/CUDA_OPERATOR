cmake_minimum_required(VERSION 3.0)

project(cuda_operators VERSION 1.0 LANGUAGES CXX CUDA)

add_definitions(-std=c++14)

set(TARGET_NAME "cuda_operators_test")

set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75")

option(CUDA_USE_STATIC_CUDA_RUNTIME "Use static cuda runtime" OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

find_package(OpenMP REQUIRED)
include_directories(${OpenMP_INCLUDE_DIRS})

# 设置 cuDNN 的路径，假设 cuDNN 已经安装在系统的标准路径中
# 如果没有安装在标准路径中，可以手动设置 CUDNN_INCLUDE_DIR 和 CUDNN_LIBRARY
find_path(CUDNN_INCLUDE_DIR cudnn.h
  PATHS /usr/local/cuda/include
)
find_library(CUDNN_LIBRARY cudnn
  PATHS /usr/local/cuda/lib64
)

# 如果找不到 cudnn.h 或 libcudnn.so，抛出错误
if(NOT CUDNN_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find cuDNN include directory")
endif()

if(NOT CUDNN_LIBRARY)
  message(FATAL_ERROR "Could not find cuDNN library")
endif()

include_directories(${CUDNN_INCLUDE_DIR})



FILE(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/test/*.cpp ${PROJECT_SOURCE_DIR}/test/*.cu ${PROJECT_SOURCE_DIR}/kernel/*.cpp 
                    ${PROJECT_SOURCE_DIR}/kernels/implGEMM.cu ${PROJECT_SOURCE_DIR}/kernels/directConv2d.cc ${PROJECT_SOURCE_DIR}/kernels/GEMM.cu
                    ${PROJECT_SOURCE_DIR}/cudnn/*.cc
                    ${PROJECT_SOURCE_DIR}/include/*.h 
                    ${PROJECT_SOURCE_DIR}/utils/*.cpp
                    ${PROJECT_SOURCE_DIR}/main.cu)

# MESSAGE(FATAL_ERROR "SRC_FILES: ${SRC_FILES}")



add_executable(${TARGET_NAME} ${SRC_FILES})
target_link_libraries(${TARGET_NAME} cuda cudart ${OpenMP_CXX_LIBRARIES} cublas ${CUDNN_LIBRARY})

target_compile_options(${TARGET_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-G>  # CUDA 编译选项示例
)

add_definitions(-pthread -g)