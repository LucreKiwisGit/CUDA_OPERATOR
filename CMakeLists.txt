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


FILE(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/test/*.cpp ${PROJECT_SOURCE_DIR}/test/*.cu ${PROJECT_SOURCE_DIR}/kernel/*.cpp 
                    ${PROJECT_SOURCE_DIR}/kernels/implGEMM.cu ${PROJECT_SOURCE_DIR}/kernels/directConv2d.cc ${PROJECT_SOURCE_DIR}/kernels/GEMM.cu
                    ${PROJECT_SOURCE_DIR}/include/*.h 
                    ${PROJECT_SOURCE_DIR}/utils/*.cpp
                    ${PROJECT_SOURCE_DIR}/main.cu)

# MESSAGE(FATAL_ERROR "SRC_FILES: ${SRC_FILES}")



add_executable(${TARGET_NAME} ${SRC_FILES})
target_link_libraries(${TARGET_NAME} cuda cudart ${OpenMP_CXX_LIBRARIES} cublas)

# target_compile_options(${TARGET_NAME} PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:-G>  # CUDA 编译选项示例
# )

add_definitions(-pthread -g)