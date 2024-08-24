#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include "utils.h"
#include <cuda_fp16.h>
#include <iostream>

// CUDA 核函数：将 float 数组转换为 __half 数组
__global__ void floatToHalfKernel(const float* d_floatArray, __half* d_halfArray, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_halfArray[idx] = __float2half(d_floatArray[idx]);
    }
}

void float2Half(const float* d_floatArray, __half* d_halfArray, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    floatToHalfKernel<<<gridSize, blockSize>>>(d_floatArray, d_halfArray, size);

    // 检查 CUDA 核函数的执行是否有错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 等待设备完成，并检查是否有错误
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronization error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

__global__ void halfToFloatKernel(const __half* d_halfArray, float* d_floatArray, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_floatArray[idx] = __half2float(d_halfArray[idx]);
    }
}

void half2Float(const __half* d_halfArray, float* d_floatArray, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    halfToFloatKernel<<<gridSize, blockSize>>>(d_halfArray, d_floatArray, size);

    // 检查 CUDA 核函数的执行是否有错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 等待设备完成，并检查是否有错误
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronization error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

#endif


