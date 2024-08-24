#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>
#include <memory>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "operators.h"

// 数据枚举类型
enum DataType{
    FLOAT32,
    FLOAT16,
    INT8,
    INT32
};

enum Operator{
    GEMM,
    ImplicitGEMM,
    Conv2d,
    Im2colGEMM
};

typedef struct args_t
{
    Operator op_name;    // 算子名称
    DataType op_type;    // 数据类型
} args_t;


// 自定义删除器， 用于释放CUDA内存
struct CudaDeleter {
    template <typename T>
    void operator()(T *ptr) const {
        cudaFree(ptr);
    }
};

template <typename T>
using unique_ptr_cuda = std::unique_ptr<T, CudaDeleter>;


class SampleTest {
    public:
        SampleTest() {}

        bool init_args(int argc, char** argv);

        void print_helpInfo();

        void run_test();
    private:
        args_t args;
};

// bool SampleTest::init_args(int argc, char** argv);
// void SampleTest::print_helpInfo();

// 把device内存的__half数组转换成float数组
// void half2Float(float* dst, __half* src, int size);
void half2Float(const __half* d_halfArray, float* d_floatArray, int size);

// 把device内存的float数组转换成__half数组
// void float2Half(float* src, __half* dst, int size);
void float2Half(const float* d_floatArray, __half* d_halfArray, int size);

#endif