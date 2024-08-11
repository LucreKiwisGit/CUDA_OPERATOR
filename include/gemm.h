#ifndef GEMM_H
#define GEMM_H

// 行优先存储
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
// 将一个指针转换为 float4 类型的指针并访问它指向的第一个元素
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

typedef struct {
    float* A;
    float* B;
    float* C;
    float* alpha;
    float* beta;
    int M;
    int N;
    int K;
} param_gemm;

void launch_implgemm(param_gemm param);

void cpuSgemm(
    float *A, float *B, float *C, const int M, const int N, const int K
);

__global__ void naiveSgemm(
    float * __restrict__ A, float * __restrict__ B, float * __restrict__ C,
    const int M, const int N, const int K
);

__global__ void mySgemmV1Aligned(
    float * __restrict__ A, float * __restrict__ B, float * __restrict__ C,
    const int M, const int N, const int K
);

__global__ void mySgemmV2Aligned(
    float * __restrict__ A, float * __restrict__ B, float * __restrict__ C,
    const int M, const int N, const int K
);

__global__ void mySgemmV3Aligned(
    float * __restrict__ A, float * __restrict__ B, float * __restrict__ C,
    const int M, const int N, const int K
);

#endif