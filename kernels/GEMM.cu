#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h>
#include "gemm.h"



void cpuSgemm(
    float *A, float *B, float *C, const int M, const int N, const int K
){

    for (int i = 0;i < M;i++) {
        for (int j = 0;j < N;j++) {
            float sum = 0.0;
            for (int k = 0;k < K;k++) {
                sum += A[OFFSET(i, k, K)] * B[OFFSET(k, j, N)];
            }
            C[OFFSET(i, j, N)] = sum;
        }
    }

}

__global__ void naiveSgemm(
    float * __restrict__ A, float * __restrict__ B, float * __restrict__ C,
    const int M, const int N, const int K
){

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < M && n < N) {
        float sum = 0.0;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            sum += A[OFFSET(m, k, K)] * B[OFFSET(k, n, N)];
        }
        C[OFFSET(m, n, N)] = sum;
    }

}

__global__ void mySgemmV1Aligned(
    float * __restrict__ A, float * __restrict__ B, float * __restrict__ C,
    const int M, const int N, const int K
) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0};              // 用于暂存C矩阵的数据，存放在寄存器中

    int load_a_smem_m = tid >> 1;           //  tid / 2
    int load_a_smem_k = (tid & 1) << 2;     //  (tid % 2) * 4
    int load_b_smem_k = tid >> 5;           //  tid / 32
    int load_b_smem_n = (tid & 31) << 2;    //  (tid % 32) * 4

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        // block内的线程分工把数据加载到shared memory中，即s_a、s_b
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        // FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(A[load_a_gmem_addr]);

        // 只在边界情况下进行检查，避免所有加载都进行边界判断
        if (load_a_gmem_k + 3 < K && load_a_gmem_m < M) {
            FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(A[load_a_gmem_addr]);
        } else {
            // printf("Thread %d: load_a_gmem_k = %d, load_a_gmem_m = %d, load_a_gmem_addr = %d\n", tid, load_a_gmem_k, load_a_gmem_m, load_a_gmem_addr);
            float4 a_val;
            a_val.x = (load_a_gmem_m < M && load_a_gmem_k < K) ? A[load_a_gmem_addr] : 0.0f;
            a_val.y = (load_a_gmem_m < M && load_a_gmem_k + 1 < K) ? A[load_a_gmem_addr + 1] : 0.0f;
            a_val.z = (load_a_gmem_m < M && load_a_gmem_k + 2 < K) ? A[load_a_gmem_addr + 2] : 0.0f;
            a_val.w = (load_a_gmem_m < M && load_a_gmem_k + 3 < K) ? A[load_a_gmem_addr + 3] : 0.0f;
            FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = a_val;
        }

        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        // FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(B[load_b_gmem_addr]);

        

        if (load_b_gmem_n + 3 < N && load_b_gmem_k < K) {
            FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(B[load_b_gmem_addr]);
        } else {
            float4 b_val;
            b_val.x = (load_b_gmem_n < N && load_b_gmem_k < K) ? B[load_b_gmem_addr] : 0.0f;
            b_val.y = (load_b_gmem_n + 1 < N && load_b_gmem_k < K) ? B[load_b_gmem_addr + 1] : 0.0f;
            b_val.z = (load_b_gmem_n + 2 < N && load_b_gmem_k < K) ? B[load_b_gmem_addr + 2] : 0.0f;
            b_val.w = (load_b_gmem_n + 3 < N && load_b_gmem_k < K) ? B[load_b_gmem_addr + 3] : 0.0f;
            FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = b_val;
        }

        __syncthreads(); //等待所有线程的数据全部加载到shared memory中


        // 每个线程计算TM * TN 子矩阵的结果，并存储到r_c数组寄存器里
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    r_c[m][n] += s_a[ty * TM + m][k] * s_b[k][tx * TN + n];     //注意坐标的计算
                }
            }
        }

        __syncthreads(); 
    }

    // 把计算好的TM * TN子矩阵数组r_c存储到对应C矩阵中
    // int store_c_gemm_m = by * BM + ty * TM;
    #pragma unroll
    for (int i = 0;i < TM; i++) {
        // store_c_gemm_m++;
        int store_c_gmem_m = by * BM + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            // FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);

            if (store_c_gmem_n + 3 < N && store_c_gmem_m < M) {
                FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
            }
            else {
                if (store_c_gmem_m < M) {
                    for (int k = 0; k < 4; k++) {
                        if (store_c_gmem_n + k < N) {
                            C[store_c_gmem_addr + k] = r_c[i][j + k];
                        }
                    }
                }

            }

        }
    }



}


__global__ void mySgemmV2Aligned(
    float * __restrict__ A, float * __restrict__ B, float * __restrict__ C,
    const int M, const int N, const int K
) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BK][BM];   // s_a逆转过来，以便取数据计算时可以连续取值
    __shared__ float s_b[BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];     //  存储从s_a取出的TM长度的向量
    float r_comp_b[TN];     //  存储从s_b取出的TN长度的向量

    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        if (load_a_gmem_m < M && load_a_gmem_k + 3 < K){
            FLOAT4(r_load_a[0]) = FLOAT4(A[load_a_gmem_addr]);
        }
        else {
            r_load_a[0] = (load_a_gmem_m < M && load_a_gmem_k < K) ? A[load_a_gmem_addr] : 0.0f;
            r_load_a[1] = (load_a_gmem_m < M && load_a_gmem_k + 1 < K) ? A[load_a_gmem_addr + 1] : 0.0f;
            r_load_a[2] = (load_a_gmem_m < M && load_a_gmem_k + 2 < K) ? A[load_a_gmem_addr + 2] : 0.0f;
            r_load_a[3] = (load_a_gmem_m < M && load_a_gmem_k + 3 < K) ? A[load_a_gmem_addr + 3] : 0.0f;
        }

        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        if (load_b_gmem_n + 3 < N && load_b_gmem_k < K) {
            FLOAT4(r_load_b[0]) = FLOAT4(B[load_b_gmem_addr]);
        }
        else {
            r_load_b[0] = (load_b_gmem_n < N && load_b_gmem_k < K) ? B[load_b_gmem_addr] : 0.0f;
            r_load_b[1] = (load_b_gmem_n + 1 < N && load_b_gmem_k < K) ? B[load_b_gmem_addr + 1] : 0.0f;
            r_load_b[2] = (load_b_gmem_n + 2 < N && load_b_gmem_k < K) ? B[load_b_gmem_addr + 2] : 0.0f;
            r_load_b[3] = (load_b_gmem_n + 3 < N && load_b_gmem_k < K) ? B[load_b_gmem_addr + 3] : 0.0f;
        }


        s_a[load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {

            // 从共享内存取出两个向量
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);

            // 计算外积，注意这里的矩阵内存位置不是连续的，不能直接写回C
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }

        }
        
        __syncthreads();
    }

    // 把r_c矩阵根据空间变换写回矩阵C
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

        if (store_c_gmem_m < M) {
            if (store_c_gmem_n + 3 < N) {
                FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);

                if (store_c_gmem_n + 3 + BN  / 2 < N) {
                    FLOAT4(C[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
                }
                else if (store_c_gmem_n + BN / 2 < N) {
                    for (int k = 0; k < 4; k++) {
                        if (store_c_gmem_n + k + BN / 2 < N) {
                            C[store_c_gmem_addr + BN / 2 + k] = r_c[i + TM / 2][k];
                       }
                    }
                }
            }
            else {
                for (int k = 0; k < 4; k++) {
                    if (store_c_gmem_n + k < N) {
                        C[store_c_gmem_addr + k] = r_c[i][k];
                    }
                }

            }
        }
        // FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        // FLOAT4(C[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + BM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
 
        if (store_c_gmem_m < M && store_c_gmem_n + 3 < N) {
            FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        } 
        else if (store_c_gmem_m < M) {

            for (int k = 0; k < 4; k++) {
                if (store_c_gmem_n + k < N) {
                    C[store_c_gmem_addr + k] = r_c[i + TM / 2][k];
                    break;
                }
            }

        }

        if (store_c_gmem_m < M && store_c_gmem_n + 3 < N) {
            FLOAT4(C[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
        } 
        else if (store_c_gmem_m < M) {

            for (int k = 0; k < 4; k++) {
                if (store_c_gmem_n + k < N) {
                    C[store_c_gmem_addr + k + BN / 2] = r_c[i + TM / 2][k + 4];
                    break;
                }
            }

        }
    

        // FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        // FLOAT4(C[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}

__global__ void mySgemmV3Aligned(
    float * __restrict__ A, float * __restrict__ B, float * __restrict__ C,
    const int M, const int N, const int K
) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // 使用双倍的share memory来预取数据，
    // 在计算数据之前加载下一次循环用到的数据（从 global Memory 加载到 Shared Memory)
    // GPU无法乱序执行，必须在计算之前就进行数据的加载
    __shared__ float s_a[2][BK][BM];   
    __shared__ float s_b[2][BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];     //  存储从s_a取出的TM长度的向量
    float r_comp_b[TN];     //  存储从s_b取出的TN长度的向量

    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    // 第一次把数据写进share memory中
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        // FLOAT4(r_load_a[0]) = FLOAT4(A[load_a_gmem_addr]);
        if (load_a_gmem_m < M && load_a_gmem_k + 3 < K){
            FLOAT4(r_load_a[0]) = FLOAT4(A[load_a_gmem_addr]);
        }
        else {
            r_load_a[0] = (load_a_gmem_m < M && load_a_gmem_k < K) ? A[load_a_gmem_addr] : 0.0f;
            r_load_a[1] = (load_a_gmem_m < M && load_a_gmem_k + 1 < K) ? A[load_a_gmem_addr + 1] : 0.0f;
            r_load_a[2] = (load_a_gmem_m < M && load_a_gmem_k + 2 < K) ? A[load_a_gmem_addr + 2] : 0.0f;
            r_load_a[3] = (load_a_gmem_m < M && load_a_gmem_k + 3 < K) ? A[load_a_gmem_addr + 3] : 0.0f;    
        }

        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        // FLOAT4(r_load_b[0]) = FLOAT4(B[load_b_gmem_addr]);
        if (load_b_gmem_k < K && load_b_gmem_n + 3 < N) {
            FLOAT4(r_load_b[0]) = FLOAT4(B[load_b_gmem_addr]);
        }
        else {
            r_load_b[0] = (load_b_gmem_k < K && load_b_gmem_n < N) ? B[load_b_gmem_addr] : 0.0f;
            r_load_b[1] = (load_b_gmem_k < K && load_b_gmem_n + 1 < N) ? B[load_b_gmem_addr + 1] : 0.0f;
            r_load_b[2] = (load_b_gmem_k < K && load_b_gmem_n + 2 < N) ? B[load_b_gmem_addr + 2] : 0.0f;
            r_load_b[3] = (load_b_gmem_k < K && load_b_gmem_n + 3 < N) ? B[load_b_gmem_addr + 3] : 0.0f;
        }

        s_a[0][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
    }

    __syncthreads();
    
    // 循环次数减少一次，bk可以看做是加载第几次循环需要的数据到share memory中。
    // 循环内其实计算的是第 bk - 1 次的加载的数据
    for (int bk = 1; bk < (K + BK - 1) / BK ; bk++) {

        int smem_sel = (bk - 1) & 1;   // 当前循环计算需要使用的share memory序号
        int smem_next = bk & 1;

        // 把下一次循环用到的数据从global memory中加载到寄存器中（会不会寄存器不够用啊）
        // 这里使用的LDG指令进行load数据，不会影响后续的运算指令的发射执行
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

        // FLOAT4(r_load_a[0]) = FLOAT4(A[load_a_gmem_addr]);
        if (load_a_gmem_m < M && load_a_gmem_k + 3 < K){
            FLOAT4(r_load_a[0]) = FLOAT4(A[load_a_gmem_addr]);
        }
        else {
            r_load_a[0] = (load_a_gmem_m < M && load_a_gmem_k < K) ? A[load_a_gmem_addr] : 0.0f;
            r_load_a[1] = (load_a_gmem_m < M && load_a_gmem_k + 1 < K) ? A[load_a_gmem_addr + 1] : 0.0f;
            r_load_a[2] = (load_a_gmem_m < M && load_a_gmem_k + 2 < K) ? A[load_a_gmem_addr + 2] : 0.0f;
            r_load_a[3] = (load_a_gmem_m < M && load_a_gmem_k + 3 < K) ? A[load_a_gmem_addr + 3] : 0.0f;    
        }

        // FLOAT4(r_load_b[0]) = FLOAT4(B[load_b_gmem_addr]);
        if (load_b_gmem_k < K && load_b_gmem_n + 3 < N) {
            FLOAT4(r_load_b[0]) = FLOAT4(B[load_b_gmem_addr]);
        }
        else {
            r_load_b[0] = (load_b_gmem_k < K && load_b_gmem_n < N) ? B[load_b_gmem_addr] : 0.0f;
            r_load_b[1] = (load_b_gmem_k < K && load_b_gmem_n + 1 < N) ? B[load_b_gmem_addr + 1] : 0.0f;
            r_load_b[2] = (load_b_gmem_k < K && load_b_gmem_n + 2 < N) ? B[load_b_gmem_addr + 2] : 0.0f;
            r_load_b[3] = (load_b_gmem_k < K && load_b_gmem_n + 3 < N) ? B[load_b_gmem_addr + 3] : 0.0f;
        }

        // 还有这里的同步指令不能使用了，我们希望加载与计算能够并行执行

        // 计算预取的数据
        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            // 从共享内存取出两个向量
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

            // 计算外积，注意这里的矩阵内存位置不是连续的，不能直接写回C
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        // 把加载的数据从寄存器中写入共享内存中
        // 这部分的STS指令会等待LDG指令写回后再继续发射执行，所以不能放在计算部分之前
        s_a[smem_next][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[smem_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[smem_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
        
        __syncthreads();
    }

    // 计算最后一次循环
    int smem_sel = ((K + BK - 1) / BK - 1) & 1;  
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        // 从共享内存取出两个向量
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

        // 计算外积，注意这里的矩阵内存位置不是连续的，不能直接写回C
        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
            }
        }
    }


    // 把r_c矩阵根据空间变换写回矩阵C
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

        if (store_c_gmem_m < M && store_c_gmem_n + 3 < N) {
            FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        } 
        else if (store_c_gmem_m < M) {

            for (int k = 0; k < 4; k++) {
                if (store_c_gmem_n + k < N) {
                    C[store_c_gmem_addr + k] = r_c[i + TM / 2][k];
                    break;
                }
            }

        }

        if (store_c_gmem_m < M && store_c_gmem_n + 3 < N) {
            FLOAT4(C[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
        } 
        else if (store_c_gmem_m < M) {

            for (int k = 0; k < 4; k++) {
                if (store_c_gmem_n + k < N) {
                    C[store_c_gmem_addr + k + BN / 2] = r_c[i + TM / 2][k + 4];
                    break;
                }
            }

        }
    }

    // 把r_c矩阵根据空间变换写回矩阵C
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

        if (store_c_gmem_m < M) {
            if (store_c_gmem_n + 3 < N) {
                FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);

                if (store_c_gmem_n + 3 + BN  / 2 < N) {
                    FLOAT4(C[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
                }
                else if (store_c_gmem_n + BN / 2 < N) {
                    for (int k = 0; k < 4; k++) {
                        if (store_c_gmem_n + k + BN / 2 < N) {
                            C[store_c_gmem_addr + BN / 2 + k] = r_c[i + TM / 2][k];
                       }
                    }
                }
            }
            else {
                for (int k = 0; k < 4; k++) {
                    if (store_c_gmem_n + k < N) {
                        C[store_c_gmem_addr + k] = r_c[i][k];
                    }
                }

            }
        }
        // FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        // FLOAT4(C[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + BM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
 
        if (store_c_gmem_m < M && store_c_gmem_n + 3 < N) {
            FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        } 
        else if (store_c_gmem_m < M) {

            for (int k = 0; k < 4; k++) {
                if (store_c_gmem_n + k < N) {
                    C[store_c_gmem_addr + k] = r_c[i + TM / 2][k];
                    break;
                }
            }

        }

        if (store_c_gmem_m < M && store_c_gmem_n + 3 < N) {
            FLOAT4(C[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
        } 
        else if (store_c_gmem_m < M) {

            for (int k = 0; k < 4; k++) {
                if (store_c_gmem_n + k < N) {
                    C[store_c_gmem_addr + k + BN / 2] = r_c[i + TM / 2][k + 4];
                    break;
                }
            }

        }
    }
}



void launch_implgemm(param_gemm param) {
    /*
        感觉不是很必要重新写一个测试的接口，先用着 test_GEMM.cu 的 gemm_fp32_test();
    */
}