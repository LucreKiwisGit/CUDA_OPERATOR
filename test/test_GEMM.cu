
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h>
#include "gemm.h"
#include "operators.h"

// 自定义删除器， 用于释放CUDA内存
struct CudaDeleter {
    template <typename T>
    void operator()(T *ptr) const {
        cudaFree(ptr);
    }
};


void printMatrix(float *M, int m, int n) {
    for(int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%12.6f  ", M[OFFSET(m, n, m)]);
        }
        printf("\n");
    }
}

float testCublasMaxError(const int M, const int N, const int K) {

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    
    h_a = (float *)malloc(M * K * sizeof(float));
    h_b = (float *)malloc(K * N * sizeof(float));
    h_c = (float *)malloc(M * N * sizeof(float));
    h_d_c = (float *)malloc(M * N * sizeof(float));
    cudaMalloc(&d_a, M * K * sizeof(float));
    if (d_a == NULL) {
        // 处理错误，例如打印错误信息并退出
        printf("CUDA Malloc failed for d_a\n");
        exit(EXIT_FAILURE);
    }
    cudaMalloc(&d_b, N * K * sizeof(float));
    if (d_b == NULL) {
        // 处理错误，例如打印错误信息并退出
        printf("CUDA Malloc failed for d_b\n");
        exit(EXIT_FAILURE);
    }
    cudaMalloc(&d_c, M * N * sizeof(float));
    if (d_c == NULL) {
        // 处理错误，例如打印错误信息并退出
        printf("CUDA Malloc failed for d_c\n");
        exit(EXIT_FAILURE);
    }
    
    srand(time(0));

    for (int i = 0;i < M * K;i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0;i < N * K;i++)
        h_b[i] = rand() / float(RAND_MAX);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * K * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0.0;

    cublasSgemm_v2(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);

    cudaMemcpy(h_d_c, d_c,  M * N * sizeof(float), cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0;i < M * N;i++) {
        float error = abs(h_d_c[i] - h_c[i]);
        // 可能会出现极小的值，导致数据溢出，生成NaN
        if (max_error != max_error || error != error) {
            max_error = -NAN;
        }
        else
            max_error = max(max_error, error);
    }

    //释放空间
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(cublas_handle);

    return max_error;
}

float testCublasPerformance(const int M, const int N, const int K, const int repeat) {

    int size_a = M * K * sizeof(float);
    int size_b = N * K * sizeof(float);
    int size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;

    cudaError_t cudaStat;

    cudaStat = cudaMalloc(&d_a, size_a);
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_a with error: %s\n", cudaGetErrorString(cudaStat));
        return -1;
    }
    cudaStat = cudaMalloc(&d_b, size_b);
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_b with error: %s\n", cudaGetErrorString(cudaStat));
        return -1;
    }
    cudaStat = cudaMalloc(&d_c, size_c);
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_c with error: %s\n", cudaGetErrorString(cudaStat));
        return -1;
    }

    srand(time(0));

    float *h_a = (float *)malloc(M * K * sizeof(float));
    float *h_b = (float *)malloc(K * N * sizeof(float));

    for (int i = 0;i < M * K;i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0;i < N * K;i++)
        h_b[i] = rand() / float(RAND_MAX);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta =  0.0;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    for (int i = 0; i < repeat; i++) {
        // cublasSgemm默认认为矩阵是列优先的
        // C^t = B^t A^t , 这里的C^t是列优先的，因此这是C的行优先
        cublasSgemm_v2(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    cublasDestroy(cublas_handle);

    return sec;


}

float testMaxError(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K
) {
    size_t size_a = M * K * sizeof(float);      // 27 * 66536 *4Bytes
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    // cudaMalloc(&d_a, size_a);
    cudaError_t err = cudaMalloc(&d_a, size_a);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed for d_a: %s\n", cudaGetErrorString(err));
        // handle the error
    }

    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(cuda_err));
    }
    
    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;


}

float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat
) {
    int size_a = M * K * sizeof(float);
    int size_b = N * K * sizeof(float);
    int size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;

    
    cudaError_t cudaStat = cudaGetLastError();  // 清除之前的错误状态

    cudaStat = cudaMalloc(&d_a, size_a);
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_a with error: %s\n", cudaGetErrorString(cudaStat));
        return -1;
    }
    cudaStat = cudaMalloc(&d_b, size_b);
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_b with error: %s\n", cudaGetErrorString(cudaStat));
        return -1;
    }
    cudaStat = cudaMalloc(&d_c, size_c);
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_c with error: %s\n", cudaGetErrorString(cudaStat));
        return -1;
    }

    srand(time(0));
    float *h_a = (float *)malloc(M * K * sizeof(float));
    float *h_b = (float *)malloc(K * N * sizeof(float));
    for (int i = 0;i < M * K;i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0;i < N * K;i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    // float cublas_alpha = 1.0;
    // float cublas_beta =  0.0;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    for (int i = 0; i < repeat; i++) {
        // cublasSgemm默认认为矩阵是列优先的
        // C^t = B^t A^t , 这里的C^t是列优先的，因此这是C的行优先
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    // cublasDestroy(cublas_handle);

    return sec;
}


void gemm_fp32_test() {
    const int M_list[16] = {27, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[16] = {65536, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[16] = {1152, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    const int outer_repeat = 10, inner_repeat = 1;
    const int TEST_NUM = 15;

    {
        printf("\nKernerl = cublas \n");

        {
            const int M = 512, N = 512, K = 512;

            float max_error = testCublasMaxError(M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {


            for (int i = 0;i < TEST_NUM;i++) {
                const int M = M_list[i];
                const int N = N_list[i];
                const int K = K_list[i];

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testCublasPerformance(M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double) M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    {
        printf("\nKernel = naiveSgemm\n");

        const int BM = 32, BN = 32;
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = 
            naiveSgemm;
        
        {
            const int M = 512, N = 512, K = 512;
            dim3 blockDim(BN, BM);
            dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            float max_error = testMaxError(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {


            for (int i = 0;i < TEST_NUM;i++) {
                const int M = M_list[i];
                const int N = N_list[i];
                const int K = K_list[i];

                dim3 blockDim(BN, BM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double) M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    {
        printf("\nKernel = mySgemmV1Aligned\n");

        const int BM = 128, BN = 128, TM = 8, TN = 8;
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = 
            mySgemmV1Aligned;
        
        {
            const int M = 512, N = 512, K = 512;
            dim3 blockDim(BN / TN, BM / TM);
            dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            float max_error = testMaxError(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);


        }

        {
            

            for (int i = 0;i < TEST_NUM;i++) {
                const int M = M_list[i];
                const int N = N_list[i];
                const int K = K_list[i];

                dim3 blockDim(BN / TN, BM / TM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double) M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    {
        printf("\nKernel = mySgemmV2Aligned\n");

        const int BM = 128, BN = 128, TM = 8, TN = 8;
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = 
            mySgemmV2Aligned;
        
        {
            const int M = 512, N = 512, K = 512;
            dim3 blockDim(BN / TN, BM / TM);
            dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            float max_error = testMaxError(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {

            for (int i = 0;i < TEST_NUM;i++) {
                const int M = M_list[i];
                const int N = N_list[i];
                const int K = K_list[i];

                dim3 blockDim(BN / TN, BM / TM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double) M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    {
        printf("\nKernel = mySgemmV3Aligned\n");

        const int BM = 128, BN = 128, TM = 8, TN = 8;
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = 
            mySgemmV3Aligned;
        
        {
            const int M = 512, N = 512, K = 512;
            dim3 blockDim(BN / TN, BM / TM);
            dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            float max_error = testMaxError(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {

            for (int i = 0;i < TEST_NUM;i++) {
                const int M = M_list[i];
                const int N = N_list[i];
                const int K = K_list[i];

                dim3 blockDim(BN / TN, BM / TM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double) M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

}