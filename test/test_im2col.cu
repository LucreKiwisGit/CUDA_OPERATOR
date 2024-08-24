#include "operators.h"
#include <cuda_runtime_api.h>
#include "utils.h"
#include <stdio.h>
#include "convs2d.h"
#include <memory>
#include <omp.h>
#include "verify.h"
#include <chrono>
#include <cuda_fp16.h>


/*
    Test for im2col
*/

/*
    单元测试：用于检测im2col_kernel的正确性。
    算了，没空，有空再写。
*/
// void im2col_fp16_test() {
//     printf("im2col_fp16_test\n");

//     param_16t param;

//     const size_t size = 10000;
//     float *test_host = (float *)malloc(size * sizeof(float));
//     unique_ptr_cuda<float> test_fp32_device(nullptr, CudaDeleter());
//     unique_ptr_cuda<__half> test_fp16_device(nullptr, CudaDeleter());
//     cudaError_t err;
//     err = cudaMalloc((void **)&test_fp32_device, size * sizeof(float));
//     if (err != cudaSuccess) {
//         printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
//         exit(1);
//     }
//     err = cudaMalloc((void **)&test_fp16_device, size * sizeof(__half));
//     if (err != cudaSuccess) {
//         printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
//         exit(1);
//     }

//     srand(0);
//     for (int i = 0; i < size; ++i) {
//         test_host[i] = float(rand()) / RAND_MAX;
//     }

//     err = cudaMemcpy(test_fp32_device.get(), test_host, size * sizeof(float), cudaMemcpyHostToDevice);
//     if (err != cudaSuccess) {
//         printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
//         exit(1);
//     }

//     float2Half(test_fp32_device.get(), test_fp16_device.get(), size);



//     /*  start to launch im2col kernel   */
//     int blockx = (param.Ow + 15) / 16;
//     int blocky = (param.Oh + 15) / 16;
//     int blockz = param.n * param.c;
//     int threadx = 16;
//     int thready = 16;
//     int threadz = 1;
//     dim3 block(threadx, thready, threadz);
//     dim3 grid(blockx, blocky, blockz);

//     im2col_kernel<<<grid, block>>>(param);
// }