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


void im2col_gemm_fp16_test() {

    // 设置随机种子
    srand(0);

    int caseSize = 7;

    int N[caseSize] = {1, 64, 256, 16, 32, 2, 2};
    int C[caseSize] = {3, 256, 192, 256, 256, 1280, 960};
    int H[caseSize] = {12, 14,   14,  26,   14,  16,  64};
    int W[caseSize] = {12, 14,   14,  26,   14,  16,  64};
    int K[caseSize] = {1, 256,  192,   512,  256, 1280,   32};
    int R[caseSize] = {3, 3,    3,    3,    3,   3,   3};
    int S[caseSize] = {3, 3,    3,    3,    3,   3,   3};
    int P[caseSize] = {1, 1,    1,    1,    1,   1,   1};
    int Q[caseSize] = {1, 1,    1,    1,    1,   1,   1};
    int U[caseSize] = {1, 1,    1,    1,    1,   1,   1};
    int V[caseSize] = {1, 1,    1,    1,    1,   1,   1};

    for (int i = 0; i < caseSize; i++) {
        int n = N[i];
        int c = C[i];
        int h = H[i];
        int w = W[i];
        int k = K[i];
        int r = R[i];
        int s = S[i];
        int p = P[i];
        int q = Q[i];
        int u = U[i];
        int v = V[i];

        int OH = (w - r + 2 * p) / u + 1;
        int OW = (w - s + 2 * q) / v + 1;
        // int M = k;
        // int N = n * OH * OW;
        // int K = c * r * s;

        double flopsPerConv = 2.0 * n * k * c * r * s * OH * OW;

        // 分配空间
        auto input = std::make_unique<float[]>(n * c * h * w);
        auto weight = std::make_unique<float[]>(k * c * r * s);
        auto bias = std::make_unique<float[]>(k);
        auto output = std::make_unique<float[]>(n * k * OH * OW);
        auto output_benchmark = std::make_unique<float[]>(n * k * OH * OW);

        unique_ptr_cuda<__half> input_device(nullptr, CudaDeleter());
        unique_ptr_cuda<__half> weight_device(nullptr, CudaDeleter());
        unique_ptr_cuda<__half>  bias_device(nullptr, CudaDeleter());
        unique_ptr_cuda<__half> output_device(nullptr, CudaDeleter());
        unique_ptr_cuda<float> output_fp32_device(nullptr, CudaDeleter());
        unique_ptr_cuda<float> input_fp32_device(nullptr, CudaDeleter());
        unique_ptr_cuda<float> weight_fp32_device(nullptr, CudaDeleter());
        unique_ptr_cuda<float>  bias_fp32_device(nullptr, CudaDeleter());
        
        cudaError_t err;
        err = cudaMalloc((void **)&input_device, n * c * h * w * sizeof(__half));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        err = cudaMalloc((void **)&weight_device, k * c * r * s * sizeof(__half));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        err = cudaMalloc((void **)&bias_device, k * sizeof(__half));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        err = cudaMalloc((void **)&output_device, n * k * OH * OW * sizeof(__half));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        err = cudaMalloc((void **)&input_fp32_device, n * c * h * w * sizeof(float));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        err = cudaMalloc((void **)&weight_fp32_device, k * c * r * s * sizeof(float));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        err = cudaMalloc((void **)&bias_fp32_device, k * sizeof(float));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        err = cudaMalloc((void **)&output_fp32_device, n * k * OH * OW * sizeof(float));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        // random input
        for (int ii = 0; ii < n * c * h * w; ii++) {
            input[ii] = (float)rand() / RAND_MAX;
        }

        for (int ii = 0; ii < k * c * r * s; ii++) {
            weight[ii] = (float)rand() / RAND_MAX;
        }

        for (int ii = 0; ii < k; ii++) {
            bias[ii] = (float)rand() / RAND_MAX;
        }

        for (int ii = 0; ii < n * k * OH * OW; ii++) {
            output[ii] = (float)rand() / RAND_MAX;
            output_benchmark[ii] = (float)rand() / RAND_MAX;
        }

        cudaMemcpy(input_fp32_device.get(), input.get(), n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(weight_fp32_device.get(), weight.get(), k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(bias_fp32_device.get(), bias.get(), k * sizeof(float), cudaMemcpyHostToDevice);
        float2Half(input_fp32_device.get(), input_device.get(), n * c * h * w);
        float2Half(weight_fp32_device.get(), weight_device.get(), k * c * r * s);
        float2Half(bias_fp32_device.get(), bias_device.get(), k);

        // parameter 
        param_16t param;

        param.input = input_device.get();
        param.weight = weight_device.get();
        param.bias = bias_device.get();
        param.output = output_device.get();
        param.input_host = input.get();
        param.weight_host = weight.get();
        param.bias_host = bias.get();
        param.output_host = output.get();
        param.output_benchmark = output_benchmark.get();
        param.n = n;
        param.c = c;
        param.h = h;
        param.w = w;
        param.k = k;
        param.kh = r;
        param.kw = s;
        param.pad_h = p;
        param.pad_w = q;
        param.stride_h = u;
        param.stride_w = v;
        param.Oh = OH;
        param.Ow = OW;

        printf("\n========================Begin=============================\n");
        printf("n = %2d h = %2d w = %2d c = %2d r = %d s = %d k = %2d\n", n, h, w, c, r, s, k);

        
        // warm up
        im2col_gemm_fp16(param);
        // direct_conv2dCuDNN(param);
        
        cudaMemcpy(output.get(), output_device.get(), n * k * OH * OW * sizeof(__half), cudaMemcpyDeviceToHost);
        half2Float(output_device.get(), output_fp32_device.get(), param.n * param.k * param.Oh * param.Ow);

        // 验证正确率, 使用cudnn
        
        auto start_ref = std::chrono::steady_clock::now();  
        omp_set_num_threads(8);
        // direct_conv2dCpu(param);
        direct_conv2dCuDNN_fp16(param);
        cudaMemcpy(output_benchmark.get(), output_device.get(), n * k * OH * OW * sizeof(__half), cudaMemcpyDeviceToHost);

        auto end_ref = std::chrono::steady_clock::now();


        int error = 0;
        for (int ii = 0; ii < n * k * OH * OW; ii++) {
            if (abs(output[ii] - output_benchmark[ii]) > abs(output[ii]) * 0.00001) {
                printf("error, postion:%d, cudnnvalue:%.8f, implGEMMvalue:%.8f\n", ii, output_benchmark[ii], output[ii]);
                error++;
                // if (error > 0) {
                //     break;
                // }
            }
        }

        // launch_implgemm(param);
        // cudaMemcpy(output.get(), output_device.get(), n * k * OH * OW * sizeof(float), cudaMemcpyDeviceToHost);


        
        auto time_elapsed_ref = std::chrono::duration_cast<std::chrono::milliseconds>(end_ref - start_ref);
        double gflops = flopsPerConv / (time_elapsed_ref.count() / 1000.0) / 1e9 ;

        // printf(" time: %ld ms\n", time_elapsed_ref.count());
        printf("Verify finish,error:%d\n", error);
        printf("Cudnn Time per convolution: %ld ms\n", time_elapsed_ref.count()    );
        printf("Cudnn Performance :%f GFlops\n",  gflops);
        
    
        

        // 测量 CUDA 内核函数的执行时间
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, nullptr);  // 显式使用nullptr代替0

        float time_elapsed = 0.0;

        int iternum = 10;
        for (int i = 0; i < iternum; i++) {
            im2col_gemm_fp16(param);
        }

        cudaEventRecord(stop, nullptr);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_elapsed, start, stop);  // time_elapsed 以毫秒为单位

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // 计算每次卷积的平均时间和GFlops
        float timePerConv = time_elapsed / iternum;
        gflops = flopsPerConv / (timePerConv / 1000.0) / 1e9;

        // printf("n: %2d, h: %2d, w: %2d, c: %2d, r: %d, s: %d, k: %2d\n", n, h, w, c, r, s, k);
        printf("MyIm2colGEMM Time per convolution: %f ms\n", timePerConv);
        printf("MyIm2colGEMM Performance: %f GFlops\n", gflops);

        

    }
}


