#include "operators.h"
#include <cuda_runtime_api.h>
#include "utils.h"
#include <stdio.h>
#include "convs2d.h"
#include <memory>
#include <omp.h>
#include "verify.h"
#include <chrono>

// 自定义删除器， 用于释放CUDA内存
struct CudaDeleter {
    template <typename T>
    void operator()(T *ptr) const {
        cudaFree(ptr);
    }
};

void implicit_gemm_fp16_test() {

    int caseSize = 6;

    int N[caseSize] = {16, 16, 16, 2, 2, 2};
    int C[caseSize] = {128, 256, 64, 1920, 640, 320};
    int H[caseSize] = {64,   32,  128,   32,  64,  64};
    int W[caseSize] = {64,   32,  128,   32,  64,  64};
    int K[caseSize] = {27,  256,   64,  640, 640,   4};
    int R[caseSize] = { 3,    3,    3,    3,   3,   3};
    int S[caseSize] = { 3,    3,    3,    3,   3,   3};
    int P[caseSize] = { 1,    1,    1,    1,   1,   1};
    int Q[caseSize] = { 1,    1,    1,    1,   1,   1};
    int U[caseSize] = { 1,    1,    1,    1,   1,   1};
    int V[caseSize] = { 1,    1,    1,    1,   1,   1};

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
        // float *input = (float *)malloc(n * c * h * w * sizeof(float));
        // float *weight = (float *)malloc(k * c * r * s * sizeof(float));
        // float *bias = (float *)malloc(k * sizeof(float));
        // float *output = (float *)malloc(n * k * OH * OW * sizeof(float));
        // float *output_benchmark = (float *)malloc(n * k * OH * OW * sizeof(float));

        std::unique_ptr<float, CudaDeleter> input_device(nullptr, CudaDeleter());
        std::unique_ptr<float, CudaDeleter> weight_device(nullptr, CudaDeleter());
        std::unique_ptr<float, CudaDeleter> bias_device(nullptr, CudaDeleter());
        std::unique_ptr<float, CudaDeleter> output_device(nullptr, CudaDeleter());
        
        cudaError_t err;
        err = cudaMalloc((void **)&input_device, n * c * h * w * sizeof(float));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        err = cudaMalloc((void **)&weight_device, k * c * r * s * sizeof(float));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        err = cudaMalloc((void **)&bias_device, k * sizeof(float));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        err = cudaMalloc((void **)&output_device, n * k * OH * OW * sizeof(float));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        // float *input_device, *weight_device, *bias_device, *output_device;
        // cudaMalloc((void **)&input_device, n * c * h * w * sizeof(float));
        // cudaMalloc((void **)&weight_device, k * c * r * s * sizeof(float));
        // cudaMalloc((void **)&bias_device, k * sizeof(float));
        // cudaMalloc((void **)&output_device, n * k * OH * OW * sizeof(float));

        // random input
        for (int ii = 0; ii < n * c * h * w; ii++) {
            input[ii] = rand() / RAND_MAX;
        }

        for (int ii = 0; ii < k * c * r * s; ii++) {
            weight[ii] = rand() / RAND_MAX;
        }

        for (int ii = 0; ii < k; ii++) {
            bias[ii] = rand() / RAND_MAX;
        }

        for (int ii = 0; ii < n * k * OH * OW; ii++) {
            output[ii] = rand() / RAND_MAX;
            output_benchmark[ii] = rand() / RAND_MAX;
        }

        cudaMemcpy(input_device.get(), input.get(), n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(weight_device.get(), weight.get(), k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(bias_device.get(), bias.get(), k * sizeof(float), cudaMemcpyHostToDevice);

        // parameter 
        param_t param;

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

        // warm up
        launch_implgemm(param);

        cudaMemcpy(output.get(), output_device.get(), n * k * OH * OW * sizeof(float), cudaMemcpyDeviceToHost);


        // 验证正确率
        if (i == 0) {
            auto start_ref = std::chrono::steady_clock::now();
            omp_set_num_threads(8);
            direct_conv2dCpu(param);
            auto end_ref = std::chrono::steady_clock::now();

            int error = 0;
            for (int ii = 0; ii < n * k * OH * OW; ii++) {
                if (abs(output[ii] - output_benchmark[ii]) > getPrecision(output[ii])) {
                    printf("error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", ii, output_benchmark[ii], output[ii]);
                    error++;
                    break;
                }
            }

            
            auto time_elapsed_ref = std::chrono::duration_cast<std::chrono::milliseconds>(end_ref - start_ref);
            double gflops = flopsPerConv / (time_elapsed_ref.count() / 1000.0) / 1e9 ;
            printf("%2d %2d %2d %2d %d %d %2d\n", n, h, w, c, r, s, k);
            printf(" time: %ld ms\n", time_elapsed_ref.count());
            printf("Performance :%f GFlops\n",  gflops);
            printf("================finish,error:%d=========================\n", error);
        }
        

        // cost time test
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        float time_elapsed = 0.0;

        int iternum = 10;
        for (int i = 0; i < iternum; i++) {
            launch_implgemm(param);
        }

        

        cudaEventRecord(stop, 0);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_elapsed, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        float timePerConv = time_elapsed / iternum;
        double gflops = flopsPerConv / (time_elapsed / 1000.0) / 1e9;
        printf("%2d %2d %2d %2d %d %d %2d\n", n, h, w, c, r, s, k);
        printf("time: %f ms\n", timePerConv);
        printf("Performance :%f GFlops\n",  gflops);

    }
}


