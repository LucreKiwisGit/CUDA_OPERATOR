#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "convs2d.h"
#include <memory>
#include <cstring>
#include "utils.h"
#include "gemm.h"


// __global__ void im2col_kernel_V1(const __half* data_im, int n, int channels, int height, int width,
//                               int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
//                               int output_h, int output_w, __half* data_col) {
//     int ow = blockIdx.x * blockDim.x + threadIdx.x;
//     int oh = blockIdx.y * blockDim.y + threadIdx.y;
//     int c = blockIdx.z;

    
//     if (ow < output_w && oh < output_h) {
//         int b = c / channels;
//         int channel = c % channels;

//         int im_col = ow * stride_w - pad_w;
//         int im_row = oh * stride_h - pad_h;
        
//         for (int kh = 0; kh < kernel_h; ++kh) {
//             #pragma unroll
//             for (int kw = 0; kw < kernel_w; ++kw) {
//                 int h_in = im_row + kh;
//                 int w_in = im_col + kw;

//                 int col_index = ((c * kernel_h + kh) * kernel_w + kw) * output_h * output_w + oh * output_w + ow;
                 
//                 if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
//                     data_col[col_index] = data_im[(b * channels + channel) * height * width + h_in * width + w_in];
//                 } else {
//                     data_col[col_index] = 0.0;
//                 }
//             }
//         }
//     }
// }

// im2col kernel
// im2col_input dims : [crs, n * h * w]
// blocksize(16, 16) gridsize((oh + 15)/ 16, (ow + 15)/ 16, n * c)
__global__ void im2col_kernel(param_16t param) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    
    if (ow < param.Ow && oh < param.Oh) {
        int b = z / param.c;
        int channel = z % param.c;

        int im_col = ow * param.stride_w -param.pad_w;
        int im_row = oh * param.stride_h -param.pad_h;
        int input_offset = b * param.c * param.h * param.w + channel * param.h * param.w + oh * param.Ow + ow;
        int col_offset = channel * param.kh * param.kw * param.n * param.Oh * param.Ow + b * param.Oh * param.Ow; 

        #pragma unroll
        for (int kh = 0; kh < param.kh; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < param.kw; ++kw) {
                int ori_h = im_row + kh;
                int ori_w = im_col + kw;
                // int col_index = (b * param.c + channel) * param.kh * param.kw * param.Oh * param.Ow + 
                //                 (kh * param.kw + kw) * param.Oh * param.Ow + oh * param.Ow + ow;
                // int col_index = ((z * param.kh + kh) * param.kw + kw) * param.Oh * param.Ow + oh * param.Ow + ow;
                int col_tmp = (kh * param.kw + kw) * param.n * param.Oh * param.Ow;
                int input_tmp = kh * param.w + kw;   
                
                if (ori_h >= 0 && ori_h < param.h && ori_w >= 0 && ori_w < param.w) {
                    param.im2col_input[col_offset + col_tmp] = param.input[input_offset + input_tmp];
                }
                else {
                    param.im2col_input[col_offset + col_tmp] = 0.0;
                }
            }
        }
    }
}

/*
    im2col + gemm 得到的结果矩阵是 [k, n * Oh * Ow],
    需要转换为 [n , k , Oh, Ow].
    blocksize(16, 16) gridsize((oh + 15)/ 16, (ow + 15)/ 16, n * k)
*/
__global__ void im2col_reshape_kernel(param_16t param, __half* output_im2col) {
    int oh = blockIdx.x * blockDim.x + threadIdx.x;
    int ow = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if (ow < param.Ow && oh < param.Oh) {
        int n = z / param.k;
        int k = z % param.k;
        int input_offset = k * param.n * param.Oh * param.Ow + n * param.Oh * param.Ow + oh * param.Ow + ow;
        int output_offset = z * param.Oh * param.Ow + oh * param.Ow + ow;

        param.output[output_offset] = output_im2col[input_offset];
    }
}

void im2col_gemm_fp16(param_16t param) {
    param.Oh = (param.h + 2 * param.pad_h - param.kh) / param.stride_h + 1;
    param.Ow = (param.w + 2 * param.pad_w - param.kw) / param.stride_w + 1;

    // 分配im2col_input空间
    unique_ptr_cuda<__half> im2col_input_device(nullptr, CudaDeleter());
    cudaError_t err;
    err = cudaMalloc((void **)&im2col_input_device, param.n * param.c * param.kh * param.kw * param.Oh * param.Ow * sizeof(__half));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    param.im2col_input = im2col_input_device.get();

    // im2col
    int blockx = (param.Ow + 15) / 16;
    int blocky = (param.Oh + 15) / 16;
    int blockz = param.n * param.c;
    int threadx = 16;
    int thready = 16;
    int threadz = 1;
    dim3 block(threadx, thready, threadz);
    dim3 grid(blockx, blocky, blockz);
    im2col_kernel<<<grid, block>>>(param);

    // gemm
    int M = param.k;
    int N = param.n * param.Oh * param.Ow;
    int K = param.c * param.kh * param.kw;
    unique_ptr_cuda<__half> output_im2col(nullptr, CudaDeleter());
    err = cudaMalloc((void **)&output_im2col, M * N * sizeof(__half));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    //哎呀， 应该给核函数再套个接口的，不然还要设置线程块和网格，太麻烦了。。。
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    myHgemmV3Aligned<<<blockDim, gridDim>>>(param.weight, param.im2col_input, output_im2col.get(), M, N, K);

    // resahpe output 
    blockx = (param.Ow + 15) / 16;
    blocky = (param.Oh + 15) / 16;
    blockz = param.n * param.k;
    threadx = 16;
    thready = 16;
    block = dim3(threadx, thready, threadz);
    grid = dim3(blockx, blocky, blockz);
    im2col_reshape_kernel<<<grid, block>>>(param, output_im2col.get());

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(cuda_err));
    }

    return ;
}

