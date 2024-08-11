#include <cuda_runtime.h>
#include "convs2d.h"
#include <stdio.h> 

__global__ void implgemm_kernel_1(param_t param) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;      // 对应的是input矩阵的列维度{Oh * Ow}
    int y = blockIdx.y * blockDim.y + threadIdx.y;      // 对应的是weight矩阵的行维度{K}
    int z = blockIdx.z;                                 // 对应的是input矩阵的列维度{N}

    /*
        也就是每个线程负责计算：
            weight矩阵的第y行 * input矩阵的第 {z * x} 列
        =   outpu[y][z * x]    

        其实就是一次卷积运算，第y个卷积核和input的{N, x / Ow, x}对应的矩阵相乘；
    */
    

    // 多分配的线程
    if (x >= param.Oh || y >= param.k || z >= param.n) {
        return;
    }

    int oh = x / param.Ow;  
    int ow = x % param.Ow;
    int posh_ori = oh * param.stride_h - param.pad_h;   //  输入矩阵起始位置维度h
    int posw_ori = ow * param.stride_w - param.pad_w;   //  输入矩阵起始位置维度w

    float sum = 0.0;    // output[y][z * x], 输出维度 [N, K, Oh, Ow] ---> [z, y, oh, ow]

    int inputOffset =  z * param.h * param.w * param.c + posh_ori * param.w + posw_ori;         // 输入矩阵在 input[N][C][H][W]中的起始位置
    int weightOffset = y * param.c * param.kh * param.kw;           // 权重矩阵在 weight[K][C][KH][KW]中的起始位置

    int input_size = param.h * param.w;
    int kernel_size = param.kh * param.kw;

    // 开始遍历计算当前卷积 ，总共 Channel * kh * kw 个元素相乘
    for (int i = 0; i < param.kh; i++) {    

        for (int j = 0; j < param.kw; j++) {
            int posh_real = posh_ori + i;
            int posw_real = posw_ori + j;

            if (posh_real < 0 || posh_real >= param.h || posw_real < 0 || posw_real >= param.w)
            {
                continue ;
            }
            else {
                int inputOffsetTmp = inputOffset;           // 这里是加上了channnel的维度
                int weightOffsetTmp = weightOffset;         // 这里是加上了channnel的维度

                for (int k = 0; k < param.c; k++) {
                    sum += param.input[inputOffsetTmp + i * param.w + j] * param.weight[weightOffsetTmp + i * param.kw + j];

                    inputOffsetTmp += input_size;
                    weightOffsetTmp += kernel_size;
                }
            }
            
        }
    }

    // 计算输出偏移
    int outOffset = z * param.k * param.Oh * param.Ow + y * param.Oh * param.Ow + x;
    param.output[outOffset] = sum;

}

__global__ void implgemm_kernel_2(param_t param) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if (x >= param.Oh || y >= param.k || z >= param.n) {
        return;
    }

    int oh = x / param.Ow;
    int ow = x % param.Ow;
    int posh_ori = oh * param.stride_h - param.pad_h;
    int posw_ori = ow * param.stride_w - param.pad_w;

    float sum = 0.0;

    int inputOffset = z * param.h * param.w * param.c;
    int weightOffset = y * param.c * param.kh * param.kw;

    int intput_size = param.h * param.w;
    int kernel_size = param.kh * param.kw;

    for (int i = 0; i < param.kh * param.kw * param.c; i++) {

        int weightOffsetTmp = i;
        int cur_c = i / kernel_size;
        int cur_kh = (i % kernel_size) / param.kw;
        int cur_kw = (i % kernel_size) % param.kw;
        int curH = posh_ori + cur_kh;
        int curW = posw_ori + cur_kw;
        int inputOffsetTmp = cur_c * intput_size + curH * param.w + curW;

        if (curH >= 0 && curW >= 0 && curH < param.h && curW < param.w)
        {
            sum += param.input[inputOffsetTmp + inputOffset] * param.weight[weightOffsetTmp + weightOffset];
        }

    }

    // 计算输出偏移
    int outOffset = z * param.k * param.Oh * param.Ow + y * param.Oh * param.Ow + x;
    param.output[outOffset] = sum;

}

__global__ void implgemm_kernel_2_v2(param_t param) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float shInput[16][16];
    __shared__ float shWeight[16][16];

    if (x >= param.Oh || y >= param.k || z >= param.n) {
        return;
    }

    int oh = x / param.Ow;
    int ow = x % param.Ow;
    int posh_ori = oh * param.stride_h - param.pad_h;
    int posw_ori = ow * param.stride_w - param.pad_w;

    float sum = 0.0;

    int inputOffset = z * param.h * param.w * param.c;
    int weightOffset = y * param.c * param.kh * param.kw;

    int intput_size = param.h * param.w;
    int kernel_size = param.kh * param.kw;

    int rcs = param.kh * param.kw * param.c;

    for (int i = 0; i < rcs; i+= 16) {

        // 读取数据加载到shared memory

        int weightOffsetTmp = i + tx;
        if (tx + i < rcs){
            shInput[ty][tx] = param.input[weightOffset + weightOffsetTmp];
        }

        if (ty + i < rcs){
            int cur_c = (i + ty) / kernel_size;
            int cur_kh = ((i + ty) % kernel_size) / param.kw;
            int cur_kw = ((i + ty) % kernel_size) % param.kw;
            int curH = posh_ori + cur_kh;
            int curW = posw_ori + cur_kw;
            int inputOffsetTmp = cur_c * intput_size + curH * param.w + curW;
            shWeight[ty][tx] = param.input[inputOffset + inputOffsetTmp];
        }

        __syncthreads();

        #pragma unroll
        for (int subcrs = 0; subcrs < 16; subcrs++) {
            sum += shInput[subcrs][tx] * shWeight[ty][subcrs];
        }

        __syncthreads();
    }

    // 计算输出偏移
    int outOffset = z * param.k * param.Oh * param.Ow + y * param.Oh * param.Ow + x;
    if (x < param.Oh * param.Ow && y < param.k){
        param.output[outOffset] = sum;
    }


}

__global__ void implgemm_kernel_3(param_t param) {
    
}

void launch_implgemm(param_t param) {

    param.Oh = (param.h - param.kh + 2 * param.pad_h) / param.stride_h + 1;
    param.Ow = (param.w - param.kw + 2 * param.pad_w) / param.stride_w + 1;

    int blockx = (param.Oh * param.Ow + 15) / 16 ;
    int blocky = (param.k + 15) / 16;
    int blockz = param.n;
    int threadx = 16;
    int thready = 16;
    int threadz = 1;

    dim3 block(threadx, thready, threadz);
    dim3 grid(blockx, blocky, blockz);
    implgemm_kernel_2_v2<<<grid, block>>>(param);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(cuda_err));
    }

}