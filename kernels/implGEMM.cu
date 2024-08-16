#include <cuda_runtime.h>
#include "convs2d.h"
#include <stdio.h> 
#include <cstdint>

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
    if (x >= param.Oh * param.Ow || y >= param.k || z >= param.n) {
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

            if (posh_real >= 0 && posh_real < param.h && posw_real >= 0 && posw_real < param.w)
            {
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

    if (x >= param.Oh * param.Ow || y >= param.k || z >= param.n) {
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

/*
    Hard to say, it needs to be fixed.
*/
__global__ void implgemm_kernel_2_v2(param_t param) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float shInput[16][16];
    __shared__ float shWeight[16][16];

    // if (x >= param.Oh * param.Ow  || y >= param.k || z >= param.n) {
    //     return;
    // }

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
        if (weightOffset < rcs){
            shWeight[ty][tx] = param.weight[weightOffset + weightOffsetTmp];
        }
        else {
            shWeight[ty][tx] = 0.0;
        }

        int cur_c = (i + ty) / kernel_size;
        int cur_kh = ((i + ty) % kernel_size) / param.kw;
        int cur_kw = ((i + ty) % kernel_size) % param.kw;
        int curH = posh_ori + cur_kh;
        int curW = posw_ori + cur_kw;
        if (curH >= 0 && curW >= 0 && curH < param.h && curW < param.w)
        {
            int inputOffsetTmp = cur_c * intput_size + curH * param.w + curW;
            shInput[ty][tx] = param.input[inputOffsetTmp + inputOffset];
        }
        else {
            shInput[ty][tx] = 0.0;
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

/*

    我们知道每个线程快负责计算一个 16 * 16 的矩阵，每个线程负责计算一个元素的值，并且是weight行向量和input列向量的向量积。
    这里和上一个实现最大的不同是，在执行相同的加载操作之后， 选择的行向量 和 列向量 是不同的。
    因为， 我们将一个warp内的线程划分为 4 * 8 warp tile，这样的话， 同一时刻一个warp内的线程负责计算一个4 * 8的结果矩阵。
    这时， 产生的bank conflict的次数其实是与 warpx（8） + warpy（4） 成正比的。尽量的避免了 bank conflict。
*/
__global__ void implgemm_kernel_3(param_t param) {
    // 这里为了排列warp,合并了threadIdx的x和y维度
    // 这里做了转换，tx仍然相当于之前的x维度，对应着 Oh * Ow维度
    // ty 相当于之前的y维度, 对应着 k 维度
    uint32_t tx = threadIdx.x % 16;
    uint32_t ty = threadIdx.x / 16;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int z = blockIdx.z;

    

    // WARP TILE    
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t mma_tid_x = lane_id % 8;
    const uint32_t mma_tid_y = lane_id / 8;
    
    __shared__ float shm_input[16][16];
    __shared__ float shm_weight[16][16];

    // 当前线程加载的数据点在输入矩阵 Oh 和 Ow 上的坐标
    int pos_oh = (bx * 16 + tx) / param.Ow;
    int pos_ow = (bx * 16 + tx) % param.Ow;

    int pos_ori_h = pos_oh * param.stride_h - param.pad_h;  // 这里其实是该线程对应的卷积计算中的 “输入矩阵” 的起始 h 和 w 坐标
    int pos_ori_w = pos_ow * param.stride_w - param.pad_w;

    // 当前线程计算的数据点在输入矩阵中 Oh 和 Ow 的坐标 
    // 输出 【z, oy, ox / kw, ox % kw】
    uint32_t weight_lds_addr = (warp_id / 2) * 4 + mma_tid_y;   // 由于一个线程块负责16 * 16的结果矩阵输出，所以相当于 4 * 2 的warp排列
    uint32_t input_lds_addr = (warp_id % 2) * 8 + mma_tid_x;
    int ox = bx * 16 + input_lds_addr;          // 对应的输出矩阵的列坐标
    int oy = by * 16 + weight_lds_addr;         // 对应的输出矩阵的行坐标
    int ouput_offset = z * param.k * param.Oh * param.Ow + oy * param.Oh * param.Ow + ox;

    float sum = 0.0;    // 累计值

    int inputOffset = z * param.h * param.w * param.c;
    int weightOffset = (by * 16 + ty) * param.c * param.kh * param.kw;

    int input_channel_size = param.h * param.w;
    int kernel_channel_size = param.kh * param.kw;
    int kernel_size = param.kh * param.kw * param.c;

    for (int i = 0; i < kernel_size; i += 16) {

        /*
            每个线程需要负责加载2个数据点，分别是：
                shm_input[ty][tx] : 对应的是坐标 （z, pos_ori_h + offset_h, pos_ori_w + offset_w, offset_c）
                shm_weight[ty][tx] : 对应的是坐标 （（by * 16 + ty）, , offset_h, offset_w, offset_c）
        */

        int weight_offset_tmp = i + tx;
        // int cur_kc = (i + tx) / kernel_channel_size;
        // int cur_kh = ((i + tx) % kernel_channel_size) / param.kw;
        // int cur_kw = ((i + tx) % kernel_channel_size) % param.kw;
        // int weightOffsetTmp = cur_kc * input_channel_size + cur_kh * param.kw + cur_kw; //这部分可以省略，其实和就是 input_tmp
        if (weight_offset_tmp < kernel_size && weight_offset_tmp >= 0){
            shm_weight[ty][tx] = param.weight[weightOffset + weight_offset_tmp];    // 这里其实也需要判断是否越界
        }
        else {
            shm_weight[ty][tx] = 0.0;
        }


        int cur_c = (i + ty) / kernel_channel_size;
        int cur_kh = ((i + ty) % kernel_channel_size) / param.kw;
        int cur_kw = ((i + ty) % kernel_channel_size) % param.kw;
        int cur_h = pos_ori_h + cur_kh;
        int cur_w = pos_ori_w + cur_kw;

        if (cur_h >= 0 && cur_w >= 0 && cur_h < param.h && cur_w < param.w)
        {
            int inputOffsetTmp = cur_c * input_channel_size + cur_h * param.w + cur_w;
            shm_input[ty][tx] = param.input[inputOffsetTmp + inputOffset];
        }
        else {
            shm_input[ty][tx] = 0.0;
        }
        
        // 必须等待所有数据加载完毕，后续才能从shared memory中读取数据
        __syncthreads();

        #pragma unroll
        for (int subcrs = 0; subcrs < 16; subcrs++) {
            sum += shm_input[subcrs][input_lds_addr] * shm_weight[weight_lds_addr][subcrs];
        }

        __syncthreads();
    }

    
    if (ox < param.Oh * param.Ow && oy < param.k){
        param.output[ouput_offset] = sum;
    }

}

__global__ void implgemm_kernel_4(param_t param) {
    
}


void launch_implgemm(param_t param) {

    param.Oh = (param.h - param.kh + 2 * param.pad_h) / param.stride_h + 1;
    param.Ow = (param.w - param.kw + 2 * param.pad_w) / param.stride_w + 1;

    int blockx = (param.Oh * param.Ow + 15) / 16 ;
    int blocky = (param.k + 15) / 16;
    int blockz = param.n;
    int threadx = 256;         
    int thready = 1;
    int threadz = 1;

    dim3 block(threadx, thready, threadz);
    dim3 grid(blockx, blocky, blockz);
    implgemm_kernel_3<<<grid, block>>>(param);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(cuda_err));
    }

}