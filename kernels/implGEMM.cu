#include <cuda_runtime.h>
#include "convs2d.h"
#include <stdio.h> 
#include <cstdint>
#include <cuda_fp16.h>

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

/*
    为了增大计算访存比，每个线程需要负责计算一个 4 * 4 的矩阵, 每个线程块仍然有 16 * 16 个线程，所以每个线程块需要负责计算 64 * 64 的矩阵。
    这里，（N * Oh * Ow）tile = (K)tile = 64, (CRS)tile = 4.
    这时的 warp tile 策略保持不变。

*/
__global__ void implgemm_kernel_4(param_t param) {
    
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // warp tile
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t mma_tid_x = lane_id % 8;
    const uint32_t mma_tid_y = lane_id / 8;
    uint32_t weight_lds_addr = (warp_id / 2) * 16 + mma_tid_y * 4;  // 一个warp负责 16 * 32 的结果矩阵
    uint32_t input_lds_addr = (warp_id % 2) * 32 + mma_tid_x * 4;

    // 输出结果矩阵, 输出坐标左上角 【weight_lds_addr, input_lds_addr】
    float output_temp[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
    #pragma unroll
        for (int j = 0; j < 4; j++) {
            output_temp[i][j] = 0.0;
        }
    }

    // share memory buffer, 一个线程块负责 64 * 64 的结果矩阵，而且 (CRS)tile = 4, 所以buffer大小为 64 * 4
    // __shared__ float shm_input[4][64];  行主序
    // __shared__ float shm_weight[64][4];  列主序, 这里难道不会增加bank conflict吗？
    __shared__ float shm_input[4 * 64];
    __shared__ float shm_weight[4 * 64];


    // 每个线程仍然负责两个元素的加载，但是坐标不再是shm_input[ty][tx]，而是shm_input[ty][tx]
    uint32_t input_sts_addr = (tx / 64) * 64 + tx % 64; // shm_input[tx / 64][tx % 64]
    uint32_t weight_sts_addr =  (tx % 4) * 64 + tx / 4; // shm_weight[tx / 4][tx % 4] , 这里为什么不是 tx / 4 + tx % 4

    int z = blockIdx.z;

    // 当前线程加载的数据点在输入矩阵 Oh 和 Ow 上的坐标, 注意和上面的矩阵的对应关系
    int pos_oh = (bx * 64 + tx % 64) / param.Ow;    
    int pos_ow = (bx * 64 + tx % 64) % param.Ow;
    int pos_ori_h = pos_oh * param.stride_h - param.pad_h;
    int pos_ori_w = pos_ow * param.stride_w - param.pad_w;
    int input_offset = z * param.h * param.w * param.c;
    int weight_offset = (by * 64 + tx / 4) * param.c * param.kh * param.kw;
    int input_channel_size = param.h * param.w;
    int weight_channel_size = param.kh * param.kw;
    int kernel_size = param.c * weight_channel_size;

    for (int crs = 0; crs < kernel_size; crs += 4) {
        // Laod data
        int weight_offset_tmp = crs + tx % 4;
        if (weight_offset_tmp < kernel_size && weight_offset_tmp >= 0){
            shm_weight[weight_sts_addr] = param.weight[weight_offset + weight_offset_tmp];    // 这里其实也需要判断是否越界
        }
        else {
            shm_weight[weight_sts_addr] = 0.0;
        }
        shm_weight[weight_sts_addr] = param.weight[weight_offset + weight_offset_tmp];

        int cur_c = (crs + tx / 64) / weight_channel_size;
        int cur_ih = ((crs + tx / 64) % weight_channel_size) / param.kw;
        int cur_iw = ((crs + tx / 64) % weight_channel_size) % param.kw;
        int cur_h = pos_ori_h + cur_ih;
        int cur_w = pos_ori_w + cur_iw;
        int input_offset_tmp = cur_c * input_channel_size + cur_h * param.w + cur_w;
        if (cur_h >= 0 && cur_w >= 0 && cur_h < param.h && cur_w < param.w) {
            shm_input[input_sts_addr] = param.input[input_offset_tmp + input_offset];
        }
        else {
            shm_input[input_sts_addr] = 0.0;
        }

        __syncthreads();

        // compute
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            #pragma unroll
            for (int j = 0;j < 4; j++)
            {
                #pragma unroll
                for (int subcrs = 0; subcrs < 4; subcrs++) {
                    output_temp[i][j] += shm_input[input_lds_addr + subcrs * 64 + j] * shm_weight[weight_lds_addr + subcrs * 64 + i];
                }
            }
        }

        __syncthreads();
    }


    // 计算输出偏移
    int output_offset;
    int y = weight_lds_addr + by * 64;
    int x = input_lds_addr + bx * 64;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            output_offset = z * param.Oh * param.Ow * param.k + (y + i) * param.Oh * param.Ow + (x + j);
            if ((x + j) < param.Ow * param.Oh && (y + i) < param.k)
            {
                param.output[output_offset] = output_temp[i][j];
            }
        }
    }


}

/* 
    进一步增大 （N * Oh * Ow）tile = (K)tile = 128, (CRS)tile = 8.
    int blockx = (param.Oh * param.Ow + 127) / 128 ;
    int blocky = (param.k + 127) / 128;
    int blockz = param.n;
    int threadx = 256;         
    int thready = 1;
    int threadz = 1;
*/
__global__ void implgemm_kernel_4_v2(param_t param) {
    uint32_t tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // warp tile
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t mma_tid_x = lane_id % 8;
    const uint32_t mma_tid_y = lane_id / 8;

    // 每个线程需要负责一个 8 * 8 的矩阵， 实际上这里划分为 4个 4 * 4 的矩阵
    uint32_t input_lds_addr = (warp_id % 2) * (8 * 8) + mma_tid_x * 4 ;
    uint32_t weight_lds_addr = (warp_id / 2) * (8 * 4) + mma_tid_y * 4;

    // share memory buffer, 每个线程需要负责加载 4 * 2 数据
    __shared__ float shm_weight[8 * 128];    // 列主序 shm_weight[4][32][8]
    __shared__ float shm_input[128 * 8];   // 行主序 shm_input[8][4][32]

    uint32_t weight_sts_addr = (tx % 8) * 128 + (tx / 8) * 4 ;  // shm_weight[:4][tx / 8][tx % 8]
    uint32_t input_sts_addr = (tx / 32) * 128 + (tx % 32);  // shm_input[tx / 32][：4][tx % 32]
    
    // 当前线程加载的数据点在输入矩阵 Oh 和 Ow 上的坐标, 注意和上面的矩阵的对应关系
    int pos_ori_h[4];
    int pos_ori_w[4];
    # pragma unroll
    for (int i = 0; i < 4; i++) {
        pos_ori_h[i] = ((bx * 128 + tx % 32 + i * 32) / param.Ow) * param.stride_h - param.pad_h;
        pos_ori_w[i] = ((bx * 128 + tx % 32 + i * 32) % param.Ow) * param.stride_w - param.pad_w;
    }

    // 计算对应加载数据所在矩阵的偏移
    int z = blockIdx.z;
    int input_offset = z * param.h * param.w * param.c;
    int weight_offset = (by * 128 + tx / 8 * 4) * param.c * param.kh * param.kw;
    int input_channel_size = param.h * param.w;
    int weight_channel_size = param.kh * param.kw;
    int kernel_size = param.c * weight_channel_size;


    // 初始化 输出矩阵 
    float output_temp[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
    #pragma unroll
        for (int j = 0; j < 8; j++) {
            output_temp[i][j] = 0.0;
        }
    }

    for (int crs = 0; crs < kernel_size; crs += 8) {
        // 加载数据
        int weight_offset_tmp = crs + tx % 8;
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            // if ( weight_offset_tmp < kernel_size ) {
            //     shm_weight[weight_sts_addr + i] = param.weight[weight_offset + weight_offset_tmp + i * kernel_size];
            // } 
            // else {
            //     shm_weight[weight_sts_addr + i] = 0.0;
            // }   
            shm_weight[weight_sts_addr + i] = param.weight[weight_offset + weight_offset_tmp + i * kernel_size]; // 不清楚为什么不判断越界也可以
        }

        int cur_c = (crs + tx / 32) / weight_channel_size;
        int cur_ih = ((crs + tx / 32) % weight_channel_size) / param.kw;
        int cur_iw = ((crs + tx / 32) % weight_channel_size) % param.kw;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int cur_h = pos_ori_h[i] + cur_ih;
            int cur_w = pos_ori_w[i] + cur_iw;
            int input_offset_tmp = cur_c * input_channel_size + cur_h * param.w + cur_w;

            if (cur_h >= 0 && cur_w >= 0 && cur_h < param.h && cur_w < param.w) {
                shm_input[input_sts_addr + i * 32] = param.input[input_offset_tmp + input_offset];
            }
            else {
                shm_input[input_sts_addr + i * 32] = 0.0;
            }
        }
        
        __syncthreads();

        // 计算数据
        #pragma unroll
        for (int subcrs = 0; subcrs < 8; subcrs++) {
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                #pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    output_temp[i][j] += shm_input[input_lds_addr + subcrs * 128 + j] *shm_weight[weight_lds_addr + subcrs * 128 + i];
                    output_temp[i][j + 4] += shm_input[input_lds_addr + subcrs * 128 + j + 32] *shm_weight[weight_lds_addr + subcrs * 128 + i];
                    output_temp[i + 4][j] += shm_input[input_lds_addr + subcrs * 128 + j] *shm_weight[weight_lds_addr + subcrs * 128 + i + 16];
                    output_temp[i + 4][j + 4] += shm_input[input_lds_addr + subcrs * 128 + j + 32] *shm_weight[weight_lds_addr + subcrs * 128 + i + 16];
                }
            }
        }

        __syncthreads();
    }


    // 计算输出偏移
    int output_offset;
    int y = weight_lds_addr + by * 128;
    int x = input_lds_addr + bx * 128;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            output_offset = z * param.Oh * param.Ow * param.k + (y + i) * param.Oh * param.Ow + x + j;
            if ((x + j) < param.Ow * param.Oh && (y + i) < param.k)
            {
                param.output[output_offset] = output_temp[i][j];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i) * param.Oh * param.Ow + x + j + 32;
            if ((x + j + 32) < param.Ow * param.Oh && (y + i) < param.k)
            {
                param.output[output_offset] = output_temp[i][j + 4];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i + 16) * param.Oh * param.Ow + x + j;
            if ((x + j) < param.Ow * param.Oh && (y + i + 16) < param.k)
            {
                param.output[output_offset] = output_temp[i + 4][j];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i + 16) * param.Oh * param.Ow + x + j + 32;
            if ((x + j + 32) < param.Ow * param.Oh && (y + i + 16) < param.k)
            {
                param.output[output_offset] = output_temp[i + 4][j + 4];
            }
        }
    }

}


/*
    引入了更多的优化细节：
        1. 通过padding，解决了shm_weight sts的bank冲突
        2. 分离ldg 和 sts 指令，优化流水线，充分利用内存结构的优点
        3. 重现排布warp内线程，使用z字排布，充分利用广播效果
*/
__global__ void implgemm_kernel_4_v3(param_t param) {
    uint32_t tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // warp tile, z字排布
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    // 每个线程需要负责一个 8 * 8 的矩阵， 实际上这里划分为 4个 4 * 4 的矩阵
    uint32_t input_lds_addr = (warp_id % 2) * (8 * 8) + mma_tid_x * 4 ;
    uint32_t weight_lds_addr = (warp_id / 2) * (8 * 4) + mma_tid_y * 4;

    // share memory buffer, 每个线程需要负责加载 4 * 2 数据
    __shared__ float shm_weight[8 * 132];    // 列主序 shm_weight[4][32][8]
    __shared__ float shm_input[128 * 8];   // 行主序 shm_input[8][4][32]

    uint32_t weight_sts_addr = (tx % 8) * 132 + (tx / 8) * 4 ;  // shm_weight[:4][tx / 8][tx % 8]
    uint32_t input_sts_addr = (tx / 32) * 128 + (tx % 32);  // shm_input[tx / 32][：4][tx % 32]
    
    // 当前线程加载的数据点在输入矩阵 Oh 和 Ow 上的坐标, 注意和上面的矩阵的对应关系
    int pos_ori_h[4];
    int pos_ori_w[4];
    # pragma unroll
    for (int i = 0; i < 4; i++) {
        pos_ori_h[i] = ((bx * 128 + tx % 32 + i * 32) / param.Ow) * param.stride_h - param.pad_h;
        pos_ori_w[i] = ((bx * 128 + tx % 32 + i * 32) % param.Ow) * param.stride_w - param.pad_w;
    }

    // 计算对应加载数据所在矩阵的偏移
    int z = blockIdx.z;
    int input_offset = z * param.h * param.w * param.c;
    int weight_offset = (by * 128 + tx / 8 * 4) * param.c * param.kh * param.kw;
    int input_channel_size = param.h * param.w;
    int weight_channel_size = param.kh * param.kw;
    int kernel_size = param.c * weight_channel_size;


    // 初始化 输出矩阵 , 中间矩阵
    float weight_temp[8];
    float input_temp[8];
    float output_temp[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
    #pragma unroll
        for (int j = 0; j < 8; j++) {
            output_temp[i][j] = 0.0;
        }
    }

    // 分离 ldg 和 sts
    float weight_ldg_reg[4];
    float input_ldg_reg[4];

    for (int crs = 0; crs < kernel_size; crs += 8) {
        // 加载数据 
        // ldg stage
        int weight_offset_tmp = crs + tx % 8;
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            if ( weight_offset_tmp < kernel_size && by * 128 + tx / 8 * 4 + i < param.k ) {
                weight_ldg_reg[i] = param.weight[weight_offset + weight_offset_tmp + i * kernel_size];
            } 
            else {
                weight_ldg_reg[i] = 0.0;
            }   
            // weight_ldg_reg[i] = param.weight[weight_offset + weight_offset_tmp + i * kernel_size]; // 不清楚为什么不判断越界也可以
        }

        int cur_c = (crs + tx / 32) / weight_channel_size;
        int cur_ih = ((crs + tx / 32) % weight_channel_size) / param.kw;
        int cur_iw = ((crs + tx / 32) % weight_channel_size) % param.kw;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int cur_h = pos_ori_h[i] + cur_ih;
            int cur_w = pos_ori_w[i] + cur_iw;
            int input_offset_tmp = cur_c * input_channel_size + cur_h * param.w + cur_w;

            if (cur_h >= 0 && cur_w >= 0 && cur_h < param.h && cur_w < param.w) {
                // shm_input[input_sts_addr + i * 32] = param.input[input_offset_tmp + input_offset];
                input_ldg_reg[i] = param.input[input_offset_tmp + input_offset];
            }
            else {
                input_ldg_reg[i] = 0.0;
            }
        }

        // sts stage
        for (int i = 0; i < 4; i++){
            shm_input[input_sts_addr + i * 32] = input_ldg_reg[i];
            shm_weight[weight_sts_addr + i] = weight_ldg_reg[i];
        }
        
        __syncthreads();

        // 计算数据
        // 读取share memory到reg
        #pragma unroll
        for (int subcrs = 0; subcrs < 8; subcrs++) {
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                weight_temp[i] = shm_weight[weight_lds_addr + subcrs * 132 + i];
                weight_temp[i + 4] = shm_weight[weight_lds_addr + subcrs * 132 + i + 16];
            }
            #pragma unroll
            for (int i = 0; i < 4;i++)
            {
                input_temp[i] = shm_input[input_lds_addr + subcrs * 128 + i];
                input_temp[i + 4] = shm_input[input_lds_addr + subcrs * 128 + i + 32];
            }

            // 转换为外积
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (i == 0 && j == 0     && weight_lds_addr + by * 128 == 0 && input_lds_addr + bx * 128 == 0){
                        printf("step %d : %f += %f * %f\n",subcrs + crs, output_temp[i][j], input_temp[j] , weight_temp[i]);
                    }
                    output_temp[i][j] += weight_temp[i] * input_temp[j];

                }
            }
        }

        __syncthreads();
    }


    // 计算输出偏移
    int output_offset;
    int y = weight_lds_addr + by * 128;
    int x = input_lds_addr + bx * 128;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            output_offset = z * param.Oh * param.Ow * param.k + (y + i) * param.Oh * param.Ow + x + j;
            if ((x + j) < param.Ow * param.Oh && (y + i) < param.k)
            {
                if (param.output[output_offset] != output_temp[i][j])
                {
                    i = i;
                }
                param.output[output_offset] = output_temp[i][j];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i) * param.Oh * param.Ow + x + j + 32;
            if ((x + j + 32) < param.Ow * param.Oh && (y + i) < param.k)
            {
                param.output[output_offset] = output_temp[i][j + 4];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i + 16) * param.Oh * param.Ow + x + j;
            if ((x + j) < param.Ow * param.Oh && (y + i + 16) < param.k)
            {
                param.output[output_offset] = output_temp[i + 4][j];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i + 16) * param.Oh * param.Ow + x + j + 32;
            if ((x + j + 32) < param.Ow * param.Oh && (y + i + 16) < param.k)
            {
                param.output[output_offset] = output_temp[i + 4][j + 4];
            }
        }
    }

}

/* 
    使用 DoubleBuffering 来掩盖访存延迟
*/
__global__ void implgemm_kernel_5(param_t param) {
    uint32_t tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // warp tile, z字排布
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    // 每个线程需要负责一个 8 * 8 的矩阵， 实际上这里划分为 4个 4 * 4 的矩阵
    uint32_t input_lds_addr = (warp_id % 2) * (8 * 8) + mma_tid_x * 4 ;
    uint32_t weight_lds_addr = (warp_id / 2) * (8 * 4) + mma_tid_y * 4;
    int y = weight_lds_addr + by * 128;
    int x = input_lds_addr + bx * 128;

    // share memory buffer, 每个线程需要负责加载 4 * 2 数据
    __shared__ float shm_weight[2][8 * 132];    // 列主序 shm_weight[4][32][8]
    __shared__ float shm_input[2][128 * 8];   // 行主序 shm_input[8][4][32]

    uint32_t weight_sts_addr = (tx % 8) * 132 + (tx / 8) * 4 ;  // shm_weight[:4][tx / 8][tx % 8]
    uint32_t input_sts_addr = (tx / 32) * 128 + (tx % 32);  // shm_input[tx / 32][：4][tx % 32]
    
    // 当前线程加载的数据点在输入矩阵 Oh 和 Ow 上的坐标, 注意和上面的矩阵的对应关系
    int pos_ori_h[4];
    int pos_ori_w[4];
    # pragma unroll
    for (int i = 0; i < 4; i++) {
        pos_ori_h[i] = ((bx * 128 + tx % 32 + i * 32) / param.Ow) * param.stride_h - param.pad_h;
        pos_ori_w[i] = ((bx * 128 + tx % 32 + i * 32) % param.Ow) * param.stride_w - param.pad_w;
    }

    // 计算对应加载数据所在矩阵的偏移
    int z = blockIdx.z;
    int input_offset = z * param.h * param.w * param.c;
    int weight_offset = (by * 128 + tx / 8 * 4) * param.c * param.kh * param.kw;
    int input_channel_size = param.h * param.w;
    int weight_channel_size = param.kh * param.kw;
    int kernel_size = param.c * weight_channel_size;


    // 初始化 输出矩阵 , 中间矩阵
    int write_flag = 1;
    float weight_temp[2][8];
    float input_temp[2][8];
    float output_temp[8][8];
    // float weight_temp[8];
    // float input_temp[8];
    // float output_temp[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
    #pragma unroll
        for (int j = 0; j < 8; j++) {
            output_temp[i][j] = 0.0;
        }
    }

    // 分离 ldg 和 sts
    float weight_ldg_reg[4];
    float input_ldg_reg[4];

    /*
        先预加载一个数据 
    */
   // 加载数据 
    // ldg stage
    int crs = 0;
    int weight_offset_tmp = crs + tx % 8;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        if ( weight_offset_tmp < kernel_size && by * 128 + tx / 8 * 4 + i < param.k ) {
            weight_ldg_reg[i] = param.weight[weight_offset + weight_offset_tmp + i * kernel_size];
        } 
        else {
            weight_ldg_reg[i] = 0.0;
        }   
        // weight_ldg_reg[i] = param.weight[weight_offset + weight_offset_tmp + i * kernel_size]; // 不清楚为什么不判断越界也可以
    }

    int cur_c = (crs + tx / 32) / weight_channel_size;
    int cur_ih = ((crs + tx / 32) % weight_channel_size) / param.kw;
    int cur_iw = ((crs + tx / 32) % weight_channel_size) % param.kw;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int cur_h = pos_ori_h[i] + cur_ih;
        int cur_w = pos_ori_w[i] + cur_iw;
        int input_offset_tmp = cur_c * input_channel_size + cur_h * param.w + cur_w;

        if (cur_h >= 0 && cur_w >= 0 && cur_h < param.h && cur_w < param.w) {
            // shm_input[input_sts_addr + i * 32] = param.input[input_offset_tmp + input_offset];
            input_ldg_reg[i] = param.input[input_offset_tmp + input_offset];
        }
        else {
            input_ldg_reg[i] = 0.0;
        }
    }

    // sts
    for (int i = 0; i < 4; i++){
        shm_input[0][input_sts_addr + i * 32] = input_ldg_reg[i];
        shm_weight[0][weight_sts_addr + i] = weight_ldg_reg[i];
    }
    __syncthreads();

    //lds stage , for subcrs = 0
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        input_temp[0][i] = shm_input[0][input_lds_addr + i];
        input_temp[0][i + 4] = shm_input[0][input_lds_addr + i + 32];
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        weight_temp[0][i] = shm_weight[0][weight_lds_addr + i];
        weight_temp[0][i + 4] = shm_weight[0][weight_lds_addr + i + 16];
    }
    
    // 主循环，注意每个循环内 负责一个 CRS tile 的计算，以及下一个循环需要的数据ldf + sts + (下个循环第一个lds使用的)
    // 
    // main loop
    for (crs = 0; crs < kernel_size; crs += 8) {
        // 加载数据 
        // ldg stage
        int weight_offset_tmp = crs + 8 + tx % 8;
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            if ( weight_offset_tmp < kernel_size && by * 128 + tx / 8 * 4 + i < param.k ) {
                weight_ldg_reg[i] = param.weight[weight_offset + weight_offset_tmp + i * kernel_size];
            } 
            else {
                weight_ldg_reg[i] = 0.0;
            }   
            // weight_ldg_reg[i] = param.weight[weight_offset + weight_offset_tmp + i * kernel_size]; // 不清楚为什么不判断越界也可以
        }

        int cur_c = (crs + 8 + tx / 32) / weight_channel_size;
        int cur_ih = ((crs + 8 + tx / 32) % weight_channel_size) / param.kw;
        int cur_iw = ((crs + 8 + tx / 32) % weight_channel_size) % param.kw;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int cur_h = pos_ori_h[i] + cur_ih;
            int cur_w = pos_ori_w[i] + cur_iw;
            int input_offset_tmp = cur_c * input_channel_size + cur_h * param.w + cur_w;

            if (cur_h >= 0 && cur_w >= 0 && cur_h < param.h && cur_w < param.w) {
                // shm_input[input_sts_addr + i * 32] = param.input[input_offset_tmp + input_offset];
                input_ldg_reg[i] = param.input[input_offset_tmp + input_offset];
            }
            else {
                input_ldg_reg[i] = 0.0;
            }
        }
        
        /*
            lds + compute
            注意，这里其实之前已经提前lds一个数据.
            这里lds部分也做了DoubleBuffering，不仅仅是 ldg 和 sts 部分。
            但是，这里的代码显然会出现数据覆盖的问题，导致数据计算错误。
            ！！！！ 只能弃用这部分的double buffering ，必然存在一定的数据覆盖问题。（搞错了，share memory才会出现数据覆盖的问题）
        */
        int load_flag = write_flag ^ 1; // 对应这个循环计算使用的数据标志位
        #pragma unroll
        for (int subcrs = 0; subcrs < 8 - 1; subcrs++) {
            // lds下个循环使用的数据
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                weight_temp[(subcrs + 1) % 2][i] = shm_weight[load_flag][weight_lds_addr + (subcrs + 1) * 132 + i]; 
                weight_temp[(subcrs + 1) % 2][i + 4] = shm_weight[load_flag][weight_lds_addr + (subcrs + 1) * 132 + i + 16]; 
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                input_temp[(subcrs + 1) % 2][i] = shm_input[load_flag][input_lds_addr + (subcrs + 1) * 128 + i]; 
                input_temp[(subcrs + 1) % 2][i + 4] = shm_input[load_flag][input_lds_addr + (subcrs + 1) * 128 + i + 32];
            }

            // compute
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    // if (z == 0 && y + i == 0 && ((x + j == 128 && j < 4) || (x + j - 4 == 128 + 32 && j >= 4))) {
                    //     printf("step %d : %.9f += %.9f * %.9f\n",subcrs + crs, output_temp[i][j], input_temp[subcrs % 2][j] , weight_temp[subcrs % 2][i]);
                    // }
                    output_temp[i][j] += input_temp[subcrs % 2][j] * weight_temp[subcrs % 2][i];
                }
            }
        }
        
        // int load_flag = write_flag ^ 1; // 对应这个循环计算使用的数据标志位
        // #pragma unroll
        // for (int subcrs = 0; subcrs < 8; subcrs++) {
        //     // lds下个循环使用的数据
        //     #pragma unroll
        //     for (int i = 0; i < 4; i++) {
        //         weight_temp[i] = shm_weight[load_flag][weight_lds_addr + subcrs * 132 + i]; 
        //         weight_temp[i + 4] = shm_weight[load_flag][weight_lds_addr + subcrs * 132 + i + 16]; 
        //     }

        //     #pragma unroll
        //     for (int i = 0; i < 4; i++) {
        //         input_temp[i] = shm_input[load_flag][input_lds_addr + subcrs * 128 + i]; 
        //         input_temp[i + 4] = shm_input[load_flag][input_lds_addr + subcrs * 128 + i + 32];
        //     }

        //     // compute
        //     #pragma unroll
        //     for (int i = 0; i < 8; i++) {
        //         #pragma unroll
        //         for (int j = 0; j < 8; j++) {
        //             // if (i == 0 && j == 0 && weight_lds_addr + by * 128 == 0 && input_lds_addr + bx * 128 == 0){
        //             //     printf("step %d : %f += %f * %f\n",subcrs + crs, output_temp[i][j], input_temp[subcrs % 2][j] , weight_temp[subcrs % 2][i]);
        //             // }
        //             output_temp[i][j] += input_temp[j] * weight_temp[i];
        //         }
        //     }
        // }

        /*
            上面其实还有一个循环没有计算，这里在前面塞了一个sts阶段掩藏延迟，和main loop外的sts一样的，为下一个 lds + compute 
            第一个循环计算预加载数据。
        */
        for (int i = 0; i < 4; i++)
        {
            shm_weight[write_flag][weight_sts_addr + i] = weight_ldg_reg[i];
            shm_input[write_flag][input_sts_addr + i * 32] = input_ldg_reg[i];
        }

        __syncthreads();  // 必须等待数据加载完成，不然lds会出错

        write_flag = write_flag ^ 1;

        // lds下个循环使用的数据
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            weight_temp[0][i] = shm_weight[load_flag ^ 1][weight_lds_addr + i]; 
            weight_temp[0][i + 4] = shm_weight[load_flag ^ 1][weight_lds_addr + i + 16]; 
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            input_temp[0][i] = shm_input[load_flag ^ 1][input_lds_addr + i]; 
            input_temp[0][i + 4] = shm_input[load_flag ^ 1][input_lds_addr + i + 32];
        }

        /*
            好啦，终于可以把 当前循环最后一个subcrs的数据计算完了.
            subcrs = 7
        */
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                // if (z == 0 && y + i == 0 && ((x + j == 128 && j < 4) || (x + j - 4 == 128 + 32 && j >= 4))){
                //     printf("step %d : %.9f += %.9f * %.9f\n",7 + crs, output_temp[i][j], input_temp[1][j] , weight_temp[1][i]);
                // } 
                output_temp[i][j] += input_temp[1][j] * weight_temp[1][i];
            }
        }

    }


    // 计算输出偏移
    int output_offset;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            output_offset = z * param.Oh * param.Ow * param.k + (y + i) * param.Oh * param.Ow + x + j;
            if ((x + j) < param.Ow * param.Oh && (y + i) < param.k)
            {
                // if (param.output[output_offset] != output_temp[i][j]){
                //     printf("output_temp[%d][%d] = %f, param.output[%d] = %f\n", i, j, output_temp[i][j], output_offset, param.output[output_offset]);
                // }
                param.output[output_offset] = output_temp[i][j];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i) * param.Oh * param.Ow + x + j + 32;
            if ((x + j + 32) < param.Ow * param.Oh && (y + i) < param.k)
            {
                // if (param.output[output_offset] != output_temp[i][j + 4]){
                //     printf("output_temp[%d][%d] = %f, param.output[%d] = %f\n", i, j, output_temp[i][j + 4], output_offset, param.output[output_offset]);
                // }
                param.output[output_offset] = output_temp[i][j + 4];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i + 16) * param.Oh * param.Ow + x + j;
            if ((x + j) < param.Ow * param.Oh && (y + i + 16) < param.k)
            {
                param.output[output_offset] = output_temp[i + 4][j];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i + 16) * param.Oh * param.Ow + x + j + 32;
            if ((x + j + 32) < param.Ow * param.Oh && (y + i + 16) < param.k)
            {
                param.output[output_offset] = output_temp[i + 4][j + 4];
            }
        }
    }

}

__global__ void implgemm_fp16_kernel_4_v2(param_t param) {
    uint32_t tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // warp tile
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t mma_tid_x = lane_id % 8;
    const uint32_t mma_tid_y = lane_id / 8;

    // 每个线程需要负责一个 8 * 8 的矩阵， 实际上这里划分为 4个 4 * 4 的矩阵
    uint32_t input_lds_addr = (warp_id % 2) * (8 * 8) + mma_tid_x * 4 ;
    uint32_t weight_lds_addr = (warp_id / 2) * (8 * 4) + mma_tid_y * 4;

    // share memory buffer, 每个线程需要负责加载 4 * 2 数据
    __shared__ __half shm_weight[8 * 128];    // 列主序 shm_weight[4][32][8]
    __shared__ __half shm_input[128 * 8];   // 行主序 shm_input[8][4][32]

    uint32_t weight_sts_addr = (tx % 8) * 128 + (tx / 8) * 4 ;  // shm_weight[:4][tx / 8][tx % 8]
    uint32_t input_sts_addr = (tx / 32) * 128 + (tx % 32);  // shm_input[tx / 32][：4][tx % 32]
    
    // 当前线程加载的数据点在输入矩阵 Oh 和 Ow 上的坐标, 注意和上面的矩阵的对应关系
    int pos_ori_h[4];
    int pos_ori_w[4];
    # pragma unroll
    for (int i = 0; i < 4; i++) {
        pos_ori_h[i] = ((bx * 128 + tx % 32 + i * 32) / param.Ow) * param.stride_h - param.pad_h;
        pos_ori_w[i] = ((bx * 128 + tx % 32 + i * 32) % param.Ow) * param.stride_w - param.pad_w;
    }

    // 计算对应加载数据所在矩阵的偏移
    int z = blockIdx.z;
    int input_offset = z * param.h * param.w * param.c;
    int weight_offset = (by * 128 + tx / 8 * 4) * param.c * param.kh * param.kw;
    int input_channel_size = param.h * param.w;
    int weight_channel_size = param.kh * param.kw;
    int kernel_size = param.c * weight_channel_size;


    // 初始化 输出矩阵 
    __half output_temp[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
    #pragma unroll
        for (int j = 0; j < 8; j++) {
            output_temp[i][j] = 0.0;
        }
    }

    for (int crs = 0; crs < kernel_size; crs += 8) {
        // 加载数据
        int weight_offset_tmp = crs + tx % 8;
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            // if ( weight_offset_tmp < kernel_size ) {
            //     shm_weight[weight_sts_addr + i] = param.weight[weight_offset + weight_offset_tmp + i * kernel_size];
            // } 
            // else {
            //     shm_weight[weight_sts_addr + i] = 0.0;
            // }   
            shm_weight[weight_sts_addr + i] = param.weight[weight_offset + weight_offset_tmp + i * kernel_size]; // 不清楚为什么不判断越界也可以
        }

        int cur_c = (crs + tx / 32) / weight_channel_size;
        int cur_ih = ((crs + tx / 32) % weight_channel_size) / param.kw;
        int cur_iw = ((crs + tx / 32) % weight_channel_size) % param.kw;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int cur_h = pos_ori_h[i] + cur_ih;
            int cur_w = pos_ori_w[i] + cur_iw;
            int input_offset_tmp = cur_c * input_channel_size + cur_h * param.w + cur_w;

            if (cur_h >= 0 && cur_w >= 0 && cur_h < param.h && cur_w < param.w) {
                shm_input[input_sts_addr + i * 32] = param.input[input_offset_tmp + input_offset];
            }
            else {
                shm_input[input_sts_addr + i * 32] = 0.0;
            }
        }
        
        __syncthreads();

        // 计算数据
        #pragma unroll
        for (int subcrs = 0; subcrs < 8; subcrs++) {
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                #pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    // output_temp[i][j] += __half2float(shm_input[input_lds_addr + subcrs * 128 + j] * shm_weight[weight_lds_addr + subcrs * 128 + i]);
                    // output_temp[i][j + 4] += __half2float(shm_input[input_lds_addr + subcrs * 128 + j + 32] * shm_weight[weight_lds_addr + subcrs * 128 + i]);
                    // output_temp[i + 4][j] += __half2float(shm_input[input_lds_addr + subcrs * 128 + j] * shm_weight[weight_lds_addr + subcrs * 128 + i + 16]);
                    // output_temp[i + 4][j + 4] += __half2float(shm_input[input_lds_addr + subcrs * 128 + j + 32] * shm_weight[weight_lds_addr + subcrs * 128 + i + 16]);
                    output_temp[i][j] += shm_input[input_lds_addr + subcrs * 128 + j] * shm_weight[weight_lds_addr + subcrs * 128 + i];
                    output_temp[i][j + 4] += shm_input[input_lds_addr + subcrs * 128 + j + 32] * shm_weight[weight_lds_addr + subcrs * 128 + i];
                    output_temp[i + 4][j] += shm_input[input_lds_addr + subcrs * 128 + j] * shm_weight[weight_lds_addr + subcrs * 128 + i + 16];
                    output_temp[i + 4][j + 4] += shm_input[input_lds_addr + subcrs * 128 + j + 32] * shm_weight[weight_lds_addr + subcrs * 128 + i + 16];
                }
            }
        }

        __syncthreads();
    }


    // 计算输出偏移
    int output_offset;
    int y = weight_lds_addr + by * 128;
    int x = input_lds_addr + bx * 128;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            output_offset = z * param.Oh * param.Ow * param.k + (y + i) * param.Oh * param.Ow + x + j;
            if ((x + j) < param.Ow * param.Oh && (y + i) < param.k)
            {
                param.output[output_offset] = output_temp[i][j];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i) * param.Oh * param.Ow + x + j + 32;
            if ((x + j + 32) < param.Ow * param.Oh && (y + i) < param.k)
            {
                param.output[output_offset] = output_temp[i][j + 4];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i + 16) * param.Oh * param.Ow + x + j;
            if ((x + j) < param.Ow * param.Oh && (y + i + 16) < param.k)
            {
                param.output[output_offset] = output_temp[i + 4][j];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i + 16) * param.Oh * param.Ow + x + j + 32;
            if ((x + j + 32) < param.Ow * param.Oh && (y + i + 16) < param.k)
            {
                param.output[output_offset] = output_temp[i + 4][j + 4];
            }
        }
    }

}


/*
    使用 DoubleBuffering 来掩盖访存延迟
*/
__global__ void implgemm_fp16_kernel_5(param_t param) {
    uint32_t tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // warp tile, z字排布
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) % 2 + (lane_id % 2);

    // 每个线程需要负责一个 8 * 8 的矩阵， 实际上这里划分为 4个 4 * 4 的矩阵
    uint32_t input_lds_addr = (warp_id % 2) * (8 * 8) + mma_tid_x * 4 ;
    uint32_t weight_lds_addr = (warp_id / 2) * (8 * 4) + mma_tid_y * 4;

    // share memory buffer, 每个线程需要负责加载 4 * 2 数据
    __shared__ __half shm_weight[2][8 * 128];    // 列主序 shm_weight[4][32][8]
    __shared__ __half shm_input[2][132 * 8];   // 行主序 shm_input[8][4][32]

    uint32_t weight_sts_addr = (tx % 8) * 132 + (tx / 8) * 4 ;  // shm_weight[:4][tx / 8][tx % 8]
    uint32_t input_sts_addr = (tx / 32) * 128 + (tx % 32);  // shm_input[tx / 32][：4][tx % 32]
    
    // 当前线程加载的数据点在输入矩阵 Oh 和 Ow 上的坐标, 注意和上面的矩阵的对应关系
    int pos_ori_h[4];
    int pos_ori_w[4];
    # pragma unroll
    for (int i = 0; i < 4; i++) {
        pos_ori_h[i] = ((bx * 128 + tx % 32 + i * 32) / param.Ow) * param.stride_h - param.pad_h;
        pos_ori_w[i] = ((bx * 128 + tx % 32 + i * 32) % param.Ow) * param.stride_w - param.pad_w;
    }

    // 计算对应加载数据所在矩阵的偏移
    int z = blockIdx.z;
    int input_offset = z * param.h * param.w * param.c;
    int weight_offset = (by * 128 + tx / 8 * 4) * param.c * param.kh * param.kw;
    int input_channel_size = param.h * param.w;
    int weight_channel_size = param.kh * param.kw;
    int kernel_size = param.c * weight_channel_size;


    // 初始化 输出矩阵 , 中间矩阵
    int write_flag = 1;
    __half weight_temp[2][8];
    __half input_temp[2][8];
    float output_temp[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
    #pragma unroll
        for (int j = 0; j < 8; j++) {
            output_temp[i][j] = 0.0;
        }
    }

    // 分离 ldg 和 sts
    __half weight_ldg_reg[4];
    __half input_ldg_reg[4];

    /*
        先预加载一个数据 
    */
   // 加载数据 
    // ldg stage
    int crs = 0;
    int weight_offset_tmp = crs + tx % 8;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        if ( weight_offset_tmp < kernel_size && by * 128 + tx / 8 * 4 + i < param.k ) {
            weight_ldg_reg[i] = param.weight[weight_offset + weight_offset_tmp + i * kernel_size];
        } 
        else {
            weight_ldg_reg[i] = 0.0;
        }   
        // weight_ldg_reg[i] = param.weight[weight_offset + weight_offset_tmp + i * kernel_size]; // 不清楚为什么不判断越界也可以
    }

    int cur_c = (crs + tx / 32) / weight_channel_size;
    int cur_ih = ((crs + tx / 32) % weight_channel_size) / param.kw;
    int cur_iw = ((crs + tx / 32) % weight_channel_size) % param.kw;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int cur_h = pos_ori_h[i] + cur_ih;
        int cur_w = pos_ori_w[i] + cur_iw;
        int input_offset_tmp = cur_c * input_channel_size + cur_h * param.w + cur_w;

        if (cur_h >= 0 && cur_w >= 0 && cur_h < param.h && cur_w < param.w) {
            // shm_input[input_sts_addr + i * 32] = param.input[input_offset_tmp + input_offset];
            input_ldg_reg[i] = param.input[input_offset_tmp + input_offset];
        }
        else {
            input_ldg_reg[i] = 0.0;
        }
    }

    // sts
    for (int i = 0; i < 4; i++){
        shm_input[0][input_sts_addr + i * 32] = input_ldg_reg[i];
        shm_weight[0][weight_sts_addr + i] = weight_ldg_reg[i];
    }
    __syncthreads();

    //lds stage , for subcrs = 0
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        input_temp[0][i] = shm_input[0][input_lds_addr + i];
        input_temp[0][i + 4] = shm_input[0][input_lds_addr + i + 16];
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        weight_temp[0][i] = shm_weight[0][weight_lds_addr + i];
        weight_temp[0][i + 4] = shm_weight[0][weight_lds_addr + i + 32];
    }
    
    // 主循环，注意每个循环内 负责一个 CRS tile 的计算，以及下一个循环需要的数据ldf + sts + (下个循环第一个lds使用的)
    // 
    // main loop
    for (crs = 0; crs < kernel_size; crs += 8) {
        // 加载数据 
        // ldg stage
        int weight_offset_tmp = crs + tx % 8;
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            if ( weight_offset_tmp < kernel_size && by * 128 + tx / 8 * 4 + i < param.k ) {
                weight_ldg_reg[i] = param.weight[weight_offset + weight_offset_tmp + i * kernel_size];
            } 
            else {
                weight_ldg_reg[i] = 0.0;
            }   
            // weight_ldg_reg[i] = param.weight[weight_offset + weight_offset_tmp + i * kernel_size]; // 不清楚为什么不判断越界也可以
        }

        int cur_c = (crs + tx / 32) / weight_channel_size;
        int cur_ih = ((crs + tx / 32) % weight_channel_size) / param.kw;
        int cur_iw = ((crs + tx / 32) % weight_channel_size) % param.kw;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int cur_h = pos_ori_h[i] + cur_ih;
            int cur_w = pos_ori_w[i] + cur_iw;
            int input_offset_tmp = cur_c * input_channel_size + cur_h * param.w + cur_w;

            if (cur_h >= 0 && cur_w >= 0 && cur_h < param.h && cur_w < param.w) {
                // shm_input[input_sts_addr + i * 32] = param.input[input_offset_tmp + input_offset];
                input_ldg_reg[i] = param.input[input_offset_tmp + input_offset];
            }
            else {
                input_ldg_reg[i] = 0.0;
            }
        }
        
        /*
            lds + compute
            注意，这里其实之前已经提前lds一个数据.
            这里lds部分也做了DoubleBuffering，不仅仅是 ldg 和 sts 部分。
        */
        int load_flag = write_flag ^ 1; // 对应这个循环计算使用的数据标志位
        #pragma unroll
        for (int subcrs = 0; subcrs < 8 - 1; subcrs++) {
            // lds下个循环使用的数据
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                weight_temp[(subcrs + 1) % 2][i] = shm_weight[load_flag][weight_lds_addr + (subcrs + 1) * 132 + i]; 
                weight_temp[(subcrs + 1) % 2][i + 4] = shm_weight[load_flag][weight_lds_addr + (subcrs + 1) * 132 + i + 16]; 
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                input_temp[(subcrs + 1) % 2][i] = shm_input[load_flag][input_lds_addr + (subcrs + 1) * 128 + i]; 
                input_temp[(subcrs + 1) % 2][i + 4] = shm_input[load_flag][input_lds_addr + (subcrs + 1) * 128 + i + 32];
            }

            // compute
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    output_temp[i][j] += __half2float(input_temp[subcrs % 2][i] * weight_temp[subcrs % 2][j]);
                }
            }
        }

        /*
            上面其实还有一个循环没有计算，这里在前面塞了一个sts阶段掩藏延迟，和main loop外的sts一样的，为下一个 lds + compute 
            第一个循环计算预加载数据。
        */
        for (int i = 0; i < 4; i++)
        {
            shm_weight[write_flag][weight_sts_addr + i] = weight_ldg_reg[i];
            shm_input[write_flag][input_sts_addr + i * 32] = input_ldg_reg[i];
        }

        __syncthreads();  // 必须等待数据加载完成，不然lds会出错

        write_flag = write_flag ^ 1;

        // lds下个循环使用的数据
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            weight_temp[0][i] = shm_weight[load_flag][weight_lds_addr + i]; 
            weight_temp[0][i + 4] = shm_weight[load_flag][weight_lds_addr + i + 16]; 
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            input_temp[0][i] = shm_input[load_flag][input_lds_addr + i]; 
            input_temp[0][i + 4] = shm_input[load_flag][input_lds_addr + i + 32];
        }

        /*
            好啦，终于可以把 当前循环最后一个subcrs的数据计算完了.
            subcrs = 7
        */
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                output_temp[i][j] += __half2float(input_temp[1][i] * weight_temp[1][j]);
            }
        }
    }


    // 计算输出偏移
    int output_offset;
    int y = weight_lds_addr + by * 128;
    int x = input_lds_addr + bx * 128;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            output_offset = z * param.Oh * param.Ow * param.k + (y + i) * param.Oh * param.Ow + x + j;
            if ((x + j) < param.Ow * param.Oh && (y + i) < param.k)
            {
                param.output[output_offset] = output_temp[i][j];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i) * param.Oh * param.Ow + x + j + 32;
            if ((x + j + 32) < param.Ow * param.Oh && (y + i) < param.k)
            {
                param.output[output_offset] = output_temp[i][j + 4];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i + 16) * param.Oh * param.Ow + x + j;
            if ((x + j) < param.Ow * param.Oh && (y + i + 16) < param.k)
            {
                param.output[output_offset] = output_temp[i + 4][j];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i + 16) * param.Oh * param.Ow + x + j + 32;
            if ((x + j + 32) < param.Ow * param.Oh && (y + i + 16) < param.k)
            {
                param.output[output_offset] = output_temp[i + 4][j + 4];
            }
        }
    }

}


void launch_implgemm(param_t param) {

    param.Oh = (param.h - param.kh + 2 * param.pad_h) / param.stride_h + 1;
    param.Ow = (param.w - param.kw + 2 * param.pad_w) / param.stride_w + 1;

    int blockx = (param.Oh * param.Ow + 127) / 128 ;
    int blocky = (param.k + 127) / 128;
    int blockz = param.n;
    int threadx = 256;         
    int thready = 1;
    int threadz = 1;

    dim3 block(threadx, thready, threadz);
    dim3 grid(blockx, blocky, blockz);
    implgemm_kernel_5<<<grid, block>>>(param);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(cuda_err));
    }

}


