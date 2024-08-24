#ifndef CONVS2D_H
#define CONVS2D_H

#include <cuda_fp16.h>

typedef struct {
    float* input;
    float* weight;
    float* bias;
    float* output;
    float* input_host;
    float* weight_host;
    float* bias_host;
    float* output_host;
    float* output_benchmark;
    float* im2col_input;
    unsigned int n;
    unsigned int c;
    unsigned int h;         // 高
    unsigned int w;         // 宽
    unsigned int k;         // 卷积核个数
    unsigned int kh;        // 卷积核高
    unsigned int kw;        // 卷积核宽
    unsigned int stride_h;  // 步长高
    unsigned int stride_w;  // 步长宽
    unsigned int pad_h;     // 填充高
    unsigned int pad_w;     // 填充宽
    unsigned int Oh;        // 输出高
    unsigned int Ow;        // 输出宽
} param_t;


/*
    CPU 一般把float16转换为float32, 或者使用 uint16_t 存储float16数据。
*/
typedef struct {
    __half* input;
    __half* weight;
    __half* bias;
    __half* output;
    __half* im2col_input;
    float* input_host;
    float* weight_host;
    float* bias_host;
    float* output_host;         // 输出就转换为float32，方便对比结果（感觉一般算子设计这里都会使用uint16_t暂存数据）
    float* output_benchmark;
    unsigned int n;
    unsigned int c;
    unsigned int h;         // 高
    unsigned int w;         // 宽
    unsigned int k;         // 卷积核个数
    unsigned int kh;        // 卷积核高
    unsigned int kw;        // 卷积核宽
    unsigned int stride_h;  // 步长高
    unsigned int stride_w;  // 步长宽
    unsigned int pad_h;     // 填充高
    unsigned int pad_w;     // 填充宽
    unsigned int Oh;        // 输出高
    unsigned int Ow;        // 输出宽
} param_16t;



void launch_implgemm(param_t param);

void im2col_gemm_fp16(param_16t param);

void im2col(param_16t param);

void direct_conv2dCuDNN(param_t param);
void direct_conv2dCuDNN_fp16(param_16t param);

#endif