#ifndef CONVS2D_H
#define CONVS2D_H

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

void launch_implgemm(param_t param);

void direct_conv2dCuDNN(param_t param);

#endif