#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include "convs2d.h"
#include <cuda_fp16.h>

void direct_conv2dCuDNN(param_t param)
{
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc,
                                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                param.n, param.c, param.h, param.w);

    param.Oh = (param.h - param.kh + 2 * param.pad_h) / param.stride_h + 1;
    param.Ow = (param.w - param.kw + 2 * param.pad_w) / param.stride_w + 1;

    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc,
                                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                param.n, param.k, param.Oh, param.Ow);
    
    cudnnFilterDescriptor_t filter_desc;
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnSetFilter4dDescriptor(filter_desc,
                                CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                param.k, param.c, param.kh, param.kw);

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    param.pad_h, param.pad_w,
                                    param.stride_h, param.stride_w,
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);
    
    cudnnConvolutionFwdAlgoPerf_t perfResults[10];
    int returnedAlgoCount;

    cudnnGetConvolutionForwardAlgorithm_v7(cudnn, input_desc, filter_desc, conv_desc, output_desc,
                                        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &returnedAlgoCount, perfResults);
    
    cudnnConvolutionFwdAlgo_t algo;
    
    if (returnedAlgoCount > 0) {
        cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo; // 选择第一个算法
    } else {
        // 处理没有返回算法的情况
    }


    size_t workspace_bytes = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_bytes);

    void* workspace = nullptr;
    cudaMalloc(&workspace, workspace_bytes);

    const float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, input_desc, param.input, filter_desc, param.weight,
                            conv_desc, algo, workspace, workspace_bytes, &beta, output_desc, param.output);

    cudnnDestroy(cudnn);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return;
}

void direct_conv2dCuDNN_fp16(param_16t param)
{
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc,
                                CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 
                                param.n, param.c, param.h, param.w);

    param.Oh = (param.h - param.kh + 2 * param.pad_h) / param.stride_h + 1;
    param.Ow = (param.w - param.kw + 2 * param.pad_w) / param.stride_w + 1;

    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc,
                                CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 
                                param.n, param.k, param.Oh, param.Ow);
    
    cudnnFilterDescriptor_t filter_desc;
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnSetFilter4dDescriptor(filter_desc,
                                CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
                                param.k, param.c, param.kh, param.kw);

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    param.pad_h, param.pad_w,
                                    param.stride_h, param.stride_w,
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_HALF);
    
    cudnnConvolutionFwdAlgoPerf_t perfResults[10];
    int returnedAlgoCount;

    cudnnGetConvolutionForwardAlgorithm_v7(cudnn, input_desc, filter_desc, conv_desc, output_desc,
                                        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &returnedAlgoCount, perfResults);
    
    cudnnConvolutionFwdAlgo_t algo;
    
    if (returnedAlgoCount > 0) {
        cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo; // 选择第一个算法
    } else {
        // 处理没有返回算法的情况
    }


    size_t workspace_bytes = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_bytes);

    void* workspace = nullptr;
    cudaMalloc(&workspace, workspace_bytes);

    const __half alpha = 1.0, beta = 0.0    ;
    cudnnConvolutionForward(cudnn, &alpha, input_desc, param.input, filter_desc, param.weight,
                            conv_desc, algo, workspace, workspace_bytes, &beta, output_desc, param.output);

    cudaFree(workspace);
    cudnnDestroy(cudnn);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return;
}