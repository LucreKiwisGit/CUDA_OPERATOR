#include "verify.h"
#include <convs2d.h>
#include <omp.h>

void direct_conv2dCpu(param_t param) {

    param.Oh = (param.h - param.kh + 2 * param.pad_h) / param.stride_h + 1;
    param.Ow = (param.w - param.kw + 2 * param.pad_w) / param.stride_w + 1;

    #pragma omp parallel for collapse(4)
    for (int n = 0; n < param.n; n++) {
        for (int k = 0; k < param.k; k++) {
            for (int oh = 0; oh < param.Oh; oh++) {
                for (int ow = 0; ow < param.Ow; ow++) {
                    float sum = 0;
                    for (int c = 0; c < param.c; c++) {
                        for (int kh = 0; kh < param.kh; kh++) {
                            for (int kw = 0; kw < param.kw; kw++) {
                                int ih = oh * param.stride_h - param.pad_h + kh;
                                int iw = ow * param.stride_w - param.pad_w + kw;
                                if (iw >= 0 && iw < param.w && ih >= 0 && ih < param.h)
                                {
                                    sum += param.input_host[n * param.c * param.h * param.w + c * param.h * param.w + ih * param.w + iw] * param.weight_host[k * param.c * param.kh * param.kw + c * param.kh * param.kw + kh * param.kw + kw];
                                }
                            }
                        }
                    }
                    param.output_benchmark[n * param.k * param.Oh * param.Ow + k * param.Oh * param.Ow + oh * param.Ow + ow] = sum + param.bias_host[k];
                }
        }
    } 
    }
}