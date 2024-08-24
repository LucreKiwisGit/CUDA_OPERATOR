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

// void conv2dcpu(_Float16* pin, _Float16* pwei, _Float16* pout, int n, int c, int h, int w, int k, int r, int s, int u, int v,  int p, int q)
// {
//     int oh = (h + 2*p - r)/u + 1;
//     int ow = (w + 2*q - s)/v + 1;
    
//     for(int nNum = 0; nNum < n; nNum++)
//     {
//         for(int kNum = 0; kNum< k; kNum++)
//         {
//             for(int i=0; i<oh; i++)
//             {
//                 for(int j = 0; j< ow; j++)
//                 { 
//                     double sum = 0.0;
//                     int posh = i*u - p;
//                     int posw = j*v - q;

//                     for(int cNum = 0; cNum < c; cNum++)
//                     {                       
//                         for(int khNum = 0; khNum < r; khNum++)
//                         {
//                             for(int kwNum = 0; kwNum < s; kwNum++)
//                             {
//                                 int posh_ori = posh + khNum;
//                                 int posw_ori = posw + kwNum;
//                                 if(posw_ori >= 0 && posh_ori >= 0 && posw_ori < w  && posh_ori < h)
//                                 {
//                                     sum += (double)(pin[nNum*c*h*w + cNum*(w*h)+ posh_ori*w + posw_ori] * pwei[kNum*r*s*c + cNum*r*s + khNum*s + kwNum]);
//                                 }
//                             }                       
//                         }
//                     }

//                     pout[nNum*k*oh*ow + kNum*oh*ow + i*ow + j] = (_Float16)sum;
//                 }
//             }
//         }
//     }
// }