#include <assert.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define MAX(a, b) ((a) >= (b) ? (a) : (b))
#define MIN(a, b) ((a) >= (b) ? (b) : (a))
#define ABS(x) ((x) >= 0 ? (x) : -(x))

const int WARP_SIZE = 32;
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

bool is_close(float input, float other, float rtol, float atol) { return ABS(input - other) <= atol + rtol * ABS(other); }

void cpu_conv(float *in, float *out, float *cpu_kernel, int IC, int IH, int IW, int OC, int OH, int OW, int KH, int KW) {
    int out_pos, in_pos, kernel_pos;
    for (int oc = 0; oc < OC; ++oc) {
        for (int i = 0; i < OH; ++i) {
            for (int j = 0; j < OW; ++j) {
                float val = 0;
                out_pos = oc * OH * OW + OFFSET(i, j, OW);
                for (int ic = 0; ic < IC; ++ic) {
                    for (int ii = 0; ii < KH; ++ii) {
                        for (int jj = 0; jj < KW; ++jj) {
                            if (i + ii >= IH || j + jj >= IW) continue;
                            in_pos = ic * IH * IW + OFFSET(i + ii, j + jj, IW);
                            kernel_pos = oc * IC * KH * KW + ic * KH * KW + OFFSET(ii, jj, KW);
                            val += in[in_pos] * cpu_kernel[kernel_pos];
                        }
                    }
                }
                out[out_pos] = val;
            }
        }
    }
}

template <const int KH, const int KW, const int WARPS_PER_BLOCK>
__global__ void implicit_gemm_conv(__half *input, __half *output, __half *kernel, const int N, const int IC, const int IH, const int IW, const int OC, const int OH, const int OW) {
    const int UNROLLED_KERNEL_SIZE = KH * KW;          // 每个卷积核单通道展开后的大小
    const int SLICE_SIZE = IC * UNROLLED_KERNEL_SIZE;  // 每个卷积核展开后的大小

    const int WMMA_INPUT_TILE_SIZE = WMMA_M * WMMA_K;
    const int WMMA_FILTER_TILE_SIZE = WMMA_K * WMMA_N;

    int GEMM_M = N * OH * OW;
    int GEMM_N = OC;
    int GEMM_K = SLICE_SIZE;

    const int out_n_stride = OH * OW;
    const int out_h_stride = OW;
    const int out_w_stride = 1;

    const int in_n_stride = IC * IH * IW;
    const int in_c_stride = IH * IW;
    const int in_h_stride = IW;
    const int in_w_stride = 1;

    const int kernel_n_stride = IC * KH * KW;
    const int kernel_c_stride = KH * KW;
    const int kernel_h_stride = KW;
    const int kernel_w_stride = 1;

    // 一个 block 中有 256 个线程，分成 256/32=8 个 warp
    const int global_warp_id_x = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;    // 每个 warp 在 x 方向的全局 id
    const int global_warp_id_y = (blockIdx.y * blockDim.y + threadIdx.y);                // 每个 warp 在 y 方向的全局 id
    const int block_warp_id_x = threadIdx.x / WARP_SIZE;                                 // 每个 warp 在所属 block 中 x 方向的 id
    const int block_warp_id_y = threadIdx.y;                                             // 每个 warp 在所属 block 中 y 方向的 id
    const int intra_warp_thread_id = threadIdx.x % WARP_SIZE;                            // 每个 thread 在 warp 中的 id
    const int num_warps_x = blockDim.x / WARP_SIZE;                                      // 每个 block 在 x 方向上的 warp 个数
    const int block_warp_id_linear = (block_warp_id_y * num_warps_x) + block_warp_id_x;  // 每个 warp 在所属 block 中的线性 id (0 to WARPS_PER_BLOCK - 1)

    // 共享内存中存放的是其对应 block 中 WARPS_PER_BLOCK 个 warp 计算需要的数据
    __shared__ __half smem_input_tile[WARPS_PER_BLOCK * WMMA_M * WMMA_K];
    __shared__ __half smem_weight_tile[WARPS_PER_BLOCK * WMMA_K * WMMA_N];

    // 声明 tensor core 计算需要的数据容器
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> acc_frag;
    wmma::fill_fragment(acc_frag, __float2half(0.0f));

    // 每个 warp 需要的计算数据在共享内存中的位置
    __half *input_tile_start = smem_input_tile + (block_warp_id_linear * WMMA_INPUT_TILE_SIZE);
    __half *weight_tile_start = smem_weight_tile + (block_warp_id_linear * WMMA_FILTER_TILE_SIZE);

    #pragma unroll
    for (int i = 0; i < GEMM_K; i += WMMA_K) {
        int a_row = global_warp_id_x * WMMA_M;  // 每个 warp 对应的输入 tile 在输入矩阵中的行起始索引
        int a_col = i;                          // 每个 warp 对应的输入 tile 在输入矩阵中的列起始索引
        int b_row = i;                          // 每个 warp 对应的权重 tile 在权重矩阵中的行起始索引
        int b_col = global_warp_id_y * WMMA_N;  // 每个 warp 对应的权重 tile 在权重矩阵中的列起始索引

        // 读取输入矩阵
        #pragma unroll
        for (int j = intra_warp_thread_id; j < WMMA_INPUT_TILE_SIZE; j += WARP_SIZE) {  // 每个线程在 16x16 的输入 tile 中的 id
            int rel_slice_row = j / WMMA_K;                                             // 每个线程此时需要搬运的数据在 16x16 的输入 tile 中的行索引
            int abs_slice_row = a_row + rel_slice_row;                                  // 每个线程此时需要搬运的数据在输入矩阵中的全局行索引
            int abs_slice_col = a_col + (j % WMMA_K);                                   // 每个线程此时需要搬运的数据在输入矩阵中的全局列索引

            int n = abs_slice_row / out_n_stride;                                    // 每个线程此时需要搬运的数据在输出 feature map 中的 batch 索引
            int p = (abs_slice_row % out_n_stride) / out_h_stride;                   // 每个线程此时需要搬运的数据在输出 feature map 中的行索引
            int q = ((abs_slice_row % out_n_stride) % out_h_stride) / out_w_stride;  // 每个线程此时需要搬运的数据在输出 feature map 中的列索引

            int offsets[3] = {0, 1, 2};

            int y = p + offsets[(abs_slice_col % UNROLLED_KERNEL_SIZE) / KW];  // 每个线程此时需要搬运的数据在输入 feature map 中的列索引
            int x = q + offsets[(abs_slice_col % UNROLLED_KERNEL_SIZE) % KW];  // 每个线程此时需要搬运的数据在输入 feature map 中的行索引
            int c = abs_slice_col / UNROLLED_KERNEL_SIZE;                      // 每个线程此时需要搬运的数据在输入 feature map 中的通道索引

            if (x >= 0 && x < IW && y >= 0 && y < IH) {  // 防止边缘处越界
                int idx = n * in_n_stride + c * in_c_stride + y * in_h_stride + x * in_w_stride;
                input_tile_start[j] = input[idx];
            } else {
                input_tile_start[j] = __float2half(0.f);
            }
        }

        // 读取权重
        #pragma unroll
        for (int j = intra_warp_thread_id; j < WMMA_FILTER_TILE_SIZE; j += WARP_SIZE) {  // 每个线程在 16x16 的输入 tile 中的 id
            int rel_slice_row = j / WMMA_K;                                              // 每个线程此时需要搬运的数据在 16x16 的输入 tile 中的行索引
            int abs_slice_row = b_row + rel_slice_row;                                   // 每个线程此时需要搬运的数据在权重矩阵中的全局行索引
            int abs_slice_col = b_col + (j % WMMA_K);                                    // 每个线程此时需要搬运的数据在权重矩阵中的全局列索引

            int k = abs_slice_col;                                // 每个线程此时需要搬运的数据在卷积核中的卷积核索引
            int c = abs_slice_row / UNROLLED_KERNEL_SIZE;         // 每个线程此时需要搬运的数据在卷积核中的通道索引
            int r = (abs_slice_row % UNROLLED_KERNEL_SIZE) / KW;  // 每个线程此时需要搬运的数据在卷积核中的列索引
            int s = (abs_slice_row % UNROLLED_KERNEL_SIZE) % KW;  // 每个线程此时需要搬运的数据在卷积核中的行索引

            int idx = k * kernel_n_stride + c * kernel_c_stride + r * kernel_h_stride + s * kernel_w_stride;
            weight_tile_start[j] = kernel[idx];
        }

        // 使用 tensor core 完成矩阵乘法
        if (a_row < GEMM_M && a_col < GEMM_K && b_row < GEMM_K && b_col < GEMM_N) {
            wmma::load_matrix_sync(a_frag, input_tile_start, WMMA_K);
            wmma::load_matrix_sync(b_frag, weight_tile_start, WMMA_N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    int c_col = global_warp_id_y * WMMA_N;  // 每个 warp 对应的输入 tile 在输出矩阵中的行起始索引
    int c_row = global_warp_id_x * WMMA_M;  // 每个 warp 对应的输入 tile 在输出矩阵中的列起始索引

    // 写回输出矩阵
    if (c_row < GEMM_M && c_col < GEMM_N) {
        wmma::store_matrix_sync(output + (c_row + c_col * GEMM_M), acc_frag, GEMM_M, wmma::mem_col_major);
    }
}

int main() {
    const int PADDING_H = 1;
    const int PADDING_W = 1;
    const int STRIDE_H = 1;
    const int STRIDE_W = 1;
    const int KH = 3;
    const int KW = 3;
    const int WARPS_PER_BLOCK = 8;

    const int N = 1;
    const int IC = 8;
    const int IH = 64;
    const int IW = 64;
    const int OC = 16;
    const int OH = (IH - KH + 2 * PADDING_H) / STRIDE_H + 1;
    const int OW = (IW - KW + 2 * PADDING_W) / STRIDE_W + 1;

    int GEMM_M = N * OH * OW;
    int GEMM_N = OC;
    assert(GEMM_M % WMMA_M == 0);
    assert(GEMM_N % WMMA_N == 0);
    assert(N == 1);

    int input_size = N * IC * IH * IW;
    int output_size = OC * OH * OW;
    int kernel_size = OC * IC * KH * KW;
    float *cpu_input, *cpu_output, *cpu_kernel, *cuda_output;
    cpu_input = (float *)malloc(input_size * sizeof(float));
    cpu_output = (float *)malloc(output_size * sizeof(float));
    cpu_kernel = (float *)malloc(kernel_size * sizeof(float));
    cuda_output = (float *)malloc(output_size * sizeof(float));

    for (int i = 0; i < input_size; ++i) {
        cpu_input[i] = 1.0 * i / input_size;
    }
    for (int i = 0; i < kernel_size; ++i) {
        cpu_kernel[i] = 1.0 * i / kernel_size;
    }
    for (int i = 0; i < output_size; ++i) {
        cpu_output[i] = 0.0;
        cuda_output[i] = 0.0;
    }

    /* ---- CPU BEGIN ---- */
    cpu_conv(cpu_input, cpu_output, cpu_kernel, IC, IH, IW, OC, OH, OW, KH, KW);
    /* ---- CPU END ---- */

    /* ---- GPU BEGIN ---- */
    cudaError_t err;

    __half *gpu_input_fp16, *gpu_kernel_fp16, *gpu_output_fp16;
    cudaMalloc(&gpu_input_fp16, input_size * sizeof(__half));
    cudaMalloc(&gpu_output_fp16, output_size * sizeof(__half));
    cudaMalloc(&gpu_kernel_fp16, kernel_size * sizeof(__half));

    // convert data to fp16
    __half *cpu_input_fp16, *cpu_output_fp16, *cpu_kernel_fp16, *cuda_output_fp16;
    cpu_input_fp16 = (__half *)malloc(input_size * sizeof(__half));
    cpu_output_fp16 = (__half *)malloc(output_size * sizeof(__half));
    cpu_kernel_fp16 = (__half *)malloc(kernel_size * sizeof(__half));
    cuda_output_fp16 = (__half *)malloc(output_size * sizeof(__half));

    for (int i = 0; i < input_size; ++i) cpu_input_fp16[i] = __float2half(cpu_input[i]);
    for (int i = 0; i < kernel_size; ++i) cpu_kernel_fp16[i] = __float2half(cpu_kernel[i]);
    for (int i = 0; i < output_size; ++i) cpu_output_fp16[i] = __float2half(0.f);

    cudaMemcpy(gpu_input_fp16, cpu_input_fp16, input_size * sizeof(__half), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cudaMemcpy input failed: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(gpu_kernel_fp16, cpu_kernel_fp16, kernel_size * sizeof(__half), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cudaMemcpy kernel failed: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(gpu_output_fp16, cpu_output_fp16, output_size * sizeof(__half), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cudaMemcpy output failed: %s\n", cudaGetErrorString(err));
    }

    dim3 dim_block(128, 2);
    dim3 dim_grid((GEMM_M + (WMMA_M * dim_block.x / 32 - 1)) / (WMMA_M * dim_block.x / 32), (GEMM_N + WMMA_N * dim_block.y - 1) / (WMMA_N * dim_block.y));
    assert(dim_block.y * dim_block.x / WARP_SIZE == WARPS_PER_BLOCK);

    implicit_gemm_conv<KH, KW, WARPS_PER_BLOCK><<<dim_grid, dim_block>>>(gpu_input_fp16, gpu_output_fp16, gpu_kernel_fp16, N, IC, IH, IW, OC, OH, OW);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("execute kernel function: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(cuda_output_fp16, gpu_output_fp16, output_size * sizeof(__half), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cudaMemcpy output failed: %s\n", cudaGetErrorString(err));
    }

    for (int i = 0; i < output_size; ++i) {
        float cuda_value = __half2float(cuda_output_fp16[i]);
        float cpu_value = cpu_output[i];
        if (!is_close(cuda_value, cpu_value, 1e-2, 1e-3)) {
            printf("WRONG VALUE: %.5f | %.5f at %d\n", cpu_output[i], cuda_value, i);
            break;
        }
    }
    /* ---- GPU END ---- */

    printf("done!\n");
}