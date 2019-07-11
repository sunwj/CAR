#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <stdio.h>
#include "helper_cuda.h"

#define DIM0(TENSOR) ((TENSOR).x)
#define DIM1(TENSOR) ((TENSOR).y)
#define DIM2(TENSOR) ((TENSOR).z)
#define DIM3(TENSOR) ((TENSOR).w)

#define ELEMENT4(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w))])
#define DIM4_INDEX(TENSOR, xx, yy, zz, ww) (((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w)))

#define BLOCK_SIZE 256

template <typename scalar_t>
__global__ void kernel_adaptive_gridsampler_update_output(const scalar_t* __restrict__ img, const long4 img_size, const long4 img_stride,
const scalar_t* __restrict__ kernels, const long4 kernels_size, const long4 kernels_stride,
const scalar_t* __restrict__ offsets_h, const long4 offsets_h_size, const long4 offsets_h_stride,
const scalar_t* __restrict__ offsets_v, const long4 offsets_v_size, const long4 offsets_v_stride,
const int offset_unit,
const int padding,
scalar_t* __restrict__ output, const long4 output_size, const long4 output_stride,
const size_t n)
{
    auto global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(global_idx >= n) return;

    auto dim_b = DIM0(output_size);
    auto dim_c = DIM1(output_size);
    auto dim_h = DIM2(output_size);
    auto dim_w = DIM3(output_size);

    auto idb = (global_idx / (dim_c * dim_h * dim_w)) % dim_b;
    auto idc = (global_idx / (dim_h * dim_w)) % dim_c;
    auto idy = (global_idx / dim_w) % dim_h;
    auto idx = global_idx % dim_w;

    if(idx >= dim_w || idy >= dim_h)
        return;

    int k_size = sqrt(float(DIM1(kernels_size)));
    float w = float(DIM3(img_size) - 2 * padding);
    float h = float(DIM2(img_size) - 2 * padding);

    scalar_t result = 0;
    for(int k_y = 0; k_y < k_size; ++k_y)
    {
        for(int k_x = 0; k_x < k_size; ++k_x)
        {
            scalar_t offset_h = ELEMENT4(offsets_h, idb, k_size * k_y + k_x, idy, idx) * offset_unit;
            scalar_t offset_v = ELEMENT4(offsets_v, idb, k_size * k_y + k_x, idy, idx) * offset_unit;

            scalar_t p_x = static_cast<scalar_t>(idx + 0.5) / dim_w * w + k_x + offset_h - 0.5;
            scalar_t p_y = static_cast<scalar_t>(idy + 0.5) / dim_h * h + k_y + offset_v - 0.5;
            scalar_t alpha = p_x - floor(p_x);
            scalar_t beta = p_y - floor(p_y);

            int xL = max(min(int(floor(p_x)), int(w + 2 * padding - 1)), 0);
            int xR = max(min(xL + 1, int(w + 2 * padding - 1)), 0);
            int yT = max(min(int(floor(p_y)), int(h + 2 * padding - 1)), 0);
            int yB = max(min(yT + 1, int(h + 2 * padding - 1)), 0);

            scalar_t val = 0;
            val += (1 - alpha) * (1 - beta) * ELEMENT4(img, idb, idc, yT, xL);
            val += alpha * (1 - beta) * ELEMENT4(img, idb, idc, yT, xR);
            val += (1 - alpha) * beta * ELEMENT4(img, idb, idc, yB, xL);
            val += alpha * beta * ELEMENT4(img, idb, idc, yB, xR);

            result += val * ELEMENT4(kernels, idb, k_size * k_y + k_x, idy, idx);
        }
    }
    output[DIM4_INDEX(output, idb, idc, idy, idx)] = result;
}

void adaptive_gridsampler_kernel_forward(const at::Tensor& img, const at::Tensor& kernels, const at::Tensor& offsets_h, const at::Tensor& offsets_v, const int offset_unit, const int padding, at::Tensor& output)
{
    const long4 img_size = make_long4(img.size(0), img.size(1), img.size(2), img.size(3));
    const long4 img_stride = make_long4(img.stride(0), img.stride(1), img.stride(2), img.stride(3));

    const long4 kernels_size = make_long4(kernels.size(0), kernels.size(1), kernels.size(2), kernels.size(3));
    const long4 kernels_stride = make_long4(kernels.stride(0), kernels.stride(1), kernels.stride(2), kernels.stride(3));

    const long4 offsets_h_size = make_long4(offsets_h.size(0), offsets_h.size(1), offsets_h.size(2), offsets_h.size(3));
    const long4 offsets_h_stride = make_long4(offsets_h.stride(0), offsets_h.stride(1), offsets_h.stride(2), offsets_h.stride(3));

    const long4 offsets_v_size = make_long4(offsets_v.size(0), offsets_v.size(1), offsets_v.size(2), offsets_v.size(3));
    const long4 offsets_v_stride = make_long4(offsets_v.stride(0), offsets_v.stride(1), offsets_v.stride(2), offsets_v.stride(3));

    const long4 output_size = make_long4(output.size(0), output.size(1), output.size(2), output.size(3));
    const long4 output_stride = make_long4(output.stride(0), output.stride(1), output.stride(2), output.stride(3));

    kernel_adaptive_gridsampler_update_output<float><<<(output.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
    img.data<float>(), img_size, img_stride, kernels.data<float>(), kernels_size, kernels_stride, offsets_h.data<float>(), offsets_h_size, offsets_h_stride,
    offsets_v.data<float>(), offsets_v_size, offsets_v_stride, offset_unit, padding, output.data<float>(), output_size, output_stride, output.numel());

    checkCudaErrors(cudaGetLastError());
}