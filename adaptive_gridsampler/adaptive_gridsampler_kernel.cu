#include <stdio.h>
#include <torch/extension.h>

#include "helper_cuda.h"

#define BLOCK_SIZE 256

template <typename scalar_t>
__global__ void kernel_adaptive_gridsampler_update_output(
const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> img,
const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> kernels,
const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> offsets_h,
const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> offsets_v,
const int offset_unit,
const int padding,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,
const size_t n)
{
    auto global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(global_idx >= n) return;

    auto dim_b = output.size(0);
    auto dim_c = output.size(1);
    auto dim_h = output.size(2);
    auto dim_w = output.size(3);

    auto idb = (global_idx / (dim_c * dim_h * dim_w)) % dim_b;
    auto idc = (global_idx / (dim_h * dim_w)) % dim_c;
    auto idy = (global_idx / dim_w) % dim_h;
    auto idx = global_idx % dim_w;

    if(idx >= dim_w || idy >= dim_h)
        return;

    int k_size = sqrt(float(kernels.size(1)));
    float w = float(img.size(3) - 2 * padding);
    float h = float(img.size(2) - 2 * padding);

    scalar_t result = 0;
    for(int k_y = 0; k_y < k_size; ++k_y)
    {
        for(int k_x = 0; k_x < k_size; ++k_x)
        {
            scalar_t offset_h = offsets_h[idb][k_size * k_y + k_x][idy][idx] * offset_unit;
            scalar_t offset_v = offsets_v[idb][k_size * k_y + k_x][idy][idx] * offset_unit;

            scalar_t p_x = static_cast<scalar_t>(idx + 0.5) / dim_w * w + k_x + offset_h - 0.5;
            scalar_t p_y = static_cast<scalar_t>(idy + 0.5) / dim_h * h + k_y + offset_v - 0.5;
            scalar_t alpha = p_x - floor(p_x);
            scalar_t beta = p_y - floor(p_y);

            int xL = max(min(int(floor(p_x)), int(w + 2 * padding - 1)), 0);
            int xR = max(min(xL + 1, int(w + 2 * padding - 1)), 0);
            int yT = max(min(int(floor(p_y)), int(h + 2 * padding - 1)), 0);
            int yB = max(min(yT + 1, int(h + 2 * padding - 1)), 0);

            scalar_t val = 0;
            val += (1 - alpha) * (1 - beta) * img[idb][idc][yT][xL];
            val += alpha * (1 - beta) * img[idb][idc][yT][xR];
            val += (1 - alpha) * beta * img[idb][idc][yB][xL];
            val += alpha * beta * img[idb][idc][yB][xR];

            result += val * kernels[idb][k_size * k_y + k_x][idy][idx];
        }
    }
    output[idb][idc][idy][idx] = result;
}

void adaptive_gridsampler_kernel_forward(const torch::Tensor& img, const torch::Tensor& kernels, const torch::Tensor& offsets_h, const torch::Tensor& offsets_v, const int offset_unit, const int padding, torch::Tensor& output)
{
    kernel_adaptive_gridsampler_update_output<float><<<(output.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
    img.packed_accessor32<float, 4, torch::RestrictPtrTraits>(), kernels.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
    offsets_h.packed_accessor32<float, 4, torch::RestrictPtrTraits>(), offsets_v.packed_accessor32<float, 4, torch::RestrictPtrTraits>(), offset_unit, padding,
    output.packed_accessor32<float, 4, torch::RestrictPtrTraits>(), output.numel());

    checkCudaErrors(cudaGetLastError());
}