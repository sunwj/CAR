#ifndef ADAPTIVE_GRIDSAMPLER_KERNEL_CUH
#define ADAPTIVE_GRIDSAMPLER_KERNEL_CUH

#include <torch/extension.h>

void adaptive_gridsampler_kernel_forward(const torch::Tensor& img, const torch::Tensor& kernels, const torch::Tensor& offsets_h, const torch::Tensor& offsets_v, const int offset_unit, const int padding, torch::Tensor& output);

#endif