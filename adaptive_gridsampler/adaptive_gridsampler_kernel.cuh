#ifndef ADAPTIVE_GRIDSAMPLER_KERNEL_CUH
#define ADAPTIVE_GRIDSAMPLER_KERNEL_CUH

#include <torch/extension.h>

void adaptive_gridsampler_kernel_forward(const torch::Tensor& img, const torch::Tensor& kernels, const torch::Tensor& offsets_h, const torch::Tensor& offsets_v, const int offset_unit, const int padding, torch::Tensor& output);
void adaptive_gridsampler_kernel_backward(const torch::Tensor& img, const torch::Tensor& kernels, const torch::Tensor& offsets_h, const torch::Tensor& offsets_v, const int offset_unit, const torch::Tensor& gradOutput, const int padding, torch::Tensor& gradInput_kernels, torch::Tensor& gradInput_offsets_h, torch::Tensor& gradInput_offsets_v);

#endif