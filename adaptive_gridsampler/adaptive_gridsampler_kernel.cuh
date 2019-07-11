#ifndef ADAPTIVE_GRIDSAMPLER_KERNEL_CUH
#define ADAPTIVE_GRIDSAMPLER_KERNEL_CUH

#include <ATen/ATen.h>

void adaptive_gridsampler_kernel_forward(const at::Tensor& img, const at::Tensor& kernels, const at::Tensor& offsets_h, const at::Tensor& offsets_v, const int offset_unit, const int padding, at::Tensor& output);

#endif