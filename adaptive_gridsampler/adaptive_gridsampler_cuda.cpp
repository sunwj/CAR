#include <ATen/ATen.h>
#include <torch/extension.h>

#include "adaptive_gridsampler_kernel.cuh"

int adaptive_gridsampler_cuda_forward(at::Tensor& img, at::Tensor& kernels, at::Tensor& offsets_h, at::Tensor& offsets_v, int offset_unit, int padding, at::Tensor& output)
{
    adaptive_gridsampler_kernel_forward(img, kernels, offsets_h, offsets_v, offset_unit, padding, output);
    return 1;
}

int adaptive_gridsampler_cuda_backward(at::Tensor& img, at::Tensor& kernels, at::Tensor& offsets_h, at::Tensor& offsets_v, int offset_unit, at::Tensor& gradOutput, int padding,
at::Tensor& gradInput_kernels, at::Tensor& gradInput_offsets_h, at::Tensor& gradInput_offsets_v)
{
    adaptive_gridsampler_kernel_backward(img, kernels, offsets_h, offsets_v, offset_unit, gradOutput, padding, gradInput_kernels, gradInput_offsets_h, gradInput_offsets_v);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &adaptive_gridsampler_cuda_forward, "adaptive gridsampler forward (CUDA)");
    m.def("backward", &adaptive_gridsampler_cuda_backward, "adaptive gridsampler backward (CUDA)");
}