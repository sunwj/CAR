#include <ATen/ATen.h>
#include <torch/extension.h>

#include "adaptive_gridsampler_kernel.cuh"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a cuda tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

int adaptive_gridsampler_cuda_forward(at::Tensor& img, at::Tensor& kernels, at::Tensor& offsets_h, at::Tensor& offsets_v, int offset_unit, int padding, at::Tensor& output)
{
    CHECK_INPUT(img);
    CHECK_INPUT(kernels);
    CHECK_INPUT(offsets_h);
    CHECK_INPUT(offsets_v);
    CHECK_INPUT(output);

    adaptive_gridsampler_kernel_forward(img, kernels, offsets_h, offsets_v, offset_unit, padding, output);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &adaptive_gridsampler_cuda_forward, "adaptive gridsampler forward (CUDA)");
}