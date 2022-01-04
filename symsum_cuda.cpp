#include <torch/extension.h>

// CUDA declarations
torch::Tensor symsum_cuda_forward(torch::Tensor input);
torch::Tensor symsum_cuda_backward(torch::Tensor input, torch::Tensor output_grad);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

torch::Tensor symsum_forward(torch::Tensor input) {
    CHECK_CUDA(input);
    return symsum_cuda_forward(input);
}

torch::Tensor symsum_backward(torch::Tensor input, torch::Tensor output_grad) {
    CHECK_CUDA(input);
    CHECK_CUDA(output_grad);
    return symsum_cuda_backward(input, output_grad);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &symsum_forward, "SymSum forward (CUDA)");
  m.def("backward", &symsum_backward, "SymSum backward (CUDA)");
}