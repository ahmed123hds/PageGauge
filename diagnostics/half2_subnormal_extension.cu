#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

__global__ void half2_chain_kernel(
    const __half2* input, __half2* output, int64_t pairs, int iterations) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pairs) {
    return;
  }
  // Inline PTX prevents nvcc from folding a long multiply-by-one chain away.
  uint32_t value = reinterpret_cast<const uint32_t*>(input)[index];
  constexpr uint32_t one = 0x3c003c00U;  // packed {half(1), half(1)}
  for (int iteration = 0; iteration < iterations; ++iteration) {
    asm volatile("mul.rn.f16x2 %0, %0, %1;" : "+r"(value) : "r"(one));
  }
  reinterpret_cast<uint32_t*>(output)[index] = value;
}

}  // namespace

void half2_chain(torch::Tensor input, torch::Tensor output, int64_t iterations) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input and output must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16, "input must be FP16");
  TORCH_CHECK(output.scalar_type() == torch::kFloat16, "output must be FP16");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "tensors must be contiguous");
  TORCH_CHECK(input.numel() == output.numel(), "tensor sizes must match");
  TORCH_CHECK(input.numel() % 2 == 0, "half2 requires an even element count");
  TORCH_CHECK(iterations > 0, "iterations must be positive");
  const int64_t pairs = input.numel() / 2;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((pairs + threads - 1) / threads);
  half2_chain_kernel<<<blocks, threads>>>(
      reinterpret_cast<const __half2*>(input.data_ptr<at::Half>()),
      reinterpret_cast<__half2*>(output.data_ptr<at::Half>()), pairs,
      static_cast<int>(iterations));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "half2_chain kernel launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("half2_chain", &half2_chain, "Repeated FP16x2 multiply-by-one chain");
}
