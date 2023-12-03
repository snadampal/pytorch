#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/mkldnn/Common.h>
#include <ATen/native/mkldnn/OpContext.h>

#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace mkldnn {
namespace internal {
namespace matmul {

c10::intrusive_ptr<mkldnn::MatMulOpContext> createMatMulPrePackOpContext(
    Tensor weight,
    c10::optional<Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> input_size,
    std::string attr);

Tensor matmul_run(
    const Tensor& input,
    const c10::intrusive_ptr<mkldnn::MatMulOpContext>& op_context);

ContextMatMul create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef input_size,
    const ideep::attr_t& attr);

Tensor run(ContextMatMul& context, const Tensor& input);

void run(ContextMatMul& context, const Tensor& input, void* output);

} // namespace matmul
} // namespace internal
} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
