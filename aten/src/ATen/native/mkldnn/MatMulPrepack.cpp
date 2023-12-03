#include <vector>

#include <ATen/native/mkldnn/Common.h>
#include <ATen/native/mkldnn/MatMulPrepack.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/OpContext.h>
#include <ATen/native/utils/Factory.h>
#include <ATen/native/utils/ParamUtils.h>
#include <c10/util/irange.h>

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
    std::string attr) {
  auto it = fusion_attr_map.find(attr);
  TORCH_CHECK(it != fusion_attr_map.end(), "Fusion behavior undefined.");
  ideep::attr_t op_attr = it->second;

  return mkldnn::MkldnnMatMulOpContext::create_context(
      std::move(weight),
      std::move(bias),
      std::move(padding),
      std::move(stride),
      std::move(input_size),
      op_attr);
}

ContextMatMul create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef input_size,
    const ideep::attr_t& attr) {
  auto k = weight.ndimension();
  int64_t dim = k - 2;
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  const auto input_size_expanded =
      expand_param_if_needed(input_size, "input_size", k);

  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  auto w = itensor_view_from_dense(weight);
  // TODO: what if input is nhwc but w is nchw
  bool is_channels_last =
      weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast;

ideep::tensor::desc expected_weight_desc = ideep::matmul_forward::expected_weights_desc(w.get_dims());

#if 0
  ideep::tensor::desc expected_weight_desc =
      ideep::matmul_forward::expected_weights_desc(
          w.get_dims(),
          w.get_data_type(),
          {stride_expanded.begin(), stride_expanded.end()},
          {padding_expanded.begin(), padding_expanded.end()},
          {padding_expanded.begin(), padding_expanded.end()},
          0, /*ideep algo, TODO: what it is?*/
          ideep::prop_kind::forward,
          /*x_dtype*/ w.get_data_type(),
          {input_size_expanded.begin(), input_size_expanded.end()},
          attr,
          is_channels_last);
#endif

  ideep::tensor packed_weight;
  packed_weight.init(expected_weight_desc);
  packed_weight.feed_from(w);

  return ContextMatMul{
      std::move(packed_weight),
      bias.has_value() ? c10::make_optional(*bias) : c10::nullopt,
      {padding_expanded.begin(), padding_expanded.end()},
      {stride_expanded.begin(), stride_expanded.end()},
      attr};
}

static void _mkldnn_matmul_out(
    const ideep::tensor& x,
    ideep::tensor& y,
    const ideep::tensor& w,
    const c10::optional<ideep::tensor>& b,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef output_sizes,
    const ideep::attr_t& attr = ideep::attr_t()) {

	std::cout << "_mkldnn_matmul_out prepacked" << std::endl;      
#if 0
	if (b.has_value()) {
    ideep::matmul_forward::compute_v2(
        x,
        w,
        b.value(),
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::zero_point_t(),
        ideep::zero_point_t(),
        attr);
  } else {

    ideep::matmul_forward::compute_v2(
        x,
        w,
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::zero_point_t(),
        ideep::zero_point_t(),
        attr);

  }
#endif
}

static void mkldnn_matmul_out(
    const Tensor& input,
    ideep::tensor& mkldnn_output,
    const ideep::tensor& mkldnn_weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef output_sizes,
    const ideep::attr_t& attr = ideep::attr_t()) {
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  const ideep::tensor mkldnn_input = itensor_from_tensor(input);
  c10::optional<ideep::tensor> mkldnn_bias{c10::nullopt};
  if (bias.defined()) {
    mkldnn_bias = itensor_from_tensor(bias);
  }

  _mkldnn_matmul_out(
      mkldnn_input,
      mkldnn_output,
      mkldnn_weight,
      mkldnn_bias,
      padding,
      stride,
      output_sizes,
      attr);
}

static std::vector<int64_t> get_output_sizes(
    ContextMatMul& context,
    const Tensor& input) {
  const ideep::tensor& mkldnn_weight = context.weight_packed_;
  IntArrayRef padding = context.padding_;
  IntArrayRef stride = context.stride_;

  auto kernel_size = mkldnn_weight.get_dims();

  std::vector<int64_t> input_size = input.sizes().vec();
  //return matmul_output_size(input_size, kernel_size, padding, stride);

  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
#if 0
  for (const auto d : c10::irange(2, dim)) {
    auto kernel = (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
#endif
  return output_size;


}

Tensor run(ContextMatMul& context, const Tensor& input) {
#if 0
  std::vector<int64_t> output_sizes = get_output_sizes(context, input);
  auto output = at::empty(
      output_sizes,
      input.options().memory_format(input.suggest_memory_format()));

  bool is_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  ideep::tensor y;

  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  ideep::tensor mkldnn_output = itensor_from_tensor(output);

  if (is_channels_last) {
    mkldnn_matmul_out(
        input,
        mkldnn_output,
        context.weight_packed_,
        context.at_bias_,
        context.padding_,
        context.stride_,
        output_sizes,
        context.attr_);
  } else {
	  ideep::tensor y;
    mkldnn_matmul_out(
        input,
        y,
        context.weight_packed_,
        context.at_bias_,
        context.padding_,
        context.stride_,
        output_sizes,
        context.attr_);
    mkldnn_output.feed_from(y);
 // }
  return output;
#endif
  return input;
}

void run(ContextMatMul& context, const Tensor& input, void* output) {
  std::vector<int64_t> output_sizes = get_output_sizes(context, input);

  bool is_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  ideep::tensor y;

  ideep::tag o_tag = is_channels_last ? ideep::tag::nhwc : ideep::tag::nchw;
  ideep::tensor::desc o_desc = {
      output_sizes, get_mkldnn_dtype(input.scalar_type()), o_tag};
  ideep::tensor mkldnn_output = {o_desc, output};

  if (is_channels_last) {
    mkldnn_matmul_out(
        input,
        mkldnn_output,
        context.weight_packed_,
        context.at_bias_,
        context.padding_,
        context.stride_,
        output_sizes,
        context.attr_);
  } else {
    mkldnn_matmul_out(
        input,
        y,
        context.weight_packed_,
        context.at_bias_,
        context.padding_,
        context.stride_,
        output_sizes,
        context.attr_);
    mkldnn_output.feed_from(y);
  }
}

Tensor matmul_run(
    const Tensor& input,
    const c10::intrusive_ptr<mkldnn::MatMulOpContext>& op_context) {
  return op_context->run(input);
}

} // namespace matmul
} // namespace internal
} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
