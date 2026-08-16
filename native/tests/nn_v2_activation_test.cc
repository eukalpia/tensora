#include "tensora.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\nlast_error=%s\n", message,
                 ts_last_error_message());
    std::exit(1);
  }
}

void RequireStatus(ts_status_t actual,
                   ts_status_t expected,
                   const char* message) {
  if (actual != expected) {
    std::fprintf(stderr,
                 "FAIL: %s expected=%d actual=%d last_error=%s\n",
                 message, static_cast<int>(expected), static_cast<int>(actual),
                 ts_last_error_message());
    std::exit(1);
  }
}

ts_tensor_t Tensor(const std::vector<float>& values,
                   const std::vector<int64_t>& dimensions) {
  ts_tensor_t tensor = 0;
  RequireStatus(
      ts_tensor_from_f32(values.data(), values.size(), dimensions.data(),
                         dimensions.size(), &tensor),
      TS_OK, "create tensor");
  return tensor;
}

std::vector<float> Read(ts_tensor_t tensor, size_t count) {
  std::vector<float> values(count, 0.0f);
  size_t written = 0;
  RequireStatus(ts_tensor_copy_to_host_f32(tensor, values.data(), values.size(),
                                           &written),
                TS_OK, "copy tensor");
  Require(written == count, "host copy writes expected element count");
  return values;
}

void RequireClose(float actual, float expected, float tolerance,
                  const char* message) {
  if (std::fabs(actual - expected) > tolerance) {
    std::fprintf(stderr, "FAIL: %s expected=%g actual=%g\n", message,
                 static_cast<double>(expected), static_cast<double>(actual));
    std::exit(1);
  }
}

void Release(ts_tensor_t tensor) {
  if (tensor != 0) {
    RequireStatus(ts_tensor_release(tensor), TS_OK, "release tensor");
  }
}

}  // namespace

int main() {
  ts_tensor_t input = Tensor({-1.0f, 0.0f, 1.0f}, {3});
  ts_tensor_t gelu = 0;
  ts_tensor_t silu = 0;
  RequireStatus(ts_tensor_gelu(input, &gelu), TS_OK, "GELU forward");
  RequireStatus(ts_tensor_silu(input, &silu), TS_OK, "SiLU forward");
  const auto gelu_values = Read(gelu, 3);
  const auto silu_values = Read(silu, 3);
  RequireClose(gelu_values[0], -0.15865526f, 2e-5f, "GELU(-1)");
  RequireClose(gelu_values[1], 0.0f, 2e-5f, "GELU(0)");
  RequireClose(gelu_values[2], 0.8413447f, 2e-5f, "GELU(1)");
  RequireClose(silu_values[0], -0.26894143f, 2e-5f, "SiLU(-1)");
  RequireClose(silu_values[1], 0.0f, 2e-5f, "SiLU(0)");
  RequireClose(silu_values[2], 0.7310586f, 2e-5f, "SiLU(1)");

  ts_tensor_t gelu_leaf = 0;
  ts_tensor_t silu_leaf = 0;
  RequireStatus(ts_tensor_with_requires_grad(input, 1, &gelu_leaf), TS_OK,
                "GELU leaf");
  RequireStatus(ts_tensor_with_requires_grad(input, 1, &silu_leaf), TS_OK,
                "SiLU leaf");
  ts_tensor_t gelu_value = 0;
  ts_tensor_t silu_value = 0;
  ts_tensor_t gelu_loss = 0;
  ts_tensor_t silu_loss = 0;
  RequireStatus(ts_tensor_gelu(gelu_leaf, &gelu_value), TS_OK,
                "differentiable GELU");
  RequireStatus(ts_tensor_silu(silu_leaf, &silu_value), TS_OK,
                "differentiable SiLU");
  RequireStatus(ts_tensor_sum(gelu_value, &gelu_loss), TS_OK, "GELU sum");
  RequireStatus(ts_tensor_sum(silu_value, &silu_loss), TS_OK, "SiLU sum");
  RequireStatus(ts_tensor_backward(gelu_loss), TS_OK, "GELU backward");
  RequireStatus(ts_tensor_backward(silu_loss), TS_OK, "SiLU backward");
  ts_tensor_t gelu_grad = 0;
  ts_tensor_t silu_grad = 0;
  RequireStatus(ts_tensor_grad(gelu_leaf, &gelu_grad), TS_OK, "GELU gradient");
  RequireStatus(ts_tensor_grad(silu_leaf, &silu_grad), TS_OK, "SiLU gradient");
  const auto gelu_grad_values = Read(gelu_grad, 3);
  const auto silu_grad_values = Read(silu_grad, 3);
  RequireClose(gelu_grad_values[0], -0.08331547f, 3e-5f,
               "GELU derivative -1");
  RequireClose(gelu_grad_values[1], 0.5f, 3e-5f, "GELU derivative 0");
  RequireClose(gelu_grad_values[2], 1.0833155f, 3e-5f,
               "GELU derivative 1");
  RequireClose(silu_grad_values[0], 0.07232949f, 3e-5f,
               "SiLU derivative -1");
  RequireClose(silu_grad_values[1], 0.5f, 3e-5f, "SiLU derivative 0");
  RequireClose(silu_grad_values[2], 0.9276705f, 3e-5f,
               "SiLU derivative 1");

  ts_tensor_t swiglu_input = Tensor({1.0f, -1.0f, 2.0f, 3.0f}, {1, 4});
  ts_tensor_t swiglu_leaf = 0;
  RequireStatus(ts_tensor_with_requires_grad(swiglu_input, 1, &swiglu_leaf),
                TS_OK, "SwiGLU leaf");
  ts_tensor_t swiglu = 0;
  RequireStatus(ts_tensor_swiglu(swiglu_leaf, &swiglu), TS_OK,
                "SwiGLU forward");
  size_t rank = 0;
  int64_t dimensions[2] = {0, 0};
  RequireStatus(ts_tensor_shape(swiglu, dimensions, 2, &rank), TS_OK,
                "SwiGLU output shape");
  Require(rank == 2 && dimensions[0] == 1 && dimensions[1] == 2,
          "SwiGLU halves final dimension");
  const auto swiglu_values = Read(swiglu, 2);
  RequireClose(swiglu_values[0], 1.4621172f, 3e-5f, "SwiGLU first value");
  RequireClose(swiglu_values[1], -0.80682427f, 3e-5f,
               "SwiGLU second value");
  ts_tensor_t swiglu_loss = 0;
  RequireStatus(ts_tensor_sum(swiglu, &swiglu_loss), TS_OK, "SwiGLU sum");
  RequireStatus(ts_tensor_backward(swiglu_loss), TS_OK, "SwiGLU backward");
  ts_tensor_t swiglu_grad = 0;
  RequireStatus(ts_tensor_grad(swiglu_leaf, &swiglu_grad), TS_OK,
                "SwiGLU gradient");
  const auto swiglu_grad_values = Read(swiglu_grad, 4);
  RequireClose(swiglu_grad_values[0], 1.855341f, 4e-5f,
               "SwiGLU first-half derivative");
  RequireClose(swiglu_grad_values[1], 0.21698847f, 4e-5f,
               "SwiGLU second first-half derivative");
  RequireClose(swiglu_grad_values[2], 0.7310586f, 4e-5f,
               "SwiGLU gate derivative");
  RequireClose(swiglu_grad_values[3], -0.26894143f, 4e-5f,
               "SwiGLU second gate derivative");

  ts_tensor_t scalar = Tensor({1.0f}, {});
  ts_tensor_t odd = Tensor({1.0f, 2.0f, 3.0f}, {1, 3});
  ts_tensor_t invalid_output = 123;
  RequireStatus(ts_tensor_swiglu(scalar, &invalid_output), TS_INVALID_SHAPE,
                "rank-zero SwiGLU rejected");
  Require(invalid_output == 0, "failed rank-zero SwiGLU clears output");
  invalid_output = 123;
  RequireStatus(ts_tensor_swiglu(odd, &invalid_output), TS_INVALID_SHAPE,
                "odd-width SwiGLU rejected");
  Require(invalid_output == 0, "failed odd-width SwiGLU clears output");

  Release(odd);
  Release(scalar);
  Release(swiglu_grad);
  Release(swiglu_loss);
  Release(swiglu);
  Release(swiglu_leaf);
  Release(swiglu_input);
  Release(silu_grad);
  Release(gelu_grad);
  Release(silu_loss);
  Release(gelu_loss);
  Release(silu_value);
  Release(gelu_value);
  Release(silu_leaf);
  Release(gelu_leaf);
  Release(silu);
  Release(gelu);
  Release(input);

  std::puts("NN V2 activation/autograd contract passed");
  return 0;
}
