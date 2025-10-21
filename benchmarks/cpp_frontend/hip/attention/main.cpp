#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <memory>
#include <string>
#include <string_view>
#include <hip/hip_runtime.h>
#include <proteus/CppJitModule.hpp>
#include "inja/inja.h"
#include "../../gpu/gpu_common.h"

using namespace proteus;

#define TARGET "hip"
#define INCLUDE "#include <hip/hip_runtime.h>"

float* attention_host(const float* key, const float* value, const float* query,
  const int n, const int d)
{
// intermediate
float* dot_product = (float*) malloc (n * sizeof(float));
float* score = (float*) malloc (n * sizeof(float));
// result
float* output = (float*) malloc (d * sizeof(float));

for (int i = 0; i < n; i++) {
float sum = 0;
for (int j = 0; j < d; j++)
sum += key[i * d + j] * query[j];
dot_product[i] = sum;
}

float sum = 0;
for (int i = 0; i < n; i++)
sum += expf(dot_product[i]);

for (int i = 0; i < n; i++)
score[i] = expf(dot_product[i]) / sum;

for (int j = 0; j < d; j++) {
float sum = 0;
for (int i = 0; i < n; i++)
sum += score[i] * value[i * d + j];
output[j] = sum;
}

free(dot_product);
free(score);
return output;
}

// Kernel template for attention_kernel1
constexpr std::string_view StrAttentionKernel1Template = INCLUDE R"cpp(
extern "C" __global__ void attention_kernel1(
    const float* __restrict__ key,
    const float* __restrict__ query,
    float* __restrict__ dot_product,
    float* __restrict__ exp_sum)
{
  constexpr int n = {{ n }};
  constexpr int d = {{ d }};
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float sum = 0;
    for (int j = 0; j < d; j++)
      sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
    atomicAdd(exp_sum, __expf(sum));
  }
}
)cpp";

// Kernel template for attention_kernel2
constexpr std::string_view StrAttentionKernel2Template = INCLUDE R"cpp(
extern "C" __global__ void attention_kernel2(
    const float* __restrict__ exp_sum,
    const float* __restrict__ dot_product,
    float* __restrict__ score)
{
  constexpr int n = {{ n }};
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    score[i] = __expf(dot_product[i]) / exp_sum[0];
}
)cpp";

// Kernel template for attention_kernel3
constexpr std::string_view StrAttentionKernel3Template = INCLUDE R"cpp(
extern "C" __global__ void attention_kernel3(
    const float* __restrict__ score,
    const float* __restrict__ value,
    float* __restrict__ output)
{
  constexpr int n = {{ n }};
  constexpr int d = {{ d }};
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < d) {
    float sum = 0;
    for (int i = 0; i < n; i++)
      sum += score[i] * value[i * d + j];
    output[j] = sum;
  }
}
)cpp";

// Getter function for attention_kernel1
static auto getAttentionKernel1(int n, int d)
{
  inja::json data = {{"n", n}, {"d", d}};
  auto kernelSource = inja::render(std::string{StrAttentionKernel1Template}, data);
  auto JitMod = std::make_unique<CppJitModule>(TARGET, kernelSource);
  auto Kernel = JitMod->getKernel<void(const float *, const float *, float *, float *)>("attention_kernel1");
  return std::make_pair(std::move(JitMod), Kernel);
}

// Getter function for attention_kernel2
static auto getAttentionKernel2(int n)
{
  inja::json data = {{"n", n}};
  auto kernelSource = inja::render(std::string{StrAttentionKernel2Template}, data);
  auto JitMod = std::make_unique<CppJitModule>(TARGET, kernelSource);
  auto Kernel = JitMod->getKernel<void(const float *, const float *, float *)>("attention_kernel2");
  return std::make_pair(std::move(JitMod), Kernel);
}

// Getter function for attention_kernel3
static auto getAttentionKernel3(int n, int d)
{
  inja::json data = {{"n", n}, {"d", d}};
  auto kernelSource = inja::render(std::string{StrAttentionKernel3Template}, data);
  auto JitMod = std::make_unique<CppJitModule>(TARGET, kernelSource);
  auto Kernel = JitMod->getKernel<void(const float *, const float *, float *)>("attention_kernel3");
  return std::make_pair(std::move(JitMod), Kernel);
}

float* attention_device(const float* key, const float* value, const float* query,
                        const int n, const int d, const int repeat, const int verify)
{
  // input
  float *d_key;
  gpuErrCheck(hipMalloc((void**)&d_key, n * d * sizeof(float)));
  gpuErrCheck(hipMemcpy(d_key, key, n * d * sizeof(float), hipMemcpyHostToDevice));

  float *d_value;
  gpuErrCheck(hipMalloc((void**)&d_value, n * d * sizeof(float)));
  gpuErrCheck(hipMemcpy(d_value, value, n * d * sizeof(float), hipMemcpyHostToDevice));

  float *d_query;
  gpuErrCheck(hipMalloc((void**)&d_query, d * sizeof(float)));
  gpuErrCheck(hipMemcpy(d_query, query, d * sizeof(float), hipMemcpyHostToDevice));

  // intermediate
  float *d_dot_product;
  gpuErrCheck(hipMalloc((void**)&d_dot_product, n * sizeof(float)));

  float *d_exp_sum;
  gpuErrCheck(hipMalloc((void**)&d_exp_sum, sizeof(float)));

  // result
  float *output = (float*) malloc (d * sizeof(float));
  float *d_output;
  gpuErrCheck(hipMalloc((void**)&d_output, d * sizeof(float)));

  float *d_score;
  gpuErrCheck(hipMalloc((void**)&d_score, n * sizeof(float)));

  gpuErrCheck(hipDeviceSynchronize());

  // Get kernels with specialized n and d values
  auto [JitMod1, Kernel1] = getAttentionKernel1(n, d);
  auto [JitMod2, Kernel2] = getAttentionKernel2(n);
  auto [JitMod3, Kernel3] = getAttentionKernel3(n, d);

  auto start = std::chrono::steady_clock::now();

  for (int k = 0; k < repeat; k++) {
    if(verify) {
      gpuErrCheck(hipMemset(d_exp_sum, 0, 4));
    }
    Kernel1.launch({static_cast<unsigned int>((n+255)/256), 1, 1},
                   {256, 1, 1},
                   0, nullptr, d_key, d_query, d_dot_product, d_exp_sum);
    Kernel2.launch({static_cast<unsigned int>((n+255)/256), 1, 1},
                   {256, 1, 1},
                   0, nullptr, d_exp_sum, d_dot_product, d_score);
    Kernel3.launch({static_cast<unsigned int>((d+255)/256), 1, 1},
                   {256, 1, 1},
                   0, nullptr, d_score, d_value, d_output);
  }

  gpuErrCheck(hipDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);

  gpuErrCheck(hipMemcpy(output, d_output, d * sizeof(float), hipMemcpyDeviceToHost));
  gpuErrCheck(hipFree(d_score));
  gpuErrCheck(hipFree(d_value));
  gpuErrCheck(hipFree(d_output));
  gpuErrCheck(hipFree(d_key));
  gpuErrCheck(hipFree(d_dot_product));
  gpuErrCheck(hipFree(d_exp_sum));
  gpuErrCheck(hipFree(d_query));
  return output;
}

int main(int argc, char* argv[]) {
  if (argc != 4 && argc != 5) {
    printf("Usage: %s <rows> <columns> <repeat> [verify]\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int d = atoi(argv[2]);
  const int r = atoi(argv[3]);
  const int verify = (argc == 5) ? atoi(argv[4]) : 0;

  // input
  float* key = (float*) malloc (n * d * sizeof(float));
  float* value = (float*) malloc (n * d * sizeof(float));
  float* query = (float*) malloc (d * sizeof(float));

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  if (verify) {
  for (int i = 0; i < n * d; i++) {
    key[i] = dist(gen);
    value[i] = dist(gen);
      query[i % d] = dist(gen);
    }
  }

  float* dout = attention_device(key, value, query, n, d, r, verify);

  if (verify) {
    float* hout = attention_host(key, value, query, n, d);

    float rmse = 0;
    for (int i = 0; i < d; i++) {
      rmse += (hout[i] - dout[i]) * (hout[i] - dout[i]);
    }
    printf("RMSE = %f\n", sqrtf(rmse / d));

    free(hout);
  }

  free(key);
  free(value);
  free(query);
  free(dout);
  return 0;
}
