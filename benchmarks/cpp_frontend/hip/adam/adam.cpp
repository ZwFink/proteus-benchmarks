#include <proteus/CppJitModule.hpp>

#include "../../gpu/gpu_common.h"

#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string_view>

#include "inja/inja.h"

using namespace proteus;

#define TARGET "hip"
#define INCLUDE "#include <hip/hip_runtime.h>"

typedef enum {
  ADAM_MODE_0 = 0, // eps under square root
  ADAM_MODE_1 = 1  // eps outside square root
} adamMode_t;

constexpr std::string_view StrAdamKernelTemplate = R"cpp(
{{ include }}

typedef enum {
    ADAM_MODE_0 = 0, // eps under square root
    ADAM_MODE_1 = 1  // eps outside square root
} adamMode_t;

__launch_bounds__({{ threadsPerBlock }})
extern "C" __global__ void adam(float *__restrict__ p, float *__restrict__ m,
                               float *__restrict__ v,
                               const float *__restrict__ g,
                               adamMode_t mode) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t totThreads = gridDim.x * blockDim.x;
  constexpr float b1 = {{ b1 }};
  constexpr float b2 = {{ b2 }};
  constexpr float eps = {{ eps }};
  constexpr float grad_scale = {{ grad_scale }};
  constexpr float step_size = {{ step_size }};
  constexpr int time_step = {{ time_step }};
  constexpr size_t vector_size = {{ vector_size }};
  constexpr float decay = {{ decay }};

  for (size_t j = i; j < vector_size; j += totThreads) {
    for (int t = {{ time_loop_start }}; t < time_step; t++) {
      float scaled_grad = g[j] / grad_scale;
      m[j] = b1 * m[j] + (1.f - b1) * scaled_grad;
      v[j] = b2 * v[j] + (1.f - b2) * scaled_grad * scaled_grad;
      float m_corrected = m[j] / (1.f - powf(b1, t));
      float v_corrected = v[j] / (1.f - powf(b2, t));
      float denom;
      if (mode == ADAM_MODE_0)
        denom = sqrtf(v_corrected + eps);
      else // Mode 1
        denom = sqrtf(v_corrected) + eps;
      float update = (m_corrected / denom) + (decay * p[j]);
      p[j] -= (step_size * update);
    }
  }
})cpp";

int main(int argc, char *argv[]) {
  if (argc < 4 || argc > 5) {
    printf("Usage: %s <vector size> <number of time steps> <repeat> [print_output(0|1)]\n",
           argv[0]);
    return 1;
  }

  const int vector_size = atoi(argv[1]);
  const int time_step = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  const bool print_output = argc >= 5 ? atoi(argv[4]) != 0 : false;

  size_t size_bytes = vector_size * sizeof(float);

  float *m = (float *)malloc(size_bytes);
  float *v = (float *)malloc(size_bytes);
  float *g = (float *)malloc(size_bytes);
  float *p = (float *)malloc(size_bytes);
  float *r = (float *)malloc(size_bytes);

  srand(123);
  for (int i = 0; i < vector_size; i++) {
    m[i] = rand() / (float)RAND_MAX;
    v[i] = rand() / (float)RAND_MAX;
    g[i] = rand() / (float)RAND_MAX;
    r[i] = p[i] = rand() / (float)RAND_MAX;
    if (print_output && i < 10)
      std::cout << "init p[" << i << "] = " << p[i] << "\n";
  }

  float *d_m, *d_v, *d_g, *d_p;

  gpuErrCheck(gpuMalloc((void **)&d_m, size_bytes));
  gpuErrCheck(gpuMemcpy(d_m, m, size_bytes, gpuMemcpyHostToDevice));

  gpuErrCheck(gpuMalloc((void **)&d_v, size_bytes));
  gpuErrCheck(gpuMemcpy(d_v, v, size_bytes, gpuMemcpyHostToDevice));

  gpuErrCheck(gpuMalloc((void **)&d_g, size_bytes));
  gpuErrCheck(gpuMemcpy(d_g, g, size_bytes, gpuMemcpyHostToDevice));

  gpuErrCheck(gpuMalloc((void **)&d_p, size_bytes));
  gpuErrCheck(gpuMemcpy(d_p, p, size_bytes, gpuMemcpyHostToDevice));

  // Arbitrary constants
  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const float grad_scale = 256.f;

  const int threadsPerBlock = 256;
  const dim3 grids((vector_size + threadsPerBlock - 1) / threadsPerBlock);
  const dim3 blocks(threadsPerBlock);

  adamMode_t mode = ADAM_MODE_0;

  gpuErrCheck(gpuDeviceSynchronize());

  inja::json data = {
      {"include", std::string(INCLUDE)},
      {"time_loop_start", 1},
      {"b1", beta1},
      {"b2", beta2},
      {"eps", eps},
      {"grad_scale", grad_scale},
      {"step_size", step_size},
      {"time_step", time_step},
      {"vector_size", vector_size},
      {"decay", decay},
      {"threadsPerBlock", threadsPerBlock}
  };
  data["b1"] = beta1;
  data["b2"] = beta2;
  data["eps"] = eps;
  data["grad_scale"] = grad_scale;
  data["step_size"] = step_size;
  data["time_step"] = time_step;
  data["vector_size"] = vector_size;
  data["decay"] = decay;
  data["threadsPerBlock"] = threadsPerBlock;
  auto kernelSource = inja::render(std::string{StrAdamKernelTemplate}, data);

  CppJitModule CJM{TARGET, kernelSource};

  CJM.compile();
  using AdamSig = void(float *, float *, float *, const float *, adamMode_t);
  auto Kernel = CJM.getKernel<AdamSig>("adam");

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    Kernel.launch({grids.x, grids.y, grids.z}, {blocks.x, blocks.y, blocks.z},
                  0, nullptr, d_p, d_m, d_v, d_g, mode);
  }

  gpuErrCheck(gpuDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", time * 1e-6f / repeat);

  gpuErrCheck(gpuMemcpy(p, d_p, size_bytes, gpuMemcpyDeviceToHost));

  if (print_output) {
    for (int j = 0; j < 10; ++j) {
      std::cout << "p[" << j << "] = " << p[j] << "\n";
    }
  }


  gpuErrCheck(gpuFree(d_p));
  gpuErrCheck(gpuFree(d_m));
  gpuErrCheck(gpuFree(d_v));
  gpuErrCheck(gpuFree(d_g));

  free(p);
  free(m);
  free(v);
  free(g);
  free(r);
  return 0;
}
