#pragma once

#ifdef PROTEUS_JIT
__attribute__((annotate("jit", 5, 6)))
#endif
__global__ void attention_kernel1(const float * key,
                                  const float * query,
                                  float * dot_product,
                                  float * exp_sum, const int n,
                                  const int d) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float sum = 0.0f;
    for (int j = 0; j < d; j++) {
      sum += key[i * d + j] * query[j];
    }
    dot_product[i] = sum;
    atomicAdd(exp_sum, __expf(sum));
  }
}

#ifdef PROTEUS_JIT
__attribute__((annotate("jit", 4)))
#endif
__global__ void attention_kernel2(const float * exp_sum,
                                  const float * dot_product,
                                  float * score, const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    score[i] = __expf(dot_product[i]) / exp_sum[0];
  }
}

#ifdef PROTEUS_JIT
__attribute__((annotate("jit", 4, 5)))
#endif
__global__ void attention_kernel3(const float * score,
                                  const float * value,
                                  float * output, const int n,
                                  const int d) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < d) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
      sum += score[i] * value[i * d + j];
    }
    output[j] = sum;
  }
}
