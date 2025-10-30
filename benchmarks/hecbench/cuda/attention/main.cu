#include <chrono>
#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "../../../gpu/gpu_common.h"
#include "kernels.h"

float *attention_host(const float *key, const float *value, const float *query,
                      const int n, const int d) {
  float *dot_product = (float *)malloc(n * sizeof(float));
  float *score = (float *)malloc(n * sizeof(float));
  float *output = (float *)malloc(d * sizeof(float));

  for (int i = 0; i < n; i++) {
    float sum = 0.0f;
    for (int j = 0; j < d; j++) {
      sum += key[i * d + j] * query[j];
    }
    dot_product[i] = sum;
  }

  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += expf(dot_product[i]);
  }

  for (int i = 0; i < n; i++) {
    score[i] = expf(dot_product[i]) / sum;
  }

  for (int j = 0; j < d; j++) {
    float value_sum = 0.0f;
    for (int i = 0; i < n; i++) {
      value_sum += score[i] * value[i * d + j];
    }
    output[j] = value_sum;
  }

  free(dot_product);
  free(score);
  return output;
}

float *attention_device(const float *key, const float *value,
                        const float *query, const int n, const int d,
                        const int repeat, const int verify) {
  float *d_key = nullptr;
  gpuErrCheck(gpuMalloc((void **)&d_key, n * d * sizeof(float)));
  gpuErrCheck(gpuMemcpy(d_key, key, n * d * sizeof(float),
                        gpuMemcpyHostToDevice));

  float *d_value = nullptr;
  gpuErrCheck(gpuMalloc((void **)&d_value, n * d * sizeof(float)));
  gpuErrCheck(gpuMemcpy(d_value, value, n * d * sizeof(float),
                        gpuMemcpyHostToDevice));

  float *d_query = nullptr;
  gpuErrCheck(gpuMalloc((void **)&d_query, d * sizeof(float)));
  gpuErrCheck(
      gpuMemcpy(d_query, query, d * sizeof(float), gpuMemcpyHostToDevice));

  float *d_dot_product = nullptr;
  gpuErrCheck(gpuMalloc((void **)&d_dot_product, n * sizeof(float)));

  float *d_exp_sum = nullptr;
  gpuErrCheck(gpuMalloc((void **)&d_exp_sum, sizeof(float)));

  float *output = (float *)malloc(d * sizeof(float));
  float *d_output = nullptr;
  gpuErrCheck(gpuMalloc((void **)&d_output, d * sizeof(float)));

  float *d_score = nullptr;
  gpuErrCheck(gpuMalloc((void **)&d_score, n * sizeof(float)));

  gpuErrCheck(gpuDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();

  for (int k = 0; k < repeat; k++) {
    if (verify) {
      gpuErrCheck(gpuMemset(d_exp_sum, 0, sizeof(float)));
    }
    attention_kernel1<<<(n + 255) / 256, 256>>>(d_key, d_query, d_dot_product,
                                                d_exp_sum, n, d);
    attention_kernel2<<<(n + 255) / 256, 256>>>(d_exp_sum, d_dot_product,
                                                d_score, n);
    attention_kernel3<<<(d + 255) / 256, 256>>>(d_score, d_value, d_output, n,
                                                d);
  }

  gpuErrCheck(gpuDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernels %f (ms)\n",
         time * 1e-6f / repeat);

  gpuErrCheck(gpuMemcpy(output, d_output, d * sizeof(float),
                        gpuMemcpyDeviceToHost));

  gpuErrCheck(gpuFree(d_score));
  gpuErrCheck(gpuFree(d_value));
  gpuErrCheck(gpuFree(d_output));
  gpuErrCheck(gpuFree(d_key));
  gpuErrCheck(gpuFree(d_dot_product));
  gpuErrCheck(gpuFree(d_exp_sum));
  gpuErrCheck(gpuFree(d_query));

  return output;
}

int main(int argc, char *argv[]) {
  gpu::warmup();
  if (argc != 4 && argc != 5) {
    printf("Usage: %s <rows> <columns> <repeat> [verify]\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int d = atoi(argv[2]);
  const int r = atoi(argv[3]);
  const int verify = (argc == 5) ? atoi(argv[4]) : 0;

  float *key = (float *)malloc(n * d * sizeof(float));
  float *value = (float *)malloc(n * d * sizeof(float));
  float *query = (float *)malloc(d * sizeof(float));

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  if (verify) {
    for (int i = 0; i < n * d; i++) {
      key[i] = dist(gen);
      value[i] = dist(gen);
      query[i % d] = dist(gen);
    }
  }

  float *dout = attention_device(key, value, query, n, d, r, verify);

  if (verify) {
    float *hout = attention_host(key, value, query, n, d);

    float rmse = 0.0f;
    for (int i = 0; i < d; i++) {
      const float diff = hout[i] - dout[i];
      rmse += diff * diff;
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
