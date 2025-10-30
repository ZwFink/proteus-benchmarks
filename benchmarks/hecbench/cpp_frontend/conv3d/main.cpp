/*
  Reference
  Chapter 16 in Programming massively parallel processors,
  A hands-on approach (D. Kirk and W. Hwu)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <string>
#include <string_view>
#include <memory>
#include <tuple>
#include <utility>
#include <proteus/CppJitModule.hpp>
#include "inja/inja.h"
#include "../../../gpu/gpu_common.h"

using namespace proteus;

#define TILE_WIDTH 16

#if PROTEUS_ENABLE_HIP
constexpr const char *kDeviceInclude = "#include <hip/hip_runtime.h>";
#elif PROTEUS_ENABLE_CUDA
constexpr const char *kDeviceInclude = "#include <cuda_runtime.h>";
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA"
#endif

#define II(n,c,h,w) ((n)*C*Hin*Win+(c)*Hin*Win+(h)*Win+w)
#define WI(n,c,h,w) ((n)*C*K*K+(c)*K*K+(h)*K+w)
#define OI(n,c,h,w) ((n)*M*Hout*Wout+(c)*Hout*Wout+(h)*Wout+w)

void verify (const float* Y, float* Y_ref, size_t Y_size)
{
  bool ok = true;
  for (size_t i = 0; i < Y_size; i++) {
    if (fabs(Y[i] - Y_ref[i]) > 1e-3f) {
      printf("%f (device) != %f (reference)\n", Y[i], Y_ref[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
}

// Hin = Hout-1+K; max(h+p) is Hin - 1 as max(h) = Hout-1 and max(p) = K-1
void reference(const float * __restrict__ X,
               const float * __restrict__ W,
                     float * __restrict__ Y,
               const int N,
               const int M,
               const int C,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout)
{
  for(int n = 0; n < N; n++)
    for(int m = 0; m < M; m++)
      for(int h = 0; h < Hout; h++)
        for(int w = 0; w < Wout; w++) {
          Y[OI(n, m, h, w)] = 0;
          for(int c = 0; c < C; c++)
            for(int p = 0; p < K; p++)
              for(int q = 0; q < K; q++)
                Y[OI(n, m, h, w)] += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
}

// Kernel template containing all conv3d kernels
constexpr std::string_view StrConv3dKernelsTemplate = R"cpp({{ device_include }}
extern "C" __global__ void conv3d_s1(const float * __restrict__ X,
                                      const float * __restrict__ W,
                                            float * __restrict__ Y)
{
  constexpr int TILE_WIDTH = {{ tile_width }};
  constexpr int C = {{ C }};
  constexpr int M = {{ M }};
  constexpr int K = {{ K }};
  constexpr int Hin = {{ Hin }};
  constexpr int Win = {{ Win }};
  constexpr int Hout = {{ Hout }};
  constexpr int Wout = {{ Wout }};
  constexpr int W_grid = {{ W_grid }};

  #undef II
  #undef WI
  #undef OI
  #define II(n,c,h,w) ((n)*C*Hin*Win+(c)*Hin*Win+(h)*Win+w)
  #define WI(n,c,h,w) ((n)*C*K*K+(c)*K*K+(h)*K+w)
  #define OI(n,c,h,w) ((n)*M*Hout*Wout+(c)*Hout*Wout+(h)*Wout+w)

  int n = blockIdx.x;
  int m = blockIdx.y;
  int h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
  if (h < Hout && w < Wout) {
    float s = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
      }
    }
    Y[OI(n, m, h, w)] = s;
  }
#undef II
#undef WI
#undef OI
}

extern "C" __global__ void conv3d_s2(const float * __restrict__ X,
                                      const float * __restrict__ W,
                                            float * __restrict__ Y)
{
  constexpr int TILE_WIDTH = {{ tile_width }};
  constexpr int C = {{ C }};
  constexpr int M = {{ M }};
  constexpr int K = {{ K }};
  constexpr int Hin = {{ Hin }};
  constexpr int Win = {{ Win }};
  constexpr int Hout = {{ Hout }};
  constexpr int Wout = {{ Wout }};
  constexpr int W_grid = {{ W_grid }};

  #undef II
  #undef WI
  #undef OI
  #define II(n,c,h,w) ((n)*C*Hin*Win+(c)*Hin*Win+(h)*Win+w)
  #define WI(n,c,h,w) ((n)*C*K*K+(c)*K*K+(h)*K+w)
  #define OI(n,c,h,w) ((n)*M*Hout*Wout+(c)*Hout*Wout+(h)*Wout+w)

  int m = blockIdx.x;
  int h = blockIdx.y / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.y % W_grid * TILE_WIDTH + threadIdx.x;
  int n = blockIdx.z;
  if (h < Hout && w < Wout) {
    float s = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
      }
    }
    Y[OI(n, m, h, w)] = s;
  }
#undef II
#undef WI
#undef OI
}

extern "C" __global__ void conv3d_s3(const float * __restrict__ X,
                                      const float * __restrict__ W,
                                            float * __restrict__ Y)
{
  constexpr int TILE_WIDTH = {{ tile_width }};
  constexpr int C = {{ C }};
  constexpr int M = {{ M }};
  constexpr int K = {{ K }};
  constexpr int Hin = {{ Hin }};
  constexpr int Win = {{ Win }};
  constexpr int Hout = {{ Hout }};
  constexpr int Wout = {{ Wout }};
  constexpr int W_grid = {{ W_grid }};

  #undef II
  #undef WI
  #undef OI
  #define II(n,c,h,w) ((n)*C*Hin*Win+(c)*Hin*Win+(h)*Win+w)
  #define WI(n,c,h,w) ((n)*C*K*K+(c)*K*K+(h)*K+w)
  #define OI(n,c,h,w) ((n)*M*Hout*Wout+(c)*Hout*Wout+(h)*Wout+w)

  int h = blockIdx.x / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.x % W_grid * TILE_WIDTH + threadIdx.x;
  int n = blockIdx.y;
  int m = blockIdx.z;
  if (h < Hout && w < Wout) {
    float s = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
      }
    }
    Y[OI(n, m, h, w)] = s;
  }
#undef II
#undef WI
#undef OI
}
)cpp";

// Getter function for all conv3d kernels
static auto getConv3dKernels(int C, int M, int K, int Hin, int Win, int Hout, int Wout, int W_grid)
{
  Timer specializeTimer;
  specializeTimer.reset();
  inja::json data = {
    {"tile_width", TILE_WIDTH},
    {"C", C},
    {"M", M},
    {"K", K},
    {"Hin", Hin},
    {"Win", Win},
    {"Hout", Hout},
    {"Wout", Wout},
    {"W_grid", W_grid}
  };
  data["device_include"] = kDeviceInclude;
  auto kernelSource = inja::render(std::string{StrConv3dKernelsTemplate}, data);
  const auto specialize_ms = specializeTimer.elapsed();
  Logger::outs("Proteus") << "Specialized Kernel Construction "
                          << specialize_ms << " ms\n";
  auto JitMod = std::make_unique<CppJitModule>(TARGET, kernelSource);
  auto KernelS1 = JitMod->getKernel<void(const float *, const float *, float *)>("conv3d_s1");
  auto KernelS2 = JitMod->getKernel<void(const float *, const float *, float *)>("conv3d_s2");
  auto KernelS3 = JitMod->getKernel<void(const float *, const float *, float *)>("conv3d_s3");
  return std::tuple{std::move(JitMod), KernelS1, KernelS2, KernelS3};
}

void conv3D(const int N, const int C, const int M, const int Win, const int Hin, const int K, const int repeat, const int do_verify)
{
  const int Hout = Hin-K+1;
  const int Wout = Win-K+1;

  size_t X_size = N * C * Hin * Win;
  size_t W_size = M * C * K * K;
  size_t Y_size = N * M * Hout * Wout;
  size_t X_bytes = X_size * sizeof(float);
  size_t W_bytes = W_size * sizeof(float);
  size_t Y_bytes = Y_size * sizeof(float);

  float *X, *W, *Y, *Y_ref;
  X = (float *)malloc(X_bytes); // input
  W = (float *)malloc(W_bytes); // filter
  Y = (float *)malloc(Y_bytes); // output

  srand(123);


  if (do_verify) {

    for (size_t i = 0; i < W_size; i++) W[i] = rand() % 31;
    for (size_t i = 0; i < X_size; i++) X[i] = rand() % 13;

    for (size_t i = 0; i < Y_size; i++) {
      Y[i] = -1;
    }
    Y_ref = (float *)malloc(Y_bytes);
    for (size_t i = 0; i < Y_size; i++) {
      Y_ref[i] = -1;
    }
    reference(X, W, Y_ref, N, M, C, K, Hin, Win, Hout, Wout);
  }

  float *dX, *dW, *dY;
  gpuErrCheck(gpuMalloc((void **)&dX, X_bytes));
  gpuErrCheck(gpuMalloc((void **)&dW, W_bytes));
  gpuErrCheck(gpuMalloc((void **)&dY, Y_bytes));

  gpuErrCheck(gpuMemcpy(dX, X, X_bytes, gpuMemcpyHostToDevice));
  gpuErrCheck(gpuMemcpy(dW, W, W_bytes, gpuMemcpyHostToDevice));
  gpuErrCheck(gpuMemcpy(dY, Y, Y_bytes, gpuMemcpyHostToDevice));

  int W_grid = (Wout + TILE_WIDTH - 1) / TILE_WIDTH;
  int H_grid = (Hout + TILE_WIDTH - 1) / TILE_WIDTH;
  int Z = H_grid * W_grid;

  printf("input dimensions: C=%d Win=%d Hin=%d\n", C, Win, Hin);
  printf("output dimensions: M=%d Wout=%d Hout=%d\n", M, Wout, Hout);
  printf("3D grid dimensions: N=%d M=%d Z=%d\n", N, M, Z);

  gpuErrCheck(gpuDeviceSynchronize());

  // Get kernels with specialized parameters
  auto [JitMod, KernelS1, KernelS2, KernelS3] = getConv3dKernels(C, M, K, Hin, Win, Hout, Wout, W_grid);

  // Test conv3d_s1 (grid organization: N, M, Z)
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    KernelS1.launch({static_cast<unsigned int>(N), static_cast<unsigned int>(M), static_cast<unsigned int>(Z)},
                    {TILE_WIDTH, TILE_WIDTH, 1},
                    0, nullptr, dX, dW, dY);
  }

  gpuErrCheck(gpuDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s1 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  if (do_verify) {
    gpuErrCheck(gpuMemcpy(Y, dY, Y_bytes, gpuMemcpyDeviceToHost));
    verify(Y, Y_ref, Y_size);
  }

  // Test conv3d_s2 (grid organization: M, Z, N)
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    KernelS2.launch({static_cast<unsigned int>(M), static_cast<unsigned int>(Z), static_cast<unsigned int>(N)},
                    {TILE_WIDTH, TILE_WIDTH, 1},
                    0, nullptr, dX, dW, dY);
  }

  gpuErrCheck(gpuDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s2 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  if (do_verify) {
    gpuErrCheck(gpuMemcpy(Y, dY, Y_bytes, gpuMemcpyDeviceToHost));
    verify(Y, Y_ref, Y_size);
  }

  // Test conv3d_s3 (grid organization: Z, N, M)
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    KernelS3.launch({static_cast<unsigned int>(Z), static_cast<unsigned int>(N), static_cast<unsigned int>(M)},
                    {TILE_WIDTH, TILE_WIDTH, 1},
                    0, nullptr, dX, dW, dY);
  }

  gpuErrCheck(gpuDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s3 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  if (do_verify) {
    gpuErrCheck(gpuMemcpy(Y, dY, Y_bytes, gpuMemcpyDeviceToHost));
    verify(Y, Y_ref, Y_size);
  }

  free(X);
  free(W);
  free(Y);
  if (do_verify) {
    free(Y_ref);
  }
  gpuErrCheck(gpuFree(dX));
  gpuErrCheck(gpuFree(dW));
  gpuErrCheck(gpuFree(dY));
}

int main(int argc, char* argv[]) {
  if (argc != 8 && argc != 9) {
    printf("Usage: %s <batch size:N> <input channels:C> <output feature maps:M>", argv[0]);
    printf(" <input width:Win> <input height:Hin> <kernel size:K> <repeat> [verify (0 or 1, default 0)]\n");
    return 1;
  }

  int N = atoi(argv[1]);
  int C = atoi(argv[2]);
  int M = atoi(argv[3]);
  int W = atoi(argv[4]);
  int H = atoi(argv[5]);
  int K = atoi(argv[6]);
  int repeat = atoi(argv[7]);
  int verify = (argc == 9) ? atoi(argv[8]) : 0;

  printf("3D convolution (FP32)\n");
  gpu::warmup();
  printf("\n========== Warmup start ==========\n");
  conv3D(N, C, M, W, H, K, 1000, verify);
  printf("\n========== Warmup done ==========\n");
  conv3D(N, C, M, W, H, K, repeat, verify);

  return 0;
}
