#include <cstdlib>
#include <cstring>
#include <cctype>
#include <iostream>
#include <chrono>
#include <string>
#include <string_view>
#include <memory>
#include <utility>
#include <proteus/CppJitModule.hpp>
#include "inja/inja.h"

using namespace proteus;
#include "../../../gpu/gpu_common.h"

constexpr const char *kDeviceInclude = "#include <hip/hip_runtime.h>";

constexpr int MatmulTileSize = 16;
constexpr std::string_view StrHipNontiledMatmulKernelTemplate = R"cpp(
{{ include }}
extern "C" __global__ void hipNontiledMatmulKernel(const double * A,
                                              const double * B,
                                              double * C) {
  constexpr int N = {{ N }};
  int Tx = threadIdx.x;
  int Ty = threadIdx.y;
  int Row = blockIdx.y * blockDim.y + Ty;
  int Col = blockIdx.x * blockDim.x + Tx;

  if (Row < N && Col < N) {
    double Sum = 0.0;
    #pragma unroll 1
    for (int K = 0; K < N; ++K) {
      Sum += A[Row * N + K] * B[K * N + Col];
    }
    C[Row * N + Col] = Sum;
  }
}
)cpp";

constexpr int RegTileM = 4;
constexpr int RegTileN = 4;
constexpr int BlockTileM = 64;
constexpr int BlockTileN = 64;
constexpr int KTile = 8;
constexpr std::string_view StrHipRegSharedTiledMatmulKernelTemplate = R"cpp(
{{ include }}
extern "C" __global__ void hipRegSharedTiledMatmulKernel(const double * A,
                                              const double * B,
                                              double * C) {
  // Shared tiles for current K-slice via dynamic shared memory
  // 1 = blockTimeM, 2 = kTile
  constexpr int N = {{ N }};
  constexpr int BlockTileM = {{ BlockTileM }};
  constexpr int KTile = {{ KTile }};
  constexpr int BlockTileN = {{ BlockTileN }};
  constexpr int RegTileM = {{ RegTileM }};
  constexpr int RegTileN = {{ RegTileN }};

  __shared__ double smem[(BlockTileM*KTile) + (KTile*BlockTileN)];
  double *AsTile = smem;                                 // [BlockTileM x KTile]
  double *BsTile = AsTile + (BlockTileM * KTile);        // [KTile x BlockTileN]

  // Thread indices and block coordinates
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Block origin within C
  int blockRow = by * BlockTileM;
  int blockCol = bx * BlockTileN;

  // Per-thread micro-tile origin within C
  int row0 = blockRow + ty * RegTileM;
  int col0 = blockCol + tx * RegTileN;

  // Register accumulators
  double Creg[RegTileM * RegTileN];
  #pragma unroll
  for (int i = 0; i < RegTileM * RegTileN; ++i) { Creg[i] = 0.0; }

  // Linear thread id for cooperative loads
  const int threadsX = BlockTileN / RegTileN;
  const int tid = ty * threadsX + tx; // 0 .. ((BlockTileM/RegTileM)*(BlockTileN/RegTileN) - 1)

  // Loop over K dimension in tiles of K_TILE
  for (int kBase = 0; kBase < N; kBase += KTile) {
    // Each thread loads two elements of AsTile and BsTile
    int aIdx0 = tid * 2 + 0;
    int aIdx1 = tid * 2 + 1;
    if (aIdx0 < BlockTileM * KTile) {
      int aRow0 = aIdx0 / KTile;
      int aCol0 = aIdx0 % KTile;
      int asIdx0 = aRow0 * KTile + aCol0;
      int aGlobIdx0 = (blockRow + aRow0) * N + (kBase + aCol0);
      AsTile[asIdx0] = A[aGlobIdx0];
    }
    if (aIdx1 < BlockTileM * KTile) {
      int aRow1 = aIdx1 / KTile;
      int aCol1 = aIdx1 % KTile;
      int asIdx1 = aRow1 * KTile + aCol1;
      int aGlobIdx1 = (blockRow + aRow1) * N + (kBase + aCol1);
      AsTile[asIdx1] = A[aGlobIdx1];
    }

    int bIdx0 = tid * 2 + 0;
    int bIdx1 = tid * 2 + 1;
    if (bIdx0 < KTile * BlockTileN) {
      int bRow0 = bIdx0 / BlockTileN;
      int bCol0 = bIdx0 % BlockTileN;
      int bsIdx0 = bRow0 * BlockTileN + bCol0;
      int bGlobIdx0 = (kBase + bRow0) * N + (blockCol + bCol0);
      BsTile[bsIdx0] = B[bGlobIdx0];
    }
    if (bIdx1 < KTile * BlockTileN) {
      int bRow1 = bIdx1 / BlockTileN;
      int bCol1 = bIdx1 % BlockTileN;
      int bsIdx1 = bRow1 * BlockTileN + bCol1;
      int bGlobIdx1 = (kBase + bRow1) * N + (blockCol + bCol1);
      BsTile[bsIdx1] = B[bGlobIdx1];
    }

    __syncthreads();

    // Compute this micro-tile using the shared tiles and register blocking
    double Areg[RegTileM];
    double Breg[RegTileN];

    #pragma unroll
    for (int kIt = 0; kIt < KTile; ++kIt) {
      // Load a REG_TILE_M row from AsTile into registers
      #pragma unroll
      for (int i = 0; i < RegTileM; ++i) {
        int r = ty * RegTileM + i;
        int asIdx = r * KTile + kIt;
        Areg[i] = AsTile[asIdx];
      }

      // Load a REG_TILE_N column from BsTile into registers
      #pragma unroll
      for (int j = 0; j < RegTileN; ++j) {
        int c = tx * RegTileN + j;
        int bsIdx = kIt * BlockTileN + c;
        Breg[j] = BsTile[bsIdx];
      }

      // FMA on the per-thread micro-tile
      #pragma unroll
      for (int i = 0; i < RegTileM; ++i) {
        #pragma unroll
        for (int j = 0; j < RegTileN; ++j) {
          int cIdx = i * RegTileN + j;
          Creg[cIdx] += Areg[i] * Breg[j];
        }
      }
    }

    __syncthreads();
  }

  // Write back the per-thread micro-tile to C
  #pragma unroll
  for (int i = 0; i < RegTileM; ++i) {
    #pragma unroll
    for (int j = 0; j < RegTileN; ++j) {
      int cGlobIdx = (row0 + i) * N + (col0 + j);
      int rIdx = i * RegTileN + j;
      C[cGlobIdx] = Creg[rIdx];
    }
  }
}
)cpp";

static auto getRegSharedTiledMatMulKernel(int N)
{
  inja::json data = {
    {"include", kDeviceInclude},
    {"N", N},
    {"BlockTileM", BlockTileM},
    {"KTile", KTile},
    {"BlockTileN", BlockTileN},
    {"RegTileM", RegTileM},
    {"RegTileN", RegTileN}
  };
  auto KernelStr = inja::render(std::string{StrHipRegSharedTiledMatmulKernelTemplate}, data);
  auto JitMod = std::make_unique<CppJitModule>("hip", KernelStr);
  auto Kernel = JitMod->getKernel<void(const double *, const double *, double *)>("hipRegSharedTiledMatmulKernel");
  return std::make_pair(std::move(JitMod), Kernel);
}

static auto getNontiledMatMulKernel(int N)
{
  inja::json data = {
    {"include", kDeviceInclude},
    {"N", N}
  };
  auto KernelStr = inja::render(std::string{StrHipNontiledMatmulKernelTemplate}, data);
  auto JitMod = std::make_unique<CppJitModule>("hip", KernelStr);
  auto Kernel = JitMod->getKernel<void(const double *, const double *, double *)>("hipNontiledMatmulKernel");
  return std::make_pair(std::move(JitMod), Kernel);
}

static bool verify_all_ones(double *G, int N) {
  // With A,B,C,D all ones: E=N, F=N, G=N^3 everywhere
  const double expected = static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
  for (int i = 0; i < N * N; ++i) {
    if (G[i] != expected) return false;
  }
  return true;
}

int main(int argc, char** argv) {
  unsigned int N = 2048;
  int NumTrials = 5;
  bool DoVerify = true;
  std::string KernelType = "gpu_regtiled";

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--N") || !std::strcmp(argv[i], "-n")) {
      if (i + 1 < argc) N = static_cast<unsigned int>(std::atoi(argv[++i]));
    } else if (!std::strcmp(argv[i], "--trials") || !std::strcmp(argv[i], "-t")) {
      if (i + 1 < argc) NumTrials = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--verify")) {
      DoVerify = true;
    } else if (!std::strcmp(argv[i], "--no-verify")) {
      DoVerify = false;
    } else if (!std::strcmp(argv[i], "--kernel")) {
      if (i + 1 < argc) KernelType = argv[++i];
    } else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
      std::cout << "Usage: " << argv[0]
                << " [-n|--N N] [-t|--trials T] [--verify|--no-verify] [--kernel KERNEL)\n"
                << "  KERNEL: hip, gpu_regtiled (default: gpu_regtiled)" << std::endl;
      return 0;
    }
  }

  if (KernelType != "hip" && KernelType != "gpu_regtiled") {
    std::cerr << "Error: Invalid kernel type '" << KernelType
              << "'. Valid options: hip, gpu_regtiled\n";
    return 1;
  }

  std::cout << "3mm cpp-frontend: N=" << N << ", trials=" << NumTrials << ", verify=" << DoVerify
            << ", kernel=" << KernelType << std::endl;

  size_t Bytes = sizeof(double) * N * N;

  // Host matrices: A,B,C,D inputs; E,F intermediates; G output
  double *AH = (double *)new double[N * N];
  double *BH = (double *)new double[N * N];
  double *CH = (double *)new double[N * N];
  double *DH = (double *)new double[N * N];
  double *EH = (double *)new double[N * N];
  double *FH = (double *)new double[N * N];
  double *GH = (double *)new double[N * N];

  if (DoVerify) {
    for (unsigned int i = 0; i < N * N; ++i) {
      AH[i] = 1.0; BH[i] = 1.0; CH[i] = 1.0; DH[i] = 1.0;
      EH[i] = 0.0; FH[i] = 0.0; GH[i] = 0.0;
    }
  }

  // Device allocations: A,B,C,D,E,F,G
  double *AD, *BD, *CD, *DD, *ED, *FD, *GD;
  gpuErrCheck(gpuMalloc((void **)&AD, Bytes));
  gpuErrCheck(gpuMalloc((void **)&BD, Bytes));
  gpuErrCheck(gpuMalloc((void **)&CD, Bytes));
  gpuErrCheck(gpuMalloc((void **)&DD, Bytes));
  gpuErrCheck(gpuMalloc((void **)&ED, Bytes));
  gpuErrCheck(gpuMalloc((void **)&FD, Bytes));
  gpuErrCheck(gpuMalloc((void **)&GD, Bytes));

  // Copy inputs
  gpuErrCheck(gpuMemcpy(AD, AH, Bytes, gpuMemcpyHostToDevice));
  gpuErrCheck(gpuMemcpy(BD, BH, Bytes, gpuMemcpyHostToDevice));
  gpuErrCheck(gpuMemcpy(CD, CH, Bytes, gpuMemcpyHostToDevice));
  gpuErrCheck(gpuMemcpy(DD, DH, Bytes, gpuMemcpyHostToDevice));

  if (KernelType == "gpu_regtiled") {
    auto [JitMod, Kernel] = getRegSharedTiledMatMulKernel(N);

    // Warm-up: E=A*B, F=C*D, G=E*F
    Kernel.launch({static_cast<unsigned int>(N / BlockTileN), static_cast<unsigned int>(N / BlockTileM), 1},
                  {static_cast<unsigned int>(BlockTileN / RegTileN), static_cast<unsigned int>(BlockTileM / RegTileM), 1},
                  0, nullptr, AD, BD, ED);
    Kernel.launch({static_cast<unsigned int>(N / BlockTileN), static_cast<unsigned int>(N / BlockTileM), 1},
                  {static_cast<unsigned int>(BlockTileN / RegTileN), static_cast<unsigned int>(BlockTileM / RegTileM), 1},
                  0, nullptr, CD, DD, FD);
    Kernel.launch({static_cast<unsigned int>(N / BlockTileN), static_cast<unsigned int>(N / BlockTileM), 1},
                  {static_cast<unsigned int>(BlockTileN / RegTileN), static_cast<unsigned int>(BlockTileM / RegTileM), 1},
                  0, nullptr, ED, FD, GD);
    gpuErrCheck(gpuDeviceSynchronize());

    // Timed trials across the 3 multiplies
    double TotalMs = 0.0;
    auto Start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < NumTrials; ++t) {
      Kernel.launch({static_cast<unsigned int>(N / BlockTileN), static_cast<unsigned int>(N / BlockTileM), 1},
                    {static_cast<unsigned int>(BlockTileN / RegTileN), static_cast<unsigned int>(BlockTileM / RegTileM), 1},
                    0, nullptr, AD, BD, ED);
      Kernel.launch({static_cast<unsigned int>(N / BlockTileN), static_cast<unsigned int>(N / BlockTileM), 1},
                    {static_cast<unsigned int>(BlockTileN / RegTileN), static_cast<unsigned int>(BlockTileM / RegTileM), 1},
                    0, nullptr, CD, DD, FD);
      Kernel.launch({static_cast<unsigned int>(N / BlockTileN), static_cast<unsigned int>(N / BlockTileM), 1},
                    {static_cast<unsigned int>(BlockTileN / RegTileN), static_cast<unsigned int>(BlockTileM / RegTileM), 1},
                    0, nullptr, ED, FD, GD);
    }
    gpuErrCheck(gpuDeviceSynchronize());
    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> Ms = End - Start;
    TotalMs = Ms.count();
    double AvgMs = TotalMs / static_cast<double>(NumTrials);
    std::cerr.setf(std::ios::fixed);
    std::cerr.precision(3);
    std::cerr << "Average over " << NumTrials << " trials (3 GEMMs): " << AvgMs << " ms" << '\n';

  } else {
    auto [JitModNT, KernelNT] = getNontiledMatMulKernel(N);
    unsigned int blockX = static_cast<unsigned int>(MatmulTileSize);
    unsigned int blockY = static_cast<unsigned int>(MatmulTileSize);
    unsigned int gridX = static_cast<unsigned int>((N + blockX - 1) / blockX);
    unsigned int gridY = static_cast<unsigned int>((N + blockY - 1) / blockY);

    // Warm-up: E=A*B, F=C*D, G=E*F
    KernelNT.launch({gridX, gridY, 1}, {blockX, blockY, 1}, 0, nullptr, AD, BD, ED);
    KernelNT.launch({gridX, gridY, 1}, {blockX, blockY, 1}, 0, nullptr, CD, DD, FD);
    KernelNT.launch({gridX, gridY, 1}, {blockX, blockY, 1}, 0, nullptr, ED, FD, GD);
    gpuErrCheck(gpuDeviceSynchronize());

    // Timed trials across the 3 multiplies
    double TotalMs = 0.0;
    auto Start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < NumTrials; ++t) {
      KernelNT.launch({gridX, gridY, 1}, {blockX, blockY, 1}, 0, nullptr, AD, BD, ED);
      KernelNT.launch({gridX, gridY, 1}, {blockX, blockY, 1}, 0, nullptr, CD, DD, FD);
      KernelNT.launch({gridX, gridY, 1}, {blockX, blockY, 1}, 0, nullptr, ED, FD, GD);
    }
    gpuErrCheck(gpuDeviceSynchronize());
    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> Ms = End - Start;
    TotalMs = Ms.count();
    double AvgMs = TotalMs / static_cast<double>(NumTrials);
    std::cerr.setf(std::ios::fixed);
    std::cerr.precision(3);
    std::cerr << "Average over " << NumTrials << " trials (3 GEMMs): " << AvgMs << " ms" << '\n';
  }

  if (DoVerify) {
    gpuErrCheck(gpuMemcpy(GH, GD, Bytes, gpuMemcpyDeviceToHost));
    if (verify_all_ones(GH, N)) std::cout << "Verification passed" << std::endl;
    else                        std::cout << "Verification failed" << std::endl;
  }

  // Cleanup
  gpuErrCheck(gpuFree(AD));
  gpuErrCheck(gpuFree(BD));
  gpuErrCheck(gpuFree(CD));
  gpuErrCheck(gpuFree(DD));
  gpuErrCheck(gpuFree(ED));
  gpuErrCheck(gpuFree(FD));
  gpuErrCheck(gpuFree(GD));
  delete[] AH; delete[] BH; delete[] CH; delete[] DH;
  delete[] EH; delete[] FH; delete[] GH;

  return 0;
}
