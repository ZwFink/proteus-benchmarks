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

#if PROTEUS_ENABLE_HIP
constexpr const char *kDeviceInclude = "#include <hip/hip_runtime.h>";
#elif PROTEUS_ENABLE_CUDA
constexpr const char *kDeviceInclude = "#include <cuda_runtime.h>";
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

constexpr int MatmulTileSize = 16;
constexpr std::string_view StrGpuNontiledMatmulKernelTemplate = R"cpp(
{{ include }}
extern "C" __global__ void gpuNontiledMatmulKernel(const double * A,
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
#if PROTEUS_ENABLE_HIP
// Non-tiled (naive) GPU kernel for matrix multiplication: C = A * B
__global__ void gpuNontiledMatmulKernelStatic (const double * __restrict__ A,
                                           const double * __restrict__ B,
                                           double * __restrict__ C,
                                           int N) {
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

static inline void gpuNontiledMatmulLaunch(const double *A,
                                              const double *B,
                                              double *C,
                                              int N) {
  dim3 Block(MatmulTileSize, MatmulTileSize, 1);
  dim3 Grid((N + Block.x - 1) / Block.x,
            (N + Block.y - 1) / Block.y,
            1);
  gpuNontiledMatmulKernelStatic<<<Grid, Block>>>(A, B, C, N);
}
#endif


// GPU kernel: Register + shared-memory tiled matmul (C = A x B)
// Defaults match pj-dsl structure (constexpr, not macros)
constexpr int RegTileM = 4;
constexpr int RegTileN = 4;
constexpr int BlockTileM = 64;
constexpr int BlockTileN = 64;
constexpr int KTile = 8;
constexpr std::string_view StrGpuRegSharedTiledMatmulKernelTemplate = R"cpp(
{{ include }}
extern "C" __global__ void gpuRegSharedTiledMatmulKernel(const double * A,
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
    {"include", std::string{kDeviceInclude}},
    {"N", N},
    {"BlockTileM", BlockTileM},
    {"KTile", KTile},
    {"BlockTileN", BlockTileN},
    {"RegTileM", RegTileM},
    {"RegTileN", RegTileN}
  };
  auto KernelStr = inja::render(std::string{StrGpuRegSharedTiledMatmulKernelTemplate}, data);
  auto JitMod = std::make_unique<CppJitModule>(TARGET, KernelStr);
  auto Kernel = JitMod->getKernel<void(const double *, const double *, double *)>("gpuRegSharedTiledMatmulKernel");
  return std::make_pair(std::move(JitMod), Kernel);
}

static auto getNontiledMatMulKernel(int N)
{
  inja::json data = {
    {"include", std::string{kDeviceInclude}},
    {"N", N}
  };
  auto KernelStr = inja::render(std::string{StrGpuNontiledMatmulKernelTemplate}, data);
  auto JitMod = std::make_unique<CppJitModule>(TARGET, KernelStr);
  auto Kernel = JitMod->getKernel<void(const double *, const double *, double *)>("gpuNontiledMatmulKernel");
  return std::make_pair(std::move(JitMod), Kernel);
}

static bool verify(double *C, int N) {
  for (int I = 0; I < N; I++) {
    for (int J = 0; J < N; J++) {
      if (C[I*N + J] != N) {
        return false;
      }
    }
  }
  return true;
}

int main(int argc, char** argv) {
  unsigned int N = 8192;
  int NumTrials = 5;
  bool DoVerify = true;
  std::string KernelType = "gpu_regtiled";
  // Positional tile sizes (mirroring pj-dsl main)
  int blockTileMArg = BlockTileM;
  int blockTileNArg = BlockTileN;
  int kTileArg = KTile;
  int posIdx = 0;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--N") || !std::strcmp(argv[i], "-n")) {
      if (i + 1 < argc) {
        N = static_cast<unsigned int>(std::atoi(argv[++i]));
      }
    } else if (!std::strcmp(argv[i], "--trials") || !std::strcmp(argv[i], "-t")) {
      if (i + 1 < argc) {
        NumTrials = std::atoi(argv[++i]);
      }
    } else if (!std::strcmp(argv[i], "--kernel")) {
      if (i + 1 < argc) {
        KernelType = argv[++i];
        if (KernelType != "hip" && KernelType != "gpu_naive" && KernelType != "gpu_regtiled") {
          std::cerr << "Error: Invalid kernel type '" << KernelType
                    << "'. Valid options: hip (alias for gpu_naive), gpu_naive, gpu_regtiled\n";
          return 1;
        }
      }
    } else if (!std::strcmp(argv[i], "--verify")) {
      DoVerify = true;
    } else if (!std::strcmp(argv[i], "--no-verify")) {
      DoVerify = false;
    } else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
      std::cout << "Usage: " << argv[0]
                << " [-n|--N N] [-t|--trials T] [--kernel KERNEL] [--verify|--no-verify]"
                << " [blockTileM blockTileN kTile]\n"
                << "  KERNEL: hip (alias of gpu_naive), gpu_naive, gpu_regtiled (default: gpu_regtiled)\n"
                << "  Positional tile sizes are used for the reg-tiled kernel;"
                << " defaults are " << BlockTileM << " " << BlockTileN << " " << KTile << "\n";
      return 0;
    } else {
      // Positional ints for blockTileM, blockTileN, kTile (for reg-tiled)
      if (argv[i] && std::isdigit(static_cast<unsigned char>(argv[i][0]))) {
        int val = std::atoi(argv[i]);
        if (posIdx == 0) blockTileMArg = val;
        else if (posIdx == 1) blockTileNArg = val;
        else if (posIdx == 2) kTileArg = val;
        ++posIdx;
      }
    }
  }
  std::cout << "Configuration: (N, NumTrials, DoVerify, Kernel, Tiles) = ("
            << N << ", " << NumTrials << ", " << DoVerify << ", " << KernelType
            << ", blkM=" << blockTileMArg << ", blkN=" << blockTileNArg
            << ", k=" << kTileArg << ")" << std::endl;

  // Warn if runtime tile args differ from compile-time constants (HIP kernel is static)
  if (KernelType == "gpu_regtiled") {
    if (blockTileMArg != BlockTileM || blockTileNArg != BlockTileN || kTileArg != KTile) {
      std::cerr << "Warning: Reg-tiled kernel uses compile-time tile sizes; ignoring positional values. Using blkM="
                << BlockTileM << ", blkN=" << BlockTileN << ", k=" << KTile << "\n";
    }
  }

  // Host allocations
  double *AH = (double *)new double[N * N];
  double *BH = (double *)new double[N * N];
  double *CH = (double *)new double[N * N];

  // Device allocations
  double *AD;
  double *BD;
  double *CD;
  size_t Bytes = sizeof(double) * N * N;
  gpuErrCheck(gpuMalloc((void **)&AD, Bytes));
  gpuErrCheck(gpuMalloc((void **)&BD, Bytes));
  gpuErrCheck(gpuMalloc((void **)&CD, Bytes));

  if (DoVerify) {
    for (int I = 0; I < (int)N; I++) {
      for (int J = 0; J < (int)N; J++) {
        AH[I*N+J] = 1.0;
        BH[I*N+J] = 1.0;
        CH[I*N+J] = 0.0;
      }
    }
  }

  // Stage inputs to device
  gpuErrCheck(gpuMemcpy(AD, AH, Bytes, gpuMemcpyHostToDevice));
  gpuErrCheck(gpuMemcpy(BD, BH, Bytes, gpuMemcpyHostToDevice));
  gpuErrCheck(gpuMemcpy(CD, CH, Bytes, gpuMemcpyHostToDevice));

  // Kernel execution based on type
  if (KernelType == "gpu_regtiled") {
    Timer specializeTimer;
    specializeTimer.reset();
    auto [JitMod, Kernel] = getRegSharedTiledMatMulKernel(N);
    Logger::outs("Proteus") << "Specialized Kernel Construction "
                            << specializeTimer.elapsed() << " ms\n";
    Kernel.launch({static_cast<unsigned int>(N / BlockTileN), static_cast<unsigned int>(N / BlockTileM), 1},
                  {static_cast<unsigned int>(BlockTileN / RegTileN), static_cast<unsigned int>(BlockTileM / RegTileM), 1},
                  0, nullptr, AD, BD, CD);
    gpuErrCheck(gpuDeviceSynchronize());

    // Timed trials
    double TotalMs = 0.0;
    auto Start = std::chrono::high_resolution_clock::now();
    for (int T = 0; T < NumTrials; ++T) {
      Kernel.launch({static_cast<unsigned int>(N / BlockTileN), static_cast<unsigned int>(N / BlockTileM), 1},
                    {static_cast<unsigned int>(BlockTileN / RegTileN), static_cast<unsigned int>(BlockTileM / RegTileM), 1},
                    0, nullptr, AD, BD, CD);
    }
    gpuErrCheck(gpuDeviceSynchronize());
    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> Ms = End - Start;
    TotalMs = Ms.count();
    double AvgMs = TotalMs / static_cast<double>(NumTrials);
    std::cerr.setf(std::ios::fixed);
    std::cerr.precision(3);
    std::cerr << "Average over " << NumTrials << " trials: " << AvgMs << " ms" << '\n';

  } else if (KernelType == "hip" || KernelType == "gpu_naive") {
    Timer specializeTimer;
    specializeTimer.reset();
    auto [JitModNT, KernelNT] = getNontiledMatMulKernel(N);
    Logger::outs("Proteus") << "Specialized Kernel Construction "
                            << specializeTimer.elapsed() << " ms\n";
    unsigned int blockX = static_cast<unsigned int>(MatmulTileSize);
    unsigned int blockY = static_cast<unsigned int>(MatmulTileSize);
    unsigned int gridX = static_cast<unsigned int>((N + blockX - 1) / blockX);
    unsigned int gridY = static_cast<unsigned int>((N + blockY - 1) / blockY);

    KernelNT.launch({gridX, gridY, 1}, {blockX, blockY, 1}, 0, nullptr, AD, BD, CD);
    gpuErrCheck(gpuDeviceSynchronize());

    // Timed trials
    double TotalMs = 0.0;
    auto Start = std::chrono::high_resolution_clock::now();
    for (int T = 0; T < NumTrials; ++T) {
      KernelNT.launch({gridX, gridY, 1}, {blockX, blockY, 1}, 0, nullptr, AD, BD, CD);
    }
    gpuErrCheck(gpuDeviceSynchronize());
    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> Ms = End - Start;
    TotalMs = Ms.count();
    double AvgMs = TotalMs / static_cast<double>(NumTrials);
    std::cerr.setf(std::ios::fixed);
    std::cerr.precision(3);
    std::cerr << "Average over " << NumTrials << " trials: " << AvgMs << " ms" << '\n';
  }

  // Final result back to host after timing loop
  if (DoVerify) {
    gpuErrCheck(gpuMemcpy(CH, CD, Bytes, gpuMemcpyDeviceToHost));
    if (verify(CH, N)) {
      std::cout << "Verification passed" << std::endl;
    } else {
      std::cout << "Verification failed" << std::endl;
    }
  }

  // Cleanup device and host memory
  gpuErrCheck(gpuFree(AD));
  gpuErrCheck(gpuFree(BD));
  gpuErrCheck(gpuFree(CD));
  delete[] AH;
  delete[] BH;
  delete[] CH;

  return 0;

}
