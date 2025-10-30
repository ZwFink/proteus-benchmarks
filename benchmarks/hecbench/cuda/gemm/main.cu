// RUN: rm -rf .proteus
// RUN: ./for | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <cstdlib>
#include <cstring>
#include <proteus/JitInterface.hpp>
#include <cctype>
#include <iostream>
#include <chrono>
#include <string>


#include "../../../gpu/gpu_common.h"

constexpr int MatmulTileSize = 16;
// Non-tiled (naive) CUDA kernel for matrix multiplication: C = A * B
__attribute__((annotate("jit", 4)))
__global__
void cudaNontiledMatmulKernel(const double * __restrict__ A,
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

static inline void cudaNontiledMatmulLaunch(const double *A,
                                              const double *B,
                                              double *C,
                                              int N) {
  dim3 Block(MatmulTileSize, MatmulTileSize, 1);
  dim3 Grid((N + Block.x - 1) / Block.x,
            (N + Block.y - 1) / Block.y,
            1);
  cudaNontiledMatmulKernel<<<Grid, Block>>>(A, B, C, N);
}


// CUDA kernel: Register + shared-memory tiled matmul (C = A x B)
// Defaults match pj-dsl structure (constexpr, not macros)
constexpr int RegTileM = 4;
constexpr int RegTileN = 4;
constexpr int BlockTileM = 64;
constexpr int BlockTileN = 64;
constexpr int KTile = 8;
#ifdef ENABLE_PROTEUS
#warning "PROTEUS_ENABLE_CUDA"
__attribute__((annotate("jit", 4)))
#endif
__global__
void cudaRegSharedTiledMatmulKernel(const double * __restrict__ A,
                                              const double * __restrict__ B,
                                              double * __restrict__ C,
                                              int N) {
  // Shared tiles for current K-slice via dynamic shared memory
  #ifdef ENABLE_PROTEUS
  double *smem = proteus::shared_array<double, BlockTileM * KTile + KTile * BlockTileN>(
      BlockTileM * KTile + KTile * BlockTileN);
  #else
  extern __shared__ double smem[];
  #endif
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
  for (int i = 0; i < RegTileM * RegTileN; ++i) Creg[i] = 0.0;

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

static inline void cudaRegSharedTiledMatmulLaunch(const double *A,
                                                 const double *B,
                                                 double *C,
                                                 int N) {
  dim3 Block(BlockTileN / RegTileN, BlockTileM / RegTileM, 1);
  dim3 Grid(N / BlockTileN, N / BlockTileM, 1);
  #ifdef ENABLE_PROTEUS
  cudaRegSharedTiledMatmulKernel<<<Grid, Block>>>(A, B, C, N);
  #else
  size_t SmemBytes = (BlockTileM * KTile + KTile * BlockTileN) * sizeof(double);
  cudaRegSharedTiledMatmulKernel<<<Grid, Block, SmemBytes>>>(A, B, C, N);
  #endif
}

// Verification helper to mirror pj-dsl structure
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
  gpu::warmup();
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
        if (KernelType != "cuda" && KernelType != "hip" && KernelType != "gpu_regtiled") {
          std::cerr << "Error: Invalid kernel type '" << KernelType
                    << "'. Valid options: cuda, gpu_regtiled\n";
          return 1;
        }
        if (KernelType == "hip") {
          std::cerr << "Warning: kernel type 'hip' is deprecated; use 'cuda' for the non-tiled baseline.\n";
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
                << "  KERNEL: cuda, gpu_regtiled (default: gpu_regtiled)\n"
                << "  Positional tile sizes are used for the CUDA reg-tiled kernel;"
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

  // Warn if runtime tile args differ from compile-time constants (CUDA kernel is static)
  if (KernelType == "gpu_regtiled") {
    if (blockTileMArg != BlockTileM || blockTileNArg != BlockTileN || kTileArg != KTile) {
      std::cerr << "Warning: CUDA reg-tiled kernel uses compile-time tile sizes; ignoring positional values. Using blkM="
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
    cudaRegSharedTiledMatmulLaunch(AD, BD, CD, N);
    gpuErrCheck(gpuDeviceSynchronize());

    // Timed trials
    double TotalMs = 0.0;
    auto Start = std::chrono::high_resolution_clock::now();
    for (int T = 0; T < NumTrials; ++T) {
      cudaRegSharedTiledMatmulLaunch(AD, BD, CD, N);
    }
    gpuErrCheck(gpuDeviceSynchronize());
    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> Ms = End - Start;
    TotalMs = Ms.count();
    double AvgMs = TotalMs / static_cast<double>(NumTrials);
    std::cerr.setf(std::ios::fixed);
    std::cerr.precision(3);
    std::cerr << "Average over " << NumTrials << " trials: " << AvgMs << " ms" << '\n';

  } else {
    cudaNontiledMatmulLaunch(AD, BD, CD, N);
    gpuErrCheck(gpuDeviceSynchronize());

    // Timed trials
    double TotalMs = 0.0;
    auto Start = std::chrono::high_resolution_clock::now();
    for (int T = 0; T < NumTrials; ++T) {
      cudaNontiledMatmulLaunch(AD, BD, CD, N);
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
