// RUN: rm -rf .proteus
// RUN: ./for | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <cstdlib>
#include <cstring>
#include <cctype>
#include <proteus/JitInterface.hpp>
#include <iostream>
#include <chrono>
#include <string>

#include "../../../gpu/gpu_common.h"

constexpr int MatmulTileSize = 16;

// Non-tiled (naive) HIP kernel for matrix multiplication: C = A * B
__attribute__((annotate("jit", 4)))
__global__
void hipNontiledMatmulKernelstatic(const double * __restrict__ A,
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

static inline void hipNontiledMatmulLaunch(const double *A,
                                           const double *B,
                                           double *C,
                                           int N) {
  dim3 Block(MatmulTileSize, MatmulTileSize, 1);
  dim3 Grid((N + Block.x - 1) / Block.x,
            (N + Block.y - 1) / Block.y,
            1);
  hipNontiledMatmulKernelstatic<<<Grid, Block>>>(A, B, C, N);
}

// HIP kernel: Register + shared-memory tiled matmul (C = A x B)
constexpr int RegTileM = 4;
constexpr int RegTileN = 4;
constexpr int BlockTileM = 64;
constexpr int BlockTileN = 64;
constexpr int KTile = 8;

#ifdef ENABLE_PROTEUS
#warning "PROTEUS_ENABLE_HIP"
__attribute__((annotate("jit", 4)))
#endif
__global__
void hipRegSharedTiledMatmulKernel(const double * __restrict__ A,
                                   const double * __restrict__ B,
                                   double * __restrict__ C,
                                   int N) {
  #ifdef ENABLE_PROTEUS
  double *smem = proteus::shared_array<double, BlockTileM * KTile + KTile * BlockTileN>(
      BlockTileM * KTile + KTile * BlockTileN);
  #else 
  extern __shared__ double smem[];
  #endif
  double *AsTile = smem;                          // [BlockTileM x KTile]
  double *BsTile = AsTile + (BlockTileM * KTile); // [KTile x BlockTileN]

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int blockRow = by * BlockTileM;
  int blockCol = bx * BlockTileN;

  int row0 = blockRow + ty * RegTileM;
  int col0 = blockCol + tx * RegTileN;

  double Creg[RegTileM * RegTileN];
  #pragma unroll
  for (int i = 0; i < RegTileM * RegTileN; ++i) Creg[i] = 0.0;

  const int threadsX = BlockTileN / RegTileN;
  const int tid = ty * threadsX + tx;

  for (int kBase = 0; kBase < N; kBase += KTile) {
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

    double Areg[RegTileM];
    double Breg[RegTileN];

    #pragma unroll
    for (int kIt = 0; kIt < KTile; ++kIt) {
      #pragma unroll
      for (int i = 0; i < RegTileM; ++i) {
        int r = ty * RegTileM + i;
        int asIdx = r * KTile + kIt;
        Areg[i] = AsTile[asIdx];
      }

      #pragma unroll
      for (int j = 0; j < RegTileN; ++j) {
        int c = tx * RegTileN + j;
        int bsIdx = kIt * BlockTileN + c;
        Breg[j] = BsTile[bsIdx];
      }

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

static inline void hipRegSharedTiledMatmulLaunch(const double *A,
                                                 const double *B,
                                                 double *C,
                                                 int N) {
  dim3 Block(BlockTileN / RegTileN, BlockTileM / RegTileM, 1);
  dim3 Grid(N / BlockTileN, N / BlockTileM, 1);
  #ifdef ENABLE_PROTEUS
  hipRegSharedTiledMatmulKernel<<<Grid, Block>>>(A, B, C, N);
  #else
  size_t SmemBytes = (BlockTileM * KTile + KTile * BlockTileN) * sizeof(double);
  hipRegSharedTiledMatmulKernel<<<Grid, Block, SmemBytes>>>(A, B, C, N);
  #endif
}

static bool verifyG(double *G, int N) {
  double expected = static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
  for (int i = 0; i < N * N; ++i) {
    if (G[i] != expected) return false;
  }
  return true;
}

int main(int argc, char** argv) {
  unsigned int N = 8192;
  int NumTrials = 5;
  bool DoVerify = true;
  std::string KernelType = "gpu_regtiled";
  int blockTileMArg = BlockTileM;
  int blockTileNArg = BlockTileN;
  int kTileArg = KTile;
  int posIdx = 0;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--N") || !std::strcmp(argv[i], "-n")) {
      if (i + 1 < argc) N = static_cast<unsigned int>(std::atoi(argv[++i]));
    } else if (!std::strcmp(argv[i], "--trials") || !std::strcmp(argv[i], "-t")) {
      if (i + 1 < argc) NumTrials = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--kernel")) {
      if (i + 1 < argc) {
        KernelType = argv[++i];
        if (KernelType != "hip" && KernelType != "gpu_regtiled") {
          std::cerr << "Error: Invalid kernel type '" << KernelType
                    << "'. Valid options: hip, gpu_regtiled\n";
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
                << "  KERNEL: hip, gpu_regtiled (default: gpu_regtiled)\n"
                << "  Positional tile sizes are used for the HIP reg-tiled kernel;"
                << " defaults are " << BlockTileM << " " << BlockTileN << " " << KTile << "\n";
      return 0;
    } else {
      if (argv[i] && std::isdigit(static_cast<unsigned char>(argv[i][0]))) {
        int val = std::atoi(argv[i]);
        if (posIdx == 0) blockTileMArg = val;
        else if (posIdx == 1) blockTileNArg = val;
        else if (posIdx == 2) kTileArg = val;
        ++posIdx;
      }
    }
  }

  gpu::warmup();

  std::cout << "3mm HIP-annotation: N=" << N
            << ", trials=" << NumTrials
            << ", verify=" << DoVerify
            << ", kernel=" << KernelType
            << ", tiles=(blkM=" << blockTileMArg
            << ", blkN=" << blockTileNArg
            << ", k=" << kTileArg << ")" << std::endl;

  if (KernelType == "gpu_regtiled") {
    if (blockTileMArg != BlockTileM || blockTileNArg != BlockTileN || kTileArg != KTile) {
      std::cerr << "Warning: HIP reg-tiled kernel uses compile-time tile sizes; ignoring positional values. Using blkM="
                << BlockTileM << ", blkN=" << BlockTileN << ", k=" << KTile << "\n";
    }
  }

  // Host allocations: A,B,C,D inputs; E,F intermediates; G output
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
  size_t Bytes = sizeof(double) * N * N;
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

  // Initial warmup compute E = A * B, F = C * D, G = E * F
  if (KernelType == "gpu_regtiled") {
    hipRegSharedTiledMatmulLaunch(AD, BD, ED, N);
    hipRegSharedTiledMatmulLaunch(CD, DD, FD, N);
    hipRegSharedTiledMatmulLaunch(ED, FD, GD, N);
  } else {
    hipNontiledMatmulLaunch(AD, BD, ED, N);
    hipNontiledMatmulLaunch(CD, DD, FD, N);
    hipNontiledMatmulLaunch(ED, FD, GD, N);
  }
  gpuErrCheck(gpuDeviceSynchronize());

  // Timed trials across the 3 multiplies
  double TotalMs = 0.0;
  auto Start = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < NumTrials; ++t) {
    if (KernelType == "gpu_regtiled") {
      hipRegSharedTiledMatmulLaunch(AD, BD, ED, N);
      hipRegSharedTiledMatmulLaunch(CD, DD, FD, N);
      hipRegSharedTiledMatmulLaunch(ED, FD, GD, N);
    } else {
      hipNontiledMatmulLaunch(AD, BD, ED, N);
      hipNontiledMatmulLaunch(CD, DD, FD, N);
      hipNontiledMatmulLaunch(ED, FD, GD, N);
    }
  }
  gpuErrCheck(gpuDeviceSynchronize());
  auto End = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> Ms = End - Start;
  TotalMs = Ms.count();
  double AvgMs = TotalMs / static_cast<double>(NumTrials);
  std::cerr.setf(std::ios::fixed);
  std::cerr.precision(3);
  std::cerr << "Average over " << NumTrials << " trials (3 GEMMs): " << AvgMs << " ms" << '\n';

  if (DoVerify) {
    gpuErrCheck(gpuMemcpy(GH, GD, Bytes, gpuMemcpyDeviceToHost));
    if (verifyG(GH, N)) std::cout << "Verification passed" << std::endl;
    else                std::cout << "Verification failed" << std::endl;
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
