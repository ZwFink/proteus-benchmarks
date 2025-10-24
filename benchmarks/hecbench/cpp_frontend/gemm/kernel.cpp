#include "../../../gpu/gpu_common.h"
extern "C" __global__ void gpuRegSharedTiledMatmulKernel(const double * A,
                                              const double * B,
                                              double * C) {
  // Shared tiles for current K-slice via dynamic shared memory
  // 1 = blockTimeM, 2 = kTile
  constexpr int N = 8192;
  constexpr int BlockTileM = 64;
  constexpr int KTile = 8;
  constexpr int BlockTileN = 64;
  constexpr int RegTileM = 4;
  constexpr int RegTileN = 4;

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
