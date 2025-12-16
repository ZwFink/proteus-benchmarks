#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <chrono>
#include <string>
#include <memory>

#include <proteus/CppJitModule.hpp>
#include <proteus/Logger.hpp>
#include <proteus/TimeTracing.hpp>
#include "inja/inja.h"

#include "../../../gpu/gpu_common.h"

#if PROTEUS_ENABLE_CUDA
#include <curand.h>
inline constexpr const char *kDeviceInclude = "#include <cuda_runtime.h>";
using RandGenerator = curandGenerator_t;
using RandStatus = curandStatus_t;
inline constexpr RandStatus RAND_STATUS_SUCCESS = CURAND_STATUS_SUCCESS;
inline constexpr curandRngType_t RAND_RNG_TYPE = CURAND_RNG_PSEUDO_DEFAULT;
inline RandStatus randCreate(RandGenerator *gen) { return curandCreateGenerator(gen, RAND_RNG_TYPE); }
inline RandStatus randSetSeed(RandGenerator gen, unsigned long long seed) {
  return curandSetPseudoRandomGeneratorSeed(gen, seed);
}
inline RandStatus randGenerate(RandGenerator gen, unsigned int *data, size_t count) {
  return curandGenerate(gen, data, count);
}
inline RandStatus randDestroy(RandGenerator gen) { return curandDestroyGenerator(gen); }
#elif PROTEUS_ENABLE_HIP
#include <hiprand/hiprand.h>
inline constexpr const char *kDeviceInclude = "#include <hip/hip_runtime.h>";
using RandGenerator = hiprandGenerator_t;
using RandStatus = hiprandStatus_t;
inline constexpr RandStatus RAND_STATUS_SUCCESS = HIPRAND_STATUS_SUCCESS;
inline constexpr hiprandRngType_t RAND_RNG_TYPE = HIPRAND_RNG_PSEUDO_DEFAULT;
inline RandStatus randCreate(RandGenerator *gen) { return hiprandCreateGenerator(gen, RAND_RNG_TYPE); }
inline RandStatus randSetSeed(RandGenerator gen, unsigned long long seed) {
  return hiprandSetPseudoRandomGeneratorSeed(gen, seed);
}
inline RandStatus randGenerate(RandGenerator gen, unsigned int *data, size_t count) {
  return hiprandGenerate(gen, data, count);
}
inline RandStatus randDestroy(RandGenerator gen) { return hiprandDestroyGenerator(gen); }
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

using namespace proteus;

inline void RandCheck(RandStatus status, const char *expr, const char *file, int line) {
  if (status != RAND_STATUS_SUCCESS) {
    std::fprintf(stderr, "Random generator error (%d) at %s:%d while executing %s\n",
                 static_cast<int>(status), file, line, expr);
    std::abort();
  }
}
#define RAND_CALL(expr) RandCheck((expr), #expr, __FILE__, __LINE__)

#define MAXDISTANCE (200)

// Kernel to map RNG output to [0, MAXDISTANCE] and zero the diagonal
extern "C" __global__ void initRandomMatrix(unsigned int * buf, const unsigned int numNodes)
{
  unsigned int xValue = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int yValue = threadIdx.y + blockIdx.y * blockDim.y;
  if (xValue >= numNodes || yValue >= numNodes) return;
  unsigned int idx = yValue * numNodes + xValue;
  unsigned int v = buf[idx] % (MAXDISTANCE + 1);
  if (xValue == yValue) v = 0u;
  buf[idx] = v;
}

// Kernel template with numNodes baked in as a constexpr via Inja replacement
static constexpr std::string_view StrFloydWarshallKernelTemplate = R"cpp(
{{ include }}
extern "C" __global__ void floydWarshallPass(
    unsigned int * pathDistanceBuffer,
    unsigned int * pathBuffer,
    const unsigned int pass)
{
  constexpr unsigned int numNodes = {{ numNodes }};
  int xValue = threadIdx.x + blockIdx.x * blockDim.x;
  int yValue = threadIdx.y + blockIdx.y * blockDim.y;

  int k = pass;
  int oldWeight = pathDistanceBuffer[yValue * numNodes + xValue];
  int tempWeight = pathDistanceBuffer[yValue * numNodes + k] +
                   pathDistanceBuffer[k * numNodes + xValue];

  if (tempWeight < oldWeight)
  {
    pathDistanceBuffer[yValue * numNodes + xValue] = tempWeight;
    pathBuffer[yValue * numNodes + xValue] = k;
  }
}
)cpp";

static auto getFloydWarshallKernel(unsigned int numNodes)
{
  Timer specializeTimer;
  specializeTimer.reset();
  inja::json data = {
    {"include", std::string{kDeviceInclude}},
    {"numNodes", numNodes}
  };
  auto kernelSource = inja::render(std::string{StrFloydWarshallKernelTemplate}, data);
  const auto specialize_ms = specializeTimer.elapsed();
  Logger::outs("Proteus") << "Specialized Kernel Construction "
                          << specialize_ms << " ms\n";
  auto JitMod = std::make_unique<CppJitModule>(TARGET, kernelSource);
  auto Kernel = JitMod->getKernel<void(unsigned int*, unsigned int*, const unsigned int)>("floydWarshallPass");
  return std::make_pair(std::move(JitMod), Kernel);
}

// Reference CPU implementation for verification
static void floydWarshallCPUReference(unsigned int * pathDistanceMatrix,
                                      unsigned int * pathMatrix,
                                      unsigned int numNodes)
{
  unsigned int width = numNodes;
  for (unsigned int k = 0; k < numNodes; ++k) {
    for (unsigned int y = 0; y < numNodes; ++y) {
      unsigned int yXwidth = y * numNodes;
      for (unsigned int x = 0; x < numNodes; ++x) {
        unsigned int distanceYtoX = pathDistanceMatrix[yXwidth + x];
        unsigned int distanceYtoK = pathDistanceMatrix[yXwidth + k];
        unsigned int distanceKtoX = pathDistanceMatrix[k * width + x];
        unsigned int indirectDistance = distanceYtoK + distanceKtoX;
        if (indirectDistance < distanceYtoX) {
          pathDistanceMatrix[yXwidth + x] = indirectDistance;
          pathMatrix[yXwidth + x] = k;
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::printf("Usage: %s <number of nodes> <iterations> <block size> [verify (0 or 1, default 0)]\n", argv[0]);
    return 1;
  }

  unsigned int numNodes = static_cast<unsigned int>(std::atoi(argv[1]));
  unsigned int numIterations = static_cast<unsigned int>(std::atoi(argv[2]));
  unsigned int blockSize = static_cast<unsigned int>(std::atoi(argv[3]));
  int do_verify = (argc == 5) ? std::atoi(argv[4]) : 0;

  gpu::warmup();

  // Host allocations
  size_t matrixSizeBytes = static_cast<size_t>(numNodes) * static_cast<size_t>(numNodes) * sizeof(unsigned int);
  auto *pathDistanceMatrix = (unsigned int *) std::malloc(matrixSizeBytes);
  auto *pathMatrix = (unsigned int *) std::malloc(matrixSizeBytes);
  assert(pathDistanceMatrix && pathMatrix);

  unsigned int* verificationPathDistanceMatrix = nullptr;
  unsigned int* verificationPathMatrix = nullptr;

  // Initialize path matrix
  if (do_verify) {
  for (unsigned int i = 0; i < numNodes; ++i) {
    for (unsigned int j = 0; j < i; ++j) {
      pathMatrix[i * numNodes + j] = i;
      pathMatrix[j * numNodes + i] = j;
    }
    pathMatrix[i * numNodes + i] = i;
  }
}

  if (do_verify) {
    verificationPathDistanceMatrix = (unsigned int *) std::malloc(matrixSizeBytes);
    verificationPathMatrix = (unsigned int *) std::malloc(matrixSizeBytes);
    assert(verificationPathDistanceMatrix && verificationPathMatrix);
    std::memcpy(verificationPathMatrix, pathMatrix, matrixSizeBytes);
  }

  // Device allocations
  unsigned int *pathDistanceBuffer = nullptr;
  unsigned int *pathBuffer = nullptr;
  gpuErrCheck(gpuMalloc(reinterpret_cast<void **>(&pathDistanceBuffer), matrixSizeBytes));
  gpuErrCheck(gpuMalloc(reinterpret_cast<void **>(&pathBuffer), matrixSizeBytes));

  // Prepare launch configuration (assumes divisibility, clamp if needed)
  unsigned int globalThreadsX = numNodes;
  unsigned int globalThreadsY = numNodes;
  unsigned int localThreadsX = blockSize;
  unsigned int localThreadsY = blockSize;
  if ((unsigned int)(localThreadsX * localThreadsY) > 256u) {
    blockSize = 16u;
    localThreadsX = blockSize;
    localThreadsY = blockSize;
  }
  dim3 grid(globalThreadsX / localThreadsX, globalThreadsY / localThreadsY);
  dim3 block(localThreadsX, localThreadsY);

  // JIT compile kernel specialized for this numNodes
  auto [JitMod, Kernel] = getFloydWarshallKernel(numNodes);

  const size_t totalElements = static_cast<size_t>(numNodes) * static_cast<size_t>(numNodes);
  // Random number generator
  RandGenerator gen{};
  RAND_CALL(randCreate(&gen));
  RAND_CALL(randSetSeed(gen, 1234ULL));

  float total_time_ns = 0.0f;

  for (unsigned int n = 0; n < numIterations; n++) {

    // Generate matrix on device using the RNG backend
    RAND_CALL(randGenerate(gen, pathDistanceBuffer, totalElements));
    // Map to [0, MAXDISTANCE] and zero diagonal
    initRandomMatrix<<<grid, block>>>(pathDistanceBuffer, numNodes);

    if (do_verify && n == numIterations - 1) {
      // Save initial matrix for CPU reference on last iteration
      gpuErrCheck(gpuMemcpy(verificationPathDistanceMatrix, pathDistanceBuffer, matrixSizeBytes,
                            gpuMemcpyDeviceToHost));
    }

    auto start = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < numNodes; ++i) {
      Kernel.launch({static_cast<unsigned int>(grid.x), static_cast<unsigned int>(grid.y), 1},
                    {static_cast<unsigned int>(block.x), static_cast<unsigned int>(block.y), 1},
                    0, nullptr, pathDistanceBuffer, pathBuffer, i);
    }
    gpuErrCheck(gpuDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time_ns += static_cast<float>(time_ns);
  }

  RAND_CALL(randDestroy(gen));

  std::printf("Average kernel execution time %f (s)\n", (total_time_ns * 1e-9f) / static_cast<float>(numIterations));

  if (do_verify) {
    gpuErrCheck(gpuMemcpy(pathDistanceMatrix, pathDistanceBuffer, matrixSizeBytes, gpuMemcpyDeviceToHost));
  }
  gpuErrCheck(gpuFree(pathDistanceBuffer));
  gpuErrCheck(gpuFree(pathBuffer));

  if (do_verify) {
    floydWarshallCPUReference(verificationPathDistanceMatrix, verificationPathMatrix, numNodes);
    if (std::memcmp(pathDistanceMatrix, verificationPathDistanceMatrix, matrixSizeBytes) == 0) {
      std::printf("PASS\n");
    } else {
      std::printf("FAIL\n");
      if (numNodes <= 8) {
        for (unsigned int i = 0; i < numNodes; i++) {
          for (unsigned int j = 0; j < numNodes; j++)
            std::printf("host: %u ", verificationPathDistanceMatrix[i*numNodes+j]);
          std::printf("\n");
        }
        for (unsigned int i = 0; i < numNodes; i++) {
          for (unsigned int j = 0; j < numNodes; j++)
            std::printf("device: %u ", pathDistanceMatrix[i*numNodes+j]);
          std::printf("\n");
        }
      }
    }
  }

  std::free(pathDistanceMatrix);
  std::free(pathMatrix);
  if (do_verify) {
    std::free(verificationPathDistanceMatrix);
    std::free(verificationPathMatrix);
  }
  return 0;
}
