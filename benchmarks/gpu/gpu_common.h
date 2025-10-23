#ifndef GPU_COMMON_H
#define GPU_COMMON_H

// Default to HIP in IDE/linter contexts when no backend is specified
#if !defined(PROTEUS_ENABLE_HIP) && !defined(PROTEUS_ENABLE_CUDA)
#define PROTEUS_ENABLE_HIP 1
#endif

// Backend selection and GPU convenience macros used across PJ DSL GPU tests
#if PROTEUS_ENABLE_CUDA
#include <cuda_runtime.h>
#define gpuError_t cudaError_t
#define gpuStream_t cudaStream_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuMallocManaged cudaMallocManaged
#define gpuFree cudaFree
#define gpuLaunchKernel cudaLaunchKernel
#define gpuMemcpyFromSymbol cudaMemcpyFromSymbol
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuStreamDestroy cudaStreamDestroy
#define gpuMalloc cudaMalloc
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemset cudaMemset
#elif PROTEUS_ENABLE_HIP
#include <hip/hip_runtime.h>
#define gpuError_t hipError_t
#define gpuStream_t hipStream_t
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuMallocManaged hipMallocManaged
#define gpuFree hipFree
#define gpuLaunchKernel hipLaunchKernel
#define gpuMemcpyFromSymbol hipMemcpyFromSymbol
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuStreamCreate hipStreamCreate
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuStreamDestroy hipStreamDestroy
#define gpuMalloc hipMalloc
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemset hipMemset
#else
#error "Must provide PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA"
#endif

#define gpuErrCheck(CALL)                                                      \
  {                                                                            \
    gpuError_t __GPU_ERROR_CALL_RESULT_CODE__ = CALL;                                                     \
    if (__GPU_ERROR_CALL_RESULT_CODE__ != gpuSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             gpuGetErrorString(__GPU_ERROR_CALL_RESULT_CODE__));                                          \
      abort();                                                                 \
    }                                                                          \
  }

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

#endif // GPU_COMMON_H
