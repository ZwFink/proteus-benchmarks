/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <iomanip>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>

#include "../../../cpp_frontend/gpu/gpu_common.h"

#define divceil(n, m) (((n)-1) / (m) + 1)

// Params ---------------------------------------------------------------------
struct Params {

  int         work_group_size;
  const char *file_name;
  int         in_size_i;
  int         in_size_j;
  int         out_size_i;
  int         out_size_j;
  int         num_trials;
  bool        verify;

  Params(int argc, char **argv) {
    work_group_size = 256;
    file_name = "../../data/bezier-surface/control.txt";
    in_size_i = in_size_j = 3;
    out_size_i = out_size_j = 300;
    num_trials = 5;
    verify = false;
    int opt;
    while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:a:f:m:n:v")) >= 0) {
      switch(opt) {
        case 'h':
          usage();
          exit(0);
          break;
        case 'g': work_group_size = atoi(optarg); break;
        case 't': num_trials = atoi(optarg); break;
        case 'f': file_name = optarg; break;
        case 'm': in_size_i = in_size_j = atoi(optarg); break;
        case 'n': out_size_i = out_size_j = atoi(optarg); break;
        case 'v': verify = true; break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
      }
    }
  }

  void usage() {
    fprintf(stderr,
        "\nUsage:  ./main [options]"
        "\n"
        "\nGeneral options:"
        "\n    -h        help"
        "\n    -g <G>    # device work-group size (default=256)"
        "\n    -t <T>    # number of trials (default=5)"
        "\n    -v        verify GPU results against CPU"
        "\n"
        "\n"
        "\nBenchmark-specific options:"
        "\n    -f <F>    name of input file with control points (default=input/control.txt)"
        "\n    -m <N>    input size in both dimensions (default=3)"
        "\n    -n <R>    output resolution in both dimensions (default=300)"
        "\n");
  }
};

// Input Data -----------------------------------------------------------------
void read_input(float *in, const Params &p) {

  // Open input file
  FILE *f = NULL;
  f       = fopen(p.file_name, "r");
  if(f == NULL) {
    puts("Error opening file");
    exit(-1);
  } else {
    printf("Read data from file %s\n", p.file_name);
  } 


  // Store points from input file to array
  int k = 0, ic = 0;
  float vx[10000];
  float vy[10000];
  float vz[10000];
  while(fscanf(f, "%f,%f,%f", &vx[ic], &vy[ic], &vz[ic]) == 3) {
    ic++;
  }
  for(int i = 0; i <= p.in_size_i; i++) {
    for(int j = 0; j <= p.in_size_j; j++) {
      int idx = (i * (p.in_size_j + 1) + j) * 3;
      in[idx + 0] = vx[k];
      in[idx + 1] = vy[k];
      in[idx + 2] = vz[k];
      k = (k + 1) % 16;
    }
  }
}

inline int compare_output(const float *outp, const float *outpCPU,
                          int NI, int NJ, int RESOLUTIONI, int RESOLUTIONJ) {
  float sum_delta2, sum_ref2, L1norm2;
  sum_delta2 = 0;
  sum_ref2   = 0;
  L1norm2    = 0;
  for(int i = 0; i < RESOLUTIONI; i++) {
    for(int j = 0; j < RESOLUTIONJ; j++) {
      int base = (i * RESOLUTIONJ + j) * 3;
      sum_delta2 += fabsf(outp[base + 0] - outpCPU[base + 0]);
      sum_ref2   += fabsf(outpCPU[base + 0]);
      sum_delta2 += fabsf(outp[base + 1] - outpCPU[base + 1]);
      sum_ref2   += fabsf(outpCPU[base + 1]);
      sum_delta2 += fabsf(outp[base + 2] - outpCPU[base + 2]);
      sum_ref2   += fabsf(outpCPU[base + 2]);
    }
  }
  L1norm2 = sum_ref2 == 0.0f ? 0.0f : (sum_delta2 / sum_ref2);
  if(L1norm2 >= 1e-6f){
    printf("Test failed\n");
    return 1;
  }
  return 0;
}

// BezierBlend (http://paulbourke.net/geometry/bezier/)
__host__ __device__
inline float BezierBlend(int k, float mu, int n) {
  int nn, kn, nkn;
  float   blend = 1.0f;
  nn        = n;
  kn        = k;
  nkn       = n - k;
  while(nn >= 1) {
    blend *= static_cast<float>(nn);
    nn--;
    if(kn > 1) {
      blend /= static_cast<float>(kn);
      kn--;
    }
    if(nkn > 1) {
      blend /= static_cast<float>(nkn);
      nkn--;
    }
  }
  if(k > 0)
    blend *= powf(mu, static_cast<float>(k));
  if(n - k > 0)
    blend *= powf(1 - mu, static_cast<float>(n - k));
  return (blend);
}

// Sequential implementation for comparison purposes
void BezierCPU(const float *inp,
               float *outp,
               const int NI, const int NJ, const int RESOLUTIONI, const int RESOLUTIONJ) {
  for(int i = 0; i < RESOLUTIONI; i++) {
    float mui = i / static_cast<float>(RESOLUTIONI - 1);
    for(int j = 0; j < RESOLUTIONJ; j++) {
      float muj = j / static_cast<float>(RESOLUTIONJ - 1);
      float out_x = 0.0f;
      float out_y = 0.0f;
      float out_z = 0.0f;
      for(int ki = 0; ki <= NI; ki++) {
        float bi = BezierBlend(ki, mui, NI);
        for(int kj = 0; kj <= NJ; kj++) {
          float bj = BezierBlend(kj, muj, NJ);
          int idx = (ki * (NJ + 1) + kj) * 3;
          float coeff = bi * bj;
          out_x += inp[idx + 0] * coeff;
          out_y += inp[idx + 1] * coeff;
          out_z += inp[idx + 2] * coeff;
        }
      }
      int out_idx = (i * RESOLUTIONJ + j) * 3;
      outp[out_idx + 0] = out_x;
      outp[out_idx + 1] = out_y;
      outp[out_idx + 2] = out_z;
    }
  }
}

#ifdef PROTEUS_JIT
__attribute__((annotate("jit",  3, 4, 5, 6)))
#endif
__global__
void BezierGPU(const float *inp,
               float *outp,
               const int NI, const int NJ, const int RESOLUTIONI, const int RESOLUTIONJ) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i > RESOLUTIONI) return;

  float mui = i / static_cast<float>(RESOLUTIONI - 1);
  for(int j = 0; j < RESOLUTIONJ; j++) {
    float muj = j / static_cast<float>(RESOLUTIONJ - 1);
    float out_x = 0.0f;
    float out_y = 0.0f;
    float out_z = 0.0f;
    for(int ki = 0; ki <= NI; ki++) {
      float bi = BezierBlend(ki, mui, NI);
      for(int kj = 0; kj <= NJ; kj++) {
        float bj = BezierBlend(kj, muj, NJ);
        int idx = (ki * (NJ + 1) + kj) * 3;
        float coeff = bi * bj;
        out_x += inp[idx + 0] * coeff;
        out_y += inp[idx + 1] * coeff;
        out_z += inp[idx + 2] * coeff;
      }
    }
    int out_idx = (i * RESOLUTIONJ + j) * 3;
    outp[out_idx + 0] = out_x;
    outp[out_idx + 1] = out_y;
    outp[out_idx + 2] = out_z;
  }
}

void run(float *in,
         int in_size_i, int in_size_j, int out_size_i, int out_size_j, const Params &p) {

  size_t out_elems = static_cast<size_t>(out_size_i) * static_cast<size_t>(out_size_j);
  float *gpu_out = (float *)malloc(out_elems * 3 * sizeof(float));
  float *cpu_out = nullptr;

  // CPU run for verification if requested
  if (p.verify) {
    cpu_out = (float *)malloc(out_elems * 3 * sizeof(float));
    auto start = std::chrono::steady_clock::now();
    BezierCPU(in, cpu_out,
              in_size_i, in_size_j, out_size_i, out_size_j);
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "host execution time: " << std::fixed << std::setprecision(4) << time << "ms" << std::endl;
  }

  // Device run - trial loop
  size_t in_size   = static_cast<size_t>(in_size_i + 1) * static_cast<size_t>(in_size_j + 1) * 3 * sizeof(float);
  size_t out_size  = out_elems * 3 * sizeof(float);

  std::vector<double> trial_times;
  dim3 block(p.work_group_size);
  dim3 grid((out_size_i + p.work_group_size - 1) / p.work_group_size);

  for (int trial = 0; trial < p.num_trials; trial++) {
    float *d_in;
    float *d_out;

    auto trial_start = std::chrono::steady_clock::now();

    // Allocate device memory
    gpuErrCheck(hipMalloc((void**)&d_in, in_size));
    gpuErrCheck(hipMalloc((void**)&d_out, out_size));

    // Transfer data to device
    gpuErrCheck(hipMemcpy(d_in, in, in_size, hipMemcpyHostToDevice));

    // Launch kernel
    gpuErrCheck(hipDeviceSynchronize());
    auto kstart = std::chrono::steady_clock::now();

    hipLaunchKernelGGL(BezierGPU, grid, block , 0, 0,
                       d_in, d_out,
                       in_size_i, in_size_j, out_size_i, out_size_j);

    gpuErrCheck(hipDeviceSynchronize());
    auto kend = std::chrono::steady_clock::now();
    auto ktime = std::chrono::duration<double, std::milli>(kend - kstart).count();

    // Transfer data back to host
    gpuErrCheck(hipMemcpy(gpu_out, d_out, out_size, hipMemcpyDeviceToHost));

    auto trial_end = std::chrono::steady_clock::now();
    auto trial_time = std::chrono::duration<double, std::milli>(trial_end - trial_start).count();
    trial_times.push_back(trial_time);

    std::cout << "Trial " << (trial + 1) << " - kernel execution time: " << std::fixed << std::setprecision(4) << ktime << "ms, total time: " << trial_time << "ms" << std::endl;

    // Free device memory
    gpuErrCheck(hipFree(d_in));
    gpuErrCheck(hipFree(d_out));
  }

  // Calculate and report statistics
  if (p.num_trials > 0) {
    std::vector<double> sorted_times = trial_times;
    std::sort(sorted_times.begin(), sorted_times.end());

    double sum = 0.0;
    for (double t : trial_times) {
      sum += t;
    }
    double mean = sum / p.num_trials;
    double median = (p.num_trials % 2 == 0) ?
                    (sorted_times[p.num_trials/2 - 1] + sorted_times[p.num_trials/2]) / 2.0 :
                    sorted_times[p.num_trials/2];

    std::cout << "\nTrial statistics (total time including transfers):" << std::endl;
    std::cout << "  Min: " << std::fixed << std::setprecision(4) << sorted_times.front() << "ms" << std::endl;
    std::cout << "  Max: " << sorted_times.back() << "ms" << std::endl;
    std::cout << "  Mean: " << mean << "ms" << std::endl;
    std::cout << "  Median: " << median << "ms" << std::endl;
  }

  // Verify
  if (p.verify) {
    int status = compare_output(gpu_out, cpu_out,
                               in_size_i, in_size_j, out_size_i, out_size_j);
    printf("%s\n", (status == 0) ? "PASS" : "FAIL");
    free(cpu_out);
  }

  free(gpu_out);
}

int main(int argc, char **argv) {

  const Params p(argc, argv);
  int num_points   = (p.in_size_i + 1) * (p.in_size_j + 1);
  size_t in_size_bytes   = static_cast<size_t>(num_points) * 3 * sizeof(float);
  //int out_size  = p.out_size_i * p.out_size_j * sizeof(XYZ);

  // load data into h_in
  float* in = (float *)malloc(in_size_bytes);
  read_input(in, p);

  // run the app on the cpu and gpu
  run(in, p.in_size_i, p.in_size_j, p.out_size_i, p.out_size_j, p);

  free(in);
  return 0;
}
