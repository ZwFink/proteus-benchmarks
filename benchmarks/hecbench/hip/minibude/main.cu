#include <cmath>
#include <memory>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <functional>
#include <algorithm>
#include "bude.h"
#include "../../../gpu/gpu_common.h"

typedef std::chrono::high_resolution_clock::time_point TimePoint;

struct Params {

  size_t natlig;
  size_t natpro;
  size_t ntypes;
  size_t nposes;

  std::vector<Atom> protein;
  std::vector<Atom> ligand;
  std::vector<FFParams> forcefield;
  std::array<std::vector<float>, 6> poses;

  size_t iterations;

  size_t posesPerWI;
  size_t wgSize;
  std::string deckDir;

  friend std::ostream &operator<<(std::ostream &os, const Params &params) {
    os <<
      "natlig:      " << params.natlig << "\n" <<
      "natpro:      " << params.natpro << "\n" <<
      "ntypes:      " << params.ntypes << "\n" <<
      "nposes:      " << params.nposes << "\n" <<
      "iterations:  " << params.iterations << "\n" <<
      "posesPerWI:  " << params.posesPerWI << "\n" <<
      "wgSize:      " << params.wgSize << "\n";
    return os;
  }
};

__global__ void fasten_main(
    const size_t posesPerWI,
    const size_t ntypes,
    const size_t nposes,
    const size_t natlig,
    const size_t natpro,
    const float *__restrict__ protein_x,
    const float *__restrict__ protein_y,
    const float *__restrict__ protein_z,
    const int32_t *__restrict__ protein_type,
    const float *__restrict__ ligand_x,
    const float *__restrict__ ligand_y,
    const float *__restrict__ ligand_z,
    const int32_t *__restrict__ ligand_type,
    const float *__restrict__ transforms_0,
    const float *__restrict__ transforms_1,
    const float *__restrict__ transforms_2,
    const float *__restrict__ transforms_3,
    const float *__restrict__ transforms_4,
    const float *__restrict__ transforms_5,
    const int32_t *__restrict__ ff_hbtype,
    const float *__restrict__ ff_radius,
    const float *__restrict__ ff_hphb,
    const float *__restrict__ ff_elsc,
    float *__restrict__ etotals);

double elapsedMillis( const TimePoint &start, const TimePoint &end){
  auto elapsedNs = static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
  return elapsedNs * 1e-6;
}

void printTimings(const Params &params, double millis) {

  // Average time per iteration
  double ms = (millis / params.iterations);
  double runtime = ms * 1e-3;

  // Compute FLOP/s
  double ops_per_wg = params.posesPerWI * 27 + params.natlig * (3 + params.posesPerWI * 18 + 
      params.natpro * (11 + params.posesPerWI * 30)) + params.posesPerWI;
  double total_ops = ops_per_wg * ((double) params.nposes / params.posesPerWI);
  double flops = total_ops / runtime;
  double gflops = flops / 1e9;

  double interactions = (double) params.nposes * (double) params.natlig * (double) params.natpro;
  double interactions_per_sec = interactions / runtime;

  // Print stats
  std::cout.precision(3);
  std::cout << std::fixed;
  std::cout << "- Total kernel time:    " << (millis) << " ms\n";
  std::cout << "- Average kernel time:   " << ms << " ms\n";
  std::cout << "- Interactions/s: " << (interactions_per_sec / 1e9) << " billion\n";
  std::cout << "- GFLOP/s:        " << gflops << "\n";
}

template<typename T>
std::vector<T> readNStruct(const std::string &path) {
  std::fstream s(path, std::ios::binary | std::ios::in);
  if (!s.good()) {
    throw std::invalid_argument("Bad file: " + path);
  }
  s.ignore(std::numeric_limits<std::streamsize>::max());
  auto len = s.gcount();
  s.clear();
  s.seekg(0, std::ios::beg);
  std::vector<T> xs(len / sizeof(T));
  s.read(reinterpret_cast<char *>(xs.data()), len);
  s.close();
  return xs;
}

Params loadParameters(const std::vector<std::string> &args) {

  Params params = {};

  // Defaults
  params.iterations = DEFAULT_ITERS;
  params.nposes = DEFAULT_NPOSES;
  params.wgSize = DEFAULT_WGSIZE;
  params.deckDir = DATA_DIR;
  params.posesPerWI = DEFAULT_PPWI;

  const auto readParam = [&args](size_t &current,
      const std::string &arg,
      const std::initializer_list<std::string> &matches,
      const std::function<void(std::string)> &handle) {
    if (matches.size() == 0) return false;
    if (std::find(matches.begin(), matches.end(), arg) != matches.end()) {
      if (current + 1 < args.size()) {
        current++;
        handle(args[current]);
      } else {
        std::cerr << "[";
        for (const auto &m : matches) std::cerr << m;
        std::cerr << "] specified but no value was given" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      return true;
    }
    return false;
  };

  const auto bindInt = [](const std::string &param, size_t &dest, const std::string &name) {
    try {
      auto parsed = std::stol(param);
      if (parsed < 0) {
        std::cerr << "positive integer required for <" << name << ">: `" << parsed << "`" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      dest = parsed;
    } catch (...) {
      std::cerr << "malformed value, integer required for <" << name << ">: `" << param << "`" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  };

  for (size_t i = 0; i < args.size(); ++i) {
    using namespace std::placeholders;
    const auto arg = args[i];
    if (readParam(i, arg, {"--iterations", "-i"}, std::bind(bindInt, _1, std::ref(params.iterations), "iterations"))) continue;
    if (readParam(i, arg, {"--numposes", "-n"}, std::bind(bindInt, _1, std::ref(params.nposes), "numposes"))) continue;
    if (readParam(i, arg, {"--posesperwi", "-p"}, std::bind(bindInt, _1, std::ref(params.posesPerWI), "posesperwi"))) continue;
    if (readParam(i, arg, {"--wgsize", "-w"}, std::bind(bindInt, _1, std::ref(params.wgSize), "wgsize"))) continue;
    if (readParam(i, arg, {"--deck"}, [&](const std::string &param) { params.deckDir = param; })) continue;

    if (arg == "--help" || arg == "-h") {
      std::cout << "\n";
      std::cout << "Usage: ./main [OPTIONS]\n\n"
        << "Options:\n"
        << "  -h  --help               Print this message\n"
        << "  -i  --iterations I       Repeat kernel I times (default: " << DEFAULT_ITERS << ")\n"
        << "  -n  --numposes   N       Compute energies for N poses (default: " << DEFAULT_NPOSES << ")\n"
        << "  -p  --posesperwi PPWI    Compute PPWI poses per work-item (default: " << DEFAULT_PPWI << ")\n"
        << "  -w  --wgsize     WGSIZE  Run with work-group size WGSIZE using nd_range, set to 0 for plain range (default: " << DEFAULT_WGSIZE << ")\n"
        << "      --deck       DECK    Use the DECK directory as input deck (default: " << DATA_DIR << ")"
        << std::endl;
      std::exit(EXIT_SUCCESS);
    }

    std::cout << "Unrecognized argument '" << arg << "' (try '--help')" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (params.posesPerWI == 0 || params.posesPerWI > MAX_PPWI) {
    std::cerr << "posesperwi must be in 1.." << MAX_PPWI << std::endl;
    std::exit(EXIT_FAILURE);
  }

  params.ligand = readNStruct<Atom>(params.deckDir + FILE_LIGAND);
  params.natlig = params.ligand.size();

  params.protein = readNStruct<Atom>(params.deckDir + FILE_PROTEIN);
  params.natpro = params.protein.size();

  params.forcefield = readNStruct<FFParams>(params.deckDir + FILE_FORCEFIELD);
  params.ntypes = params.forcefield.size();

  auto poses = readNStruct<float>(params.deckDir + FILE_POSES);
  if (poses.size() / 6 != params.nposes) {
    throw std::invalid_argument("Bad poses: " + std::to_string(poses.size()));
  }

  for (size_t i = 0; i < 6; ++i) {
    params.poses[i].resize(params.nposes);
    std::copy(
        std::next(poses.cbegin(), i * params.nposes),
        std::next(poses.cbegin(), i * params.nposes + params.nposes),
        params.poses[i].begin());

  }

  return params;
}

std::vector<float> runKernel(Params params) {

  std::vector<float> energies(params.nposes);

  // Build structure-of-arrays for protein and ligand
  std::vector<float> h_protein_x(params.natpro), h_protein_y(params.natpro), h_protein_z(params.natpro);
  std::vector<int32_t> h_protein_type(params.natpro);
  for (size_t i = 0; i < params.natpro; ++i) {
    h_protein_x[i] = params.protein[i].x;
    h_protein_y[i] = params.protein[i].y;
    h_protein_z[i] = params.protein[i].z;
    h_protein_type[i] = params.protein[i].type;
  }
  std::vector<float> h_ligand_x(params.natlig), h_ligand_y(params.natlig), h_ligand_z(params.natlig);
  std::vector<int32_t> h_ligand_type(params.natlig);
  for (size_t i = 0; i < params.natlig; ++i) {
    h_ligand_x[i] = params.ligand[i].x;
    h_ligand_y[i] = params.ligand[i].y;
    h_ligand_z[i] = params.ligand[i].z;
    h_ligand_type[i] = params.ligand[i].type;
  }

  float *d_protein_x, *d_protein_y, *d_protein_z;
  int32_t *d_protein_type;
  hipMalloc((void**)&d_protein_x, params.natpro*sizeof(float));
  hipMalloc((void**)&d_protein_y, params.natpro*sizeof(float));
  hipMalloc((void**)&d_protein_z, params.natpro*sizeof(float));
  hipMalloc((void**)&d_protein_type, params.natpro*sizeof(int32_t));
  hipMemcpy(d_protein_x, h_protein_x.data(), params.natpro*sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_protein_y, h_protein_y.data(), params.natpro*sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_protein_z, h_protein_z.data(), params.natpro*sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_protein_type, h_protein_type.data(), params.natpro*sizeof(int32_t), hipMemcpyHostToDevice);

  float *d_ligand_x, *d_ligand_y, *d_ligand_z;
  int32_t *d_ligand_type;
  hipMalloc((void**)&d_ligand_x, params.natlig*sizeof(float));
  hipMalloc((void**)&d_ligand_y, params.natlig*sizeof(float));
  hipMalloc((void**)&d_ligand_z, params.natlig*sizeof(float));
  hipMalloc((void**)&d_ligand_type, params.natlig*sizeof(int32_t));
  hipMemcpy(d_ligand_x, h_ligand_x.data(), params.natlig*sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_ligand_y, h_ligand_y.data(), params.natlig*sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_ligand_z, h_ligand_z.data(), params.natlig*sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_ligand_type, h_ligand_type.data(), params.natlig*sizeof(int32_t), hipMemcpyHostToDevice);

  float *transforms_0;
  hipMalloc((void**)&transforms_0, params.nposes*sizeof(float));
  hipMemcpy(transforms_0, params.poses[0].data(), params.nposes*sizeof(float), hipMemcpyHostToDevice);

  float *transforms_1;
  hipMalloc((void**)&transforms_1, params.nposes*sizeof(float));
  hipMemcpy(transforms_1, params.poses[1].data(), params.nposes*sizeof(float), hipMemcpyHostToDevice);

  float *transforms_2;
  hipMalloc((void**)&transforms_2, params.nposes*sizeof(float));
  hipMemcpy(transforms_2, params.poses[2].data(), params.nposes*sizeof(float), hipMemcpyHostToDevice);

  float *transforms_3;
  hipMalloc((void**)&transforms_3, params.nposes*sizeof(float));
  hipMemcpy(transforms_3, params.poses[3].data(), params.nposes*sizeof(float), hipMemcpyHostToDevice);

  float *transforms_4;
  hipMalloc((void**)&transforms_4, params.nposes*sizeof(float));
  hipMemcpy(transforms_4, params.poses[4].data(), params.nposes*sizeof(float), hipMemcpyHostToDevice);

  float *transforms_5;
  hipMalloc((void**)&transforms_5, params.nposes*sizeof(float));
  hipMemcpy(transforms_5, params.poses[5].data(), params.nposes*sizeof(float), hipMemcpyHostToDevice);

  // Build structure-of-arrays for forcefield
  std::vector<int32_t> h_ff_hbtype(params.ntypes);
  std::vector<float> h_ff_radius(params.ntypes), h_ff_hphb(params.ntypes), h_ff_elsc(params.ntypes);
  for (size_t i = 0; i < params.ntypes; ++i) {
    h_ff_hbtype[i] = params.forcefield[i].hbtype;
    h_ff_radius[i] = params.forcefield[i].radius;
    h_ff_hphb[i] = params.forcefield[i].hphb;
    h_ff_elsc[i] = params.forcefield[i].elsc;
  }
  int32_t *d_ff_hbtype; float *d_ff_radius; float *d_ff_hphb; float *d_ff_elsc;
  hipMalloc((void**)&d_ff_hbtype, params.ntypes*sizeof(int32_t));
  hipMalloc((void**)&d_ff_radius, params.ntypes*sizeof(float));
  hipMalloc((void**)&d_ff_hphb, params.ntypes*sizeof(float));
  hipMalloc((void**)&d_ff_elsc, params.ntypes*sizeof(float));
  hipMemcpy(d_ff_hbtype, h_ff_hbtype.data(), params.ntypes*sizeof(int32_t), hipMemcpyHostToDevice);
  hipMemcpy(d_ff_radius, h_ff_radius.data(), params.ntypes*sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_ff_hphb, h_ff_hphb.data(), params.ntypes*sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_ff_elsc, h_ff_elsc.data(), params.ntypes*sizeof(float), hipMemcpyHostToDevice);

  float *results;
  hipMalloc((void**)&results, params.nposes*sizeof(float));

  size_t global = ceil((params.nposes) / static_cast<double> (params.posesPerWI));
  global = ceil(static_cast<double> (global) / params.wgSize);

  dim3 grid (global);
  dim3 block (params.wgSize);

  // warmup
  hipLaunchKernelGGL(fasten_main, dim3(grid), dim3(block), 0 , 0, 
      params.posesPerWI,
      params.ntypes,
      params.nposes,
      params.natlig,
      params.natpro,
      d_protein_x,
      d_protein_y,
      d_protein_z,
      d_protein_type,
      d_ligand_x,
      d_ligand_y,
      d_ligand_z,
      d_ligand_type,
      transforms_0,
      transforms_1,
      transforms_2,
      transforms_3,
      transforms_4,
      transforms_5,
      d_ff_hbtype,
      d_ff_radius,
      d_ff_hphb,
      d_ff_elsc,
      results);

  auto kernelStart = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < params.iterations; ++i) {
    hipLaunchKernelGGL(fasten_main, dim3(grid), dim3(block), 0 , 0, 
        params.posesPerWI,
        params.ntypes,
        params.nposes,
        params.natlig,
        params.natpro,
        d_protein_x,
        d_protein_y,
        d_protein_z,
        d_protein_type,
        d_ligand_x,
        d_ligand_y,
        d_ligand_z,
        d_ligand_type,
        transforms_0,
        transforms_1,
        transforms_2,
        transforms_3,
        transforms_4,
        transforms_5,
        d_ff_hbtype,
        d_ff_radius,
        d_ff_hphb,
        d_ff_elsc,
        results);
  }

  hipDeviceSynchronize();

  auto kernelEnd = std::chrono::high_resolution_clock::now();

  hipMemcpy(energies.data(), results, params.nposes*sizeof(float), hipMemcpyDeviceToHost);

  printTimings(params, elapsedMillis(kernelStart, kernelEnd));

  hipFree(d_protein_x);
  hipFree(d_protein_y);
  hipFree(d_protein_z);
  hipFree(d_protein_type);
  hipFree(d_ligand_x);
  hipFree(d_ligand_y);
  hipFree(d_ligand_z);
  hipFree(d_ligand_type);
  hipFree(transforms_0);
  hipFree(transforms_1);
  hipFree(transforms_2);
  hipFree(transforms_3);
  hipFree(transforms_4);
  hipFree(transforms_5);
  hipFree(d_ff_hbtype);
  hipFree(d_ff_radius);
  hipFree(d_ff_hphb);
  hipFree(d_ff_elsc);
  hipFree(results);

  return energies;
}

int main(int argc, char *argv[]) {

  auto args = std::vector<std::string>(argv + 1, argv + argc);
  auto params = loadParameters(args);

  gpu::warmup();

  std::cout << "Poses     : " << params.nposes << std::endl;
  std::cout << "Iterations: " << params.iterations << std::endl;
  std::cout << "Ligands   : " << params.natlig << std::endl;
  std::cout << "Proteins  : " << params.natpro << std::endl;
  std::cout << "Deck      : " << params.deckDir << std::endl;
  std::cout << "WG        : " << params.wgSize << std::endl;
  auto energies = runKernel(params);

#ifdef DUMP
  // Keep the output format consistent with the original version
  FILE *output = fopen("result.out", "w+");

  printf("\nEnergies\n");
  for (size_t i = 0; i < params.nposes; i++) {
    fprintf(output, "%7.2f\n", energies[i]);
    if (i < 16)
      printf("%7.2f\n", energies[i]);
  }
  fclose(output);
#endif

  // Validate energies
  std::ifstream refEnergies(params.deckDir + FILE_REF_ENERGIES);
  size_t nRefPoses = params.nposes;
  if (params.nposes > REF_NPOSES) {
    std::cout << "Only validating the first " << REF_NPOSES << " poses.\n";
    nRefPoses = REF_NPOSES;
  }

  std::string line;
  float maxdiff = 0.0f;
  for (size_t i = 0; i < nRefPoses; i++) {
    if (!std::getline(refEnergies, line)) {
      throw std::logic_error("ran out of ref energies lines to verify");
    }
    float e = std::stof(line);
    if (std::fabs(e) < 1.f && std::fabs(energies[i]) < 1.f) continue;

    float diff = std::fabs(e - energies[i]) / e;
    if (diff > maxdiff) maxdiff = diff;
  }
  std::cout << "Largest difference was " <<
    std::setprecision(3) << (100 * maxdiff) << "%.\n\n"; 
  // Expect numbers to be accurate to 2 decimal places
  refEnergies.close();

  return 0;
}
