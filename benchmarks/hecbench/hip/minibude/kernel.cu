#include <cmath>
#include <cfloat>  // FLT_MAX
#include "bude.h"

#define ZERO    0.0f
#define QUARTER 0.25f
#define HALF    0.5f
#define ONE     1.0f
#define TWO     2.0f
#define FOUR    4.0f
#define CNSTNT 45.0f

// Energy evaluation parameters
#define HBTYPE_F 70
#define HBTYPE_E 69
#define HARDNESS 38.0f
#define NPNPDIST  5.5f
#define NPPDIST   1.0f

#ifdef PROTEUS_ENABLED
__attribute__((annotate("jit", 1, 2, 3, 4)))
#endif
__global__ void fasten_main(
    const size_t posesPerWI,
    const size_t ntypes,
    const size_t nposes,
    const size_t natlig,
    const size_t natpro,
    const float *__restrict protein_x,
    const float *__restrict protein_y,
    const float *__restrict protein_z,
    const int32_t *__restrict protein_type,
    const float *__restrict ligand_x,
    const float *__restrict ligand_y,
    const float *__restrict ligand_z,
    const int32_t *__restrict ligand_type,
    const float *__restrict transforms_0,
    const float *__restrict transforms_1,
    const float *__restrict transforms_2,
    const float *__restrict transforms_3,
    const float *__restrict transforms_4,
    const float *__restrict transforms_5,
    const int32_t *__restrict ff_hbtype,
    const float *__restrict ff_radius,
    const float *__restrict ff_hphb,
    const float *__restrict ff_elsc,
    float *__restrict etotals)
{
  const size_t lid = threadIdx.x;
  const size_t gid = blockIdx.x;
  const size_t lrange = blockDim.x;

  float etot[MAX_PPWI];

  // Positions
  float lpos_x[MAX_PPWI];
  float lpos_y[MAX_PPWI];
  float lpos_z[MAX_PPWI];

  float transform_x[MAX_PPWI * 3];
  float transform_y[MAX_PPWI * 3];
  float transform_z[MAX_PPWI * 3];
  float transform_w[MAX_PPWI * 3];

  size_t ix = gid * lrange * posesPerWI + lid;
  ix = ix < nposes ? ix : nposes - posesPerWI;

  // Compute transformation matrix to private memory
  for (size_t i = 0; i < posesPerWI; i++) {
    size_t index = ix + i * lrange;

    const float sx = sin(transforms_0[index]);
    const float cx = cos(transforms_0[index]);
    const float sy = sin(transforms_1[index]);
    const float cy = cos(transforms_1[index]);
    const float sz = sin(transforms_2[index]);
    const float cz = cos(transforms_2[index]);

    const size_t base = i * 3;

    transform_x[base + 0] = cy * cz;
    transform_y[base + 0] = sx * sy * cz - cx * sz;
    transform_z[base + 0] = cx * sy * cz + sx * sz;
    transform_w[base + 0] = transforms_3[index];

    transform_x[base + 1] = cy * sz;
    transform_y[base + 1] = sx * sy * sz + cx * cz;
    transform_z[base + 1] = cx * sy * sz - sx * cz;
    transform_w[base + 1] = transforms_4[index];

    transform_x[base + 2] = -sy;
    transform_y[base + 2] = sx * cy;
    transform_z[base + 2] = cx * cy;
    transform_w[base + 2] = transforms_5[index];

    etot[i] = ZERO;
  }

  // Loop over ligand atoms
  for (size_t il = 0; il < natlig; ++il) {
    // Load ligand atom data
    const int32_t l_t = ligand_type[il];
    const float l_x = ligand_x[il];
    const float l_y = ligand_y[il];
    const float l_z = ligand_z[il];

    const int32_t l_hbtype = ff_hbtype[l_t];
    const float l_radius = ff_radius[l_t];
    const float l_hphb = ff_hphb[l_t];
    const float l_elsc = ff_elsc[l_t];

    const bool lhphb_ltz = l_hphb < ZERO;
    const bool lhphb_gtz = l_hphb > ZERO;

    for (size_t i = 0; i < posesPerWI; i++) {
      const size_t base = i * 3;
      // Transform ligand atom
      lpos_x[i] = transform_w[base + 0] +
        l_x * transform_x[base + 0] +
        l_y * transform_y[base + 0] +
        l_z * transform_z[base + 0];
      lpos_y[i] = transform_w[base + 1] +
        l_x * transform_x[base + 1] +
        l_y * transform_y[base + 1] +
        l_z * transform_z[base + 1];
      lpos_z[i] = transform_w[base + 2] +
        l_x * transform_x[base + 2] +
        l_y * transform_y[base + 2] +
        l_z * transform_z[base + 2];
    }

    // Loop over protein atoms
    for (size_t ip = 0; ip < natpro; ++ip) {
      // Load protein atom data
      const int32_t p_t = protein_type[ip];
      const float p_x = protein_x[ip];
      const float p_y = protein_y[ip];
      const float p_z = protein_z[ip];

      const int32_t p_hbtype = ff_hbtype[p_t];
      const float p_radius = ff_radius[p_t];
      const float p_hphb = ff_hphb[p_t];
      const float p_elsc = ff_elsc[p_t];

      const float radij = p_radius + l_radius;
      const float r_radij = 1.f / (radij);

      const float elcdst = (p_hbtype == HBTYPE_F && l_hbtype == HBTYPE_F) ? FOUR : TWO;
      const float elcdst1 = (p_hbtype == HBTYPE_F && l_hbtype == HBTYPE_F) ? QUARTER : HALF;
      const bool type_E = ((p_hbtype == HBTYPE_E || l_hbtype == HBTYPE_E));

      const bool phphb_ltz = p_hphb < ZERO;
      const bool phphb_gtz = p_hphb > ZERO;
      const bool phphb_nz = p_hphb != ZERO;
      const float p_hphb_eff = p_hphb * (phphb_ltz && lhphb_gtz ? -ONE : ONE);
      const float l_hphb_eff = l_hphb * (phphb_gtz && lhphb_ltz ? -ONE : ONE);
      const float distdslv = (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) : (lhphb_ltz ? NPPDIST : -FLT_MAX));
      const float r_distdslv = 1.f / (distdslv);

      const float chrg_init = l_elsc * p_elsc;
      const float dslv_init = p_hphb_eff + l_hphb_eff;

      for (size_t i = 0; i < posesPerWI; i++) {
        // Calculate distance between atoms
        const float x = lpos_x[i] - p_x;
        const float y = lpos_y[i] - p_y;
        const float z = lpos_z[i] - p_z;

        const float distij = sqrt(x * x + y * y + z * z);

        // Calculate the sum of the sphere radii
        const float distbb = distij - radij;
        const bool zone1 = (distbb < ZERO);

        // Calculate steric energy
        etot[i] += (ONE - (distij * r_radij)) * (zone1 ? 2 * HARDNESS : ZERO);

        // Calculate formal and dipole charge interactions
        float chrg_e = chrg_init * ((zone1 ? 1 : (ONE - distbb * elcdst1)) * (distbb < elcdst ? 1 : ZERO));
        const float neg_chrg_e = -fabs(chrg_e);
        chrg_e = type_E ? neg_chrg_e : chrg_e;
        etot[i] += chrg_e * CNSTNT;

        // Calculate the two cases for Nonpolar-Polar repulsive interactions
        const float coeff = (ONE - (distbb * r_distdslv));
        float dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? 1 : ZERO);
        dslv_e *= (zone1 ? 1 : coeff);
        etot[i] += dslv_e;
      }
    }
  }

  // Write results
  const size_t td_base = gid * lrange * posesPerWI + lid;

  if (td_base < nposes) {
    for (size_t i = 0; i < posesPerWI; i++) {
      etotals[td_base + i * lrange] = etot[i] * HALF;
    }
  }
}
