//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_newdt.cpp
//! \brief function to compute rad timestep across all MeshBlock(s) in a MeshBlockPack

#include <math.h>
#include <float.h>

#include <limits>
#include <iostream>
#include <iomanip>    // std::setprecision()

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "radiation.hpp"
#include "radiation_tetrad.hpp"

namespace radiation {

//----------------------------------------------------------------------------------------
// \!fn void Radiation::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlockPack for radiation problems.
//        Only computed once at beginning of calculation.

TaskStatus Radiation::NewTimeStep(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();
  Real dta = std::numeric_limits<float>::max();

  // setup indicies for Kokkos parallel reduce
  auto &size = pmy_pack->pmb->mb_size;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  int nang1 = nangles - 1;

  bool &angular_fluxes_ = angular_fluxes;
  auto &nh_f_ = nh_f;
  auto &na_ = na;
  auto &tet_c_ = tet_c;
  auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &cc_rad_mask_ = pmy_pack->pcoord->cc_rad_mask;
  auto &num_neighbors_ = num_neighbors;
  auto &indn_ = ind_neighbors;
  int &nlvl = nlevel;
  auto &anorm = amesh_normals;
  auto &apnorm = ameshp_normals;

  // find smallest (dx/c) and (dangle/na) in each direction for radiation problems
  Kokkos::parallel_reduce("RadiationNudt",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx,Real &min_dt1,Real &min_dt2,Real &min_dt3,Real &min_dta) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real tmp_min_dta = (FLT_MAX);
    if (angular_fluxes_) {
      for (int n=0; n<=nang1; ++n) {
        for (int nb=0; nb<num_neighbors_.d_view(n); ++nb) {
          // find position at angle center
          Real x_n, y_n, z_n;
          int ibl0_n = (n / (2*nlvl*nlvl));
          int ibl1_n = (n % (2*nlvl*nlvl)) / (2*nlvl);
          int ibl2_n = (n % (2*nlvl*nlvl)) % (2*nlvl);
          if (ibl0_n == 5) {
            x_n = apnorm.d_view(ibl2_n, 0);
            y_n = apnorm.d_view(ibl2_n, 1);
            z_n = apnorm.d_view(ibl2_n, 2);
          } else {
            x_n = anorm.d_view(ibl0_n,ibl1_n+1,ibl2_n+1,0);
            y_n = anorm.d_view(ibl0_n,ibl1_n+1,ibl2_n+1,1);
            z_n = anorm.d_view(ibl0_n,ibl1_n+1,ibl2_n+1,2);
          }

          // find position at neighbor's angle center
          Real x_nb, y_nb, z_nb;
          int ibl0_nb = (indn_.d_view(n,nb) / (2*nlvl*nlvl));
          int ibl1_nb = (indn_.d_view(n,nb) % (2*nlvl*nlvl)) / (2*nlvl);
          int ibl2_nb = (indn_.d_view(n,nb) % (2*nlvl*nlvl)) % (2*nlvl);
          if (ibl0_nb == 5) {
            x_nb = apnorm.d_view(ibl2_nb, 0);
            y_nb = apnorm.d_view(ibl2_nb, 1);
            z_nb = apnorm.d_view(ibl2_nb, 2);
          } else {
            x_nb = anorm.d_view(ibl0_nb,ibl1_nb+1,ibl2_nb+1,0);
            y_nb = anorm.d_view(ibl0_nb,ibl1_nb+1,ibl2_nb+1,1);
            z_nb = anorm.d_view(ibl0_nb,ibl1_nb+1,ibl2_nb+1,2);
          }

          // compute timestep limitation
          Real n0 = 0.0;
          for (int d=0; d<4; ++d) { n0 += tet_c_(m,d,0,k,j,i)*nh_f_.d_view(n,nb,d); }
          Real angle_dt = fmin(tmp_min_dta,
                               (acos(x_n*x_nb+y_n*y_nb+z_n*z_nb)/
                                fabs(na_(m,n,k,j,i,nb)/n0)));

          // set timestep limitation if not excising this cell
          if (excise) {
            if (!(cc_rad_mask_(m,k,j,i))) { tmp_min_dta = angle_dt; }
          } else {
            tmp_min_dta = angle_dt;
          }
        }
      }
    }

    min_dt1 = fmin((size.d_view(m).dx1), min_dt1);
    min_dt2 = fmin((size.d_view(m).dx2), min_dt2);
    min_dt3 = fmin((size.d_view(m).dx3), min_dt3);
    min_dta = fmin((tmp_min_dta),        min_dta);

  }, Kokkos::Min<Real>(dt1),  Kokkos::Min<Real>(dt2), Kokkos::Min<Real>(dt3),
     Kokkos::Min<Real>(dta));

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }
  if (angular_fluxes_) {
    dtnew = std::min(dtnew, dta);
  }

  return TaskStatus::complete;
}
} // namespace radiation
