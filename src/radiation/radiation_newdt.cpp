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
  int nangles_ = nangles;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();
  Real dta = std::numeric_limits<float>::max();

  // setup indicies for Kokkos parallel reduce
  auto &size = pmy_pack->pmb->mb_size;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  auto nh_c_ = nh_c;
  auto nh_f_ = nh_f;
  auto num_neighbors_ = num_neighbors;
  auto ind_neighbors_ = ind_neighbors;
  bool angular_fluxes_ = angular_fluxes;
  auto &coord = pmy_pack->pcoord->coord_data;
  auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &cc_rad_mask_ = pmy_pack->pcoord->cc_rad_mask;

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
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
      ComputeTetrad(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin, e, e_cov, omega);

      for (int n=0; n<nangles_; ++n) {
        for (int nb=0; nb<num_neighbors_.d_view(n); ++nb) {
          Real zeta_f = acos(nh_f_.d_view(n,nb,3));
          Real psi_f  = atan2(nh_f_.d_view(n,nb,2), nh_f_.d_view(n,nb,1));
          Real dzeta  = fabs(acos(nh_c_.d_view(n,3)) -
                             acos(nh_c_.d_view(ind_neighbors_.d_view(n,nb),3)));
          Real dpsi = fabs(atan2(nh_c_.d_view(n,2), nh_c_.d_view(n,1)) -
                           atan2(nh_c_.d_view(ind_neighbors_.d_view(n,nb),2),
                                 nh_c_.d_view(ind_neighbors_.d_view(n,nb),1)));
          Real na1 = 0.0;
          Real na2 = 0.0;
          for (int q=0; q<4; ++q) {
            for (int p=0; p<4; ++p) {
              na1 += (1.0/sin(zeta_f)*nh_f_.d_view(n,nb,q)*nh_f_.d_view(n,nb,p)
                      * (nh_f_.d_view(n,nb,0)*omega[3][q][p]
                      -  nh_f_.d_view(n,nb,3)*omega[0][q][p]));
              na2 += (1.0/SQR(sin(zeta_f))*nh_f_.d_view(n,nb,q)*nh_f_.d_view(n,nb,p)
                      * (nh_f_.d_view(n,nb,2)*omega[1][q][p]
                      -  nh_f_.d_view(n,nb,1)*omega[2][q][p]));
            }
          }
          Real n0 = 0.0;
          for (int q=0; q<4; ++q) {
            n0 += e[q][0]*nh_f_.d_view(n,nb,q);
          }
          if (excise) {
            if (!(cc_rad_mask_(m,k,j,i))) {
              tmp_min_dta = fmin(tmp_min_dta,
                                 fmin(dzeta/fabs(na1/n0), dpsi/fabs(na2/n0)));
            }
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
    // if (dta < dtnew) {
    //   //  const int dtprcsn = std::numeric_limits<Real>::max_digits10 - 1;
    //   const int dtprcsn = 6;
    //   std::cout << "Radiation timestep limited by angular transport." << std::endl;
    //   std::cout << std::scientific << std::setprecision(dtprcsn)
    //             << "Spatial Timestep: " << (pmy_pack->pmesh->cfl_no)*dtnew << std::endl
    //             << "Angular Timestep: " << (pmy_pack->pmesh->cfl_no)*dta << std::endl;
    // }
    dtnew = std::min(dtnew, dta);
  }

  return TaskStatus::complete;
}
} // namespace radiation
