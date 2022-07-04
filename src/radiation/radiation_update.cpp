//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_update.cpp
//  \brief Performs update of Radiation conserved variables (i0) for each stage of
//   explicit SSP RK integrators (e.g. RK1, RK2, RK3). Update uses weighted average and
//   partial time step appropriate to stage.
//  Explicit (not implicit) radiation source terms are included in this update.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "srcterms/srcterms.hpp"
#include "radiation.hpp"

namespace radiation {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::Update
//  \brief Explicit RK update of flux divergence and physical source terms

TaskStatus Radiation::ExpRKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nang1 = prgeo->nangles - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto &mbsize  = pmy_pack->pmb->mb_size;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

  auto i0_ = i0;
  auto i1_ = i1;
  auto flx1 = iflx.x1f;
  auto flx2 = iflx.x2f;
  auto flx3 = iflx.x3f;

  auto &nh_c_ = nh_c;
  auto &tet_c_ = tet_c;
  auto &tetcov_c_ = tetcov_c;

  auto &angular_fluxes_ = angular_fluxes;
  auto &divfa_ = divfa;

  auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &cc_rad_mask_ = pmy_pack->pcoord->cc_rad_mask;

  par_for("r_update",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    // coordinate unit normal components n^0 and n_0
    Real n0 = 0.0; Real n_0 = 0.0;
    for (int d=0; d<4; ++d) {
      n0  += tet_c_   (m,d,0,k,j,i)*nh_c_.d_view(n,d);
      n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);
    }

    // spatial fluxes
    Real divf_s = (flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i))/mbsize.d_view(m).dx1;
    if (multi_d) {
      divf_s += (flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i))/mbsize.d_view(m).dx2;
    }
    if (three_d) {
      divf_s += (flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i))/mbsize.d_view(m).dx3;
    }
    i0_(m,n,k,j,i) = gam0*i0_(m,n,k,j,i)+gam1*i1_(m,n,k,j,i)-beta_dt*divf_s/n0;

    // angular fluxes
    if (angular_fluxes_) {
      i0_(m,n,k,j,i) -= beta_dt*divfa_(m,n,k,j,i)/n0;
    }

    // zero intensity if negative
    i0_(m,n,k,j,i) = n_0*fmax((i0_(m,n,k,j,i)/n_0), 0.0);

    // if excising, handle r_ks <= r_outer
    if (excise) {
      if (cc_rad_mask_(m,k,j,i)) {
        i0_(m,n,k,j,i) = 0.0;
      }
    }
  });

  // add beam source term, if any
  if (psrc->source_terms_enabled) {
    if (psrc->beam)  psrc->AddBeamSource(i0_, beta_dt);
  }

  return TaskStatus::complete;
}

} // namespace radiation
