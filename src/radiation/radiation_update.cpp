//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_update.cpp
//  \brief Performs update of Radiation conserved variables (i0) for each stage of
//   explicit SSP RK integrators (e.g. RK1, RK2, RK3). Update uses weighted average and
//   partial time step appropriate to stage.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"
#include "hydro/hydro.hpp"
#include "radiation.hpp"

namespace radiation {

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::Update
//  \brief Explicit RK update of flux divergence and physical source terms

TaskStatus Radiation::ExpRKUpdate(Driver *pdriver, int stage)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nangles_ = nangles;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto i0_ = i0;
  auto i1_ = i1;
  auto flx1 = iflx.x1f;
  auto flx2 = iflx.x2f;
  auto flx3 = iflx.x3f;
  auto flxa = iaflx;
  auto &mbsize = pmy_pack->coord.coord_data.mb_size;

  auto nmu_ = nmu;
  auto n_mu_ = n_mu;
  auto num_neighbors_ = num_neighbors;
  auto arc_lengths_ = arc_lengths;
  auto solid_angle_ = solid_angle;
  auto coord = pmy_pack->coord.coord_data;

  par_for("r_update",DevExeSpace(),0,nmb1,0,nangles_-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int lm, int k, int j, int i)
    {
      // coordinate unit normal components n^0 n_0
      Real n0_n_0 = nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,0);

      // spatial fluxes
      Real divf_s = (flx1(m,lm,k,j,i+1) - flx1(m,lm,k,j,i))/mbsize.d_view(m).dx1;
      if (multi_d) {
        divf_s += (flx2(m,lm,k,j+1,i) - flx2(m,lm,k,j,i))/mbsize.d_view(m).dx2;
      }
      if (three_d) {
        divf_s += (flx3(m,lm,k+1,j,i) - flx3(m,lm,k,j,i))/mbsize.d_view(m).dx3;
      }
      i0_(m,lm,k,j,i) = gam0*i0_(m,lm,k,j,i)+gam1*i1_(m,lm,k,j,i)-beta_dt*divf_s/n0_n_0;

      // angular fluxes
      Real divf_a = 0.0;
      for (int nb=0; nb<num_neighbors_.d_view(lm); ++nb) {
        divf_a += arc_lengths_.d_view(lm,nb)*flxa(m,lm,k,j,i,nb)/solid_angle_.d_view(lm);
      }
      i0_(m,lm,k,j,i) -= beta_dt*divf_a/n0_n_0;

      // zero intensity if negative
      i0_(m,lm,k,j,i) = fmax(i0_(m,lm,k,j,i), 0.0);

      // zero intensity inside bh_rmin if excising
      if (coord.bh_excise) {
        Real &x1min = coord.mb_size.d_view(m).x1min;
        Real &x1max = coord.mb_size.d_view(m).x1max;
        int nx1 = coord.mb_indcs.nx1;
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

        Real &x2min = coord.mb_size.d_view(m).x2min;
        Real &x2max = coord.mb_size.d_view(m).x2max;
        int nx2 = coord.mb_indcs.nx2;
        Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

        Real &x3min = coord.mb_size.d_view(m).x3min;
        Real &x3max = coord.mb_size.d_view(m).x3max;
        int nx3 = coord.mb_indcs.nx3;
        Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

        Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
        if (rad < coord.bh_rmin) {
          i0_(m,lm,k,j,i) = 0.0;
        }
      }
    }
  );

  // add beam source term, if any
  if (psrc->source_terms_enabled) {
    if (psrc->beam_source)  psrc->AddBeamSource(i0_, beta_dt);
  }

  return TaskStatus::complete;
}

} // namespace radiation
