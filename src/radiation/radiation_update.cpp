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
  auto coord = pmy_pack->coord.coord_data;

  int nangles_ = nangles;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real dt_ = pmy_pack->pmesh->dt;
  Real beta_dt = (pdriver->beta[stage-1])*dt_;
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

  auto nh_c_ = nh_c;
  auto norm_to_tet_ = norm_to_tet;

  par_for("s_update",DevExeSpace(),0,nmb1,0,nangles_-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int lm, int k, int j, int i)
    {
      Real divf = (flx1(m,lm,k,j,i+1) - flx1(m,lm,k,j,i))/mbsize.d_view(m).dx1;
      if (multi_d) {
        divf += (flx2(m,lm,k,j+1,i) - flx2(m,lm,k,j,i))/mbsize.d_view(m).dx2;
      }
      if (three_d) {
        divf += (flx3(m,lm,k+1,j,i) - flx3(m,lm,k,j,i))/mbsize.d_view(m).dx3;
      }
      // Update conserved variables
      i0_(m,lm,k,j,i) = (gam0*i0_(m,lm,k,j,i) + gam1*i1_(m,lm,k,j,i)
                         - beta_dt*divf/(nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,0)));
    }
  );

  par_for("a_update",DevExeSpace(),0,nmb1,0,nangles_-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int lm, int k, int j, int i)
    {
      Real divf = 0;
      for (int nb = 0; nb < num_neighbors_.d_view(lm); ++nb){
        divf += arc_lengths_.d_view(lm,nb)*flxa(m,lm,k,j,i,nb);
      }
      // Update conserved variables
      double omega = solid_angle_.d_view(lm);
      i0_(m,lm,k,j,i) -= beta_dt*(divf/omega)/(nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,0));
    }
  );

  // add radiation source terms (if any)
  if (psrc->source_terms_enabled) {
    if (psrc->beam_source)  psrc->AddBeamSource(i0_, beta_dt);
    if ( (psrc->rad_source) && (stage==pdriver->nexp_stages) ) {
      psrc->AddRadiationSourceTerm(i0_, i1_, dt_);
    }
  }

  return TaskStatus::complete;
}

} // namespace radiation
