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
  auto &aindcs = amesh_indcs;
  int &zs = aindcs.zs, &ze = aindcs.ze;
  int &ps = aindcs.ps, &pe = aindcs.pe;

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
  auto flxa1 = ia1flx;
  auto flxa2 = ia2flx;

  auto nmu_ = nmu;
  auto n_mu_ = n_mu;

  auto &angular_fluxes_ = angular_fluxes;
  auto &zeta_length_ = zeta_length;
  auto &psi_length_ = psi_length;
  auto &solid_angle_ = solid_angle;

  auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &cc_rad_mask_ = pmy_pack->pcoord->cc_rad_mask;

  par_for("r_update",DevExeSpace(),0,nmb1,zs,ze,ps,pe,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int z, int p, int k, int j, int i) {
    int n = AngleInd(z,p,false,false,aindcs);

    // coordinate unit normal components n^0 n_0
    Real n0_n_0 = nmu_(m,z,p,k,j,i,0)*n_mu_(m,z,p,k,j,i,0);

    // spatial fluxes
    Real divf_s = (flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i))/mbsize.d_view(m).dx1;
    if (multi_d) {
      divf_s += (flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i))/mbsize.d_view(m).dx2;
    }
    if (three_d) {
      divf_s += (flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i))/mbsize.d_view(m).dx3;
    }
    i0_(m,n,k,j,i) = gam0*i0_(m,n,k,j,i)+gam1*i1_(m,n,k,j,i)-beta_dt*divf_s/n0_n_0;

    // angular fluxes
    if (angular_fluxes_) {
      // Add zeta flux divergence
      int nz   = AngleInd(z  ,p,true,false,aindcs);
      int nzp1 = AngleInd(z+1,p,true,false,aindcs);
      int np   = AngleInd(z,p  ,false,true,aindcs);
      int npp1 = AngleInd(z,p+1,false,true,aindcs);
      bool left_pole = (z==zs); bool right_pole = (z==ze);
      Real zflux_l = (left_pole ) ? 0.0 : -psi_length_.d_view(z  ,p)*flxa1(m,nz,  k,j,i);
      Real zflux_r = (right_pole) ? 0.0 :  psi_length_.d_view(z+1,p)*flxa1(m,nzp1,k,j,i);
      Real divf_a = zflux_l + zflux_r;

      // Add psi flux divergence
      divf_a += (zeta_length_.d_view(z,p+1)*flxa2(m,npp1,k,j,i)
                -zeta_length_.d_view(z,p  )*flxa2(m,np  ,k,j,i));

      divf_a /= solid_angle_.d_view(z,p);
      i0_(m,n,k,j,i) -= beta_dt*divf_a/n0_n_0;
    }

    // zero intensity if negative
    i0_(m,n,k,j,i) = fmax(i0_(m,n,k,j,i), 0.0);

    // if excising, handle r_ks <= r_outer
    if (excise) {
      if (cc_rad_mask_(m,k,j,i)) {
        i0_(m,n,k,j,i) = 0.0;
      }
    }
  });

  // add beam source term, if any
  if (psrc->source_terms_enabled) {
    if (psrc->beam_source)  psrc->AddBeamSource(i0_, beta_dt);
  }

  return TaskStatus::complete;
}

} // namespace radiation
