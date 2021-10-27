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
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"
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

  auto &aindcs = amesh_indcs;
  int zs = aindcs.zs, ze = aindcs.ze;
  int ps = aindcs.ps, pe = aindcs.pe;

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
  auto flxa1 = ia1flx;
  auto flxa2 = ia2flx;

  auto &mbsize = pmy_pack->coord.coord_data.mb_size;

  auto zetav_ = zetav;
  auto zetaf_ = zetaf;
  auto psiv_ = psiv;
  auto psif_ = psif;

  auto zeta_length_ = zeta_length;
  auto psi_length_ = psi_length;
  auto solid_angle_ = solid_angle;

  auto n0_n_0_ = n0_n_0;

  par_for("s_update",DevExeSpace(),0,nmb1,zs,ze,ps,pe,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int z, int p, int k, int j, int i)
    {
      int zp = AngleInd(z,p,false,false,aindcs);
      Real divf = (flx1(m,zp,k,j,i+1) - flx1(m,zp,k,j,i))/mbsize.d_view(m).dx1;
      if (multi_d) {
        divf += (flx2(m,zp,k,j+1,i) - flx2(m,zp,k,j,i))/mbsize.d_view(m).dx2;
      }
      if (three_d) {
        divf += (flx3(m,zp,k+1,j,i) - flx3(m,zp,k,j,i))/mbsize.d_view(m).dx3;
      }
      // Update conserved variables
      i0_(m,zp,k,j,i) = (gam0*i0_(m,zp,k,j,i) + gam1*i1_(m,zp,k,j,i)
                         - beta_dt*divf/n0_n_0_(m,z,p,k,j,i));
    }
  );

  par_for("a_update",DevExeSpace(),0,nmb1,zs,ze,ps,pe,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int z, int p, int k, int j, int i)
    {
      // Determine poles
      bool left_pole = (z==zs); bool right_pole = (z==ze);

      // Calculate angle lengths and solid angles
      int zp = AngleInd(z,p,false,false,aindcs);
      int zplc = AngleInd(z,p,true,false,aindcs);
      int zprc = AngleInd(z+1,p,true,false,aindcs);
      int zpcl = AngleInd(z,p,false,true,aindcs);
      int zpcr = AngleInd(z,p+1,false,true,aindcs);
      Real zeta_length_m = zeta_length_.d_view(z,p);
      Real zeta_length_p = zeta_length_.d_view(z,p+1);
      Real psi_length_m = psi_length_.d_view(z,p);
      Real psi_length_p = psi_length_.d_view(z+1,p);
      Real omega = solid_angle_.d_view(z,p);

      // Add zeta-divergence
      Real left_flux = left_pole ? 0.0 : -psi_length_m*flxa1(m,zplc,k,j,i);
      Real right_flux = right_pole ? 0.0 : psi_length_p*flxa1(m,zprc,k,j,i);
      Real divf = left_flux + right_flux;

      // Add psi-divergence
      divf += (zeta_length_p*flxa2(m,zpcr,k,j,i)-zeta_length_m*flxa2(m,zpcl,k,j,i));

      // Update conserved variables
      i0_(m,zp,k,j,i) -= beta_dt*(divf/omega)/n0_n_0_(m,z,p,k,j,i);
    }
  );

  // add radiation source terms if any
  if (psrc->source_terms_enabled) {
    if (psrc->beam_source)  psrc->AddBeamSource(i0, beta_dt);
  }

  return TaskStatus::complete;
}
} // namespace radiation
