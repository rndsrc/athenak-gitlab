//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_update.cpp
//  \brief Performs update of Radiation conserved variables (ci0) for each stage of
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

  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nvar = nangles;

  auto ci0_ = ci0;
  auto ci1_ = ci1;
  auto flx1 = ciflx.x1f;
  auto flx2 = ciflx.x2f;
  auto flx3 = ciflx.x3f;

  auto &mbsize = pmy_pack->coord.coord_data.mb_size;

  // hierarchical parallel loop that updates conserved variables to intermediate step
  // using weights and fractional time step appropriate to stages of time-integrator.
  // Important to use vector inner loop for good performance on cpus
  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);

  par_for_outer("s_update",DevExeSpace(),scr_size,scr_level,0,nmb1,0,nvar-1,ks,ke,js,je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int j)
    {
      ScrArray1D<Real> divf(member.team_scratch(scr_level), ncells1);

      // compute dF1/dx1
      par_for_inner(member, is, ie, [&](const int i)
      {
        divf(i) = (flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i))/mbsize.d_view(m).dx1;
      });
      member.team_barrier();

      // Add dF2/dx2
      // Fluxes must be summed in pairs to symmetrize round-off error in each dir
      if (multi_d) {
        par_for_inner(member, is, ie, [&](const int i)
        {
          divf(i) += (flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i))/mbsize.d_view(m).dx2;
        });
        member.team_barrier();
      }

      // Add dF3/dx3
      // Fluxes must be summed in pairs to symmetrize round-off error in each dir
      if (three_d) {
        par_for_inner(member, is, ie, [&](const int i)
        {
          divf(i) += (flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i))/mbsize.d_view(m).dx3;
        });
        member.team_barrier();
      }

      par_for_inner(member, is, ie, [&](const int i)
      {
        ci0_(m,n,k,j,i) = gam0*ci0_(m,n,k,j,i) + gam1*ci1_(m,n,k,j,i) - beta_dt*divf(i);
      });
    }
  );

  auto &aindcs = amesh_indcs;
  int zs = aindcs.zs, ze = aindcs.ze;
  int ps = aindcs.ps, pe = aindcs.pe;

  auto flxa1 = cia1flx;
  auto flxa2 = cia2flx;

  auto zeta_length_ = zeta_length;
  auto psi_length_ = psi_length;
  auto solid_angle_ = solid_angle;

  par_for("a_update",DevExeSpace(),0,nmb1,zs,ze,ps,pe,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int z, int p, int k, int j, int i)
    {
      // Determine poles
      bool left_pole = (z == zs); bool right_pole = (z == ze);

      // Calculate angle lengths and solid angles
      int zp = AngleInd(z,p,false,false,aindcs);
      int zp_lc = AngleInd(z,p,true,false,aindcs);
      int zp_rc = AngleInd(z+1,p,true,false,aindcs);
      int zp_cl = AngleInd(z,p,false,true,aindcs);
      int zp_cr = AngleInd(z,p+1,false,true,aindcs);
      Real zeta_length_m = zeta_length_.d_view(z,p);
      Real zeta_length_p = zeta_length_.d_view(z,p+1);
      Real psi_length_m = psi_length_.d_view(z,p);
      Real psi_length_p = psi_length_.d_view(z+1,p);
      Real omega = solid_angle_.d_view(z,p);

      // Add zeta-divergence
      Real left_flux = left_pole ? 0.0 : -psi_length_m*flxa1(m,zp_lc,k,j,i);
      Real right_flux = right_pole ? 0.0 : psi_length_p*flxa1(m,zp_rc,k,j,i);
      Real divf = left_flux + right_flux;

      // Add psi-divergence
      divf += (zeta_length_p*flxa2(m,zp_cr,k,j,i)-zeta_length_m*flxa2(m,zp_cl,k,j,i));

      // Update conserved variables
      ci0_(m,zp,k,j,i) -= beta_dt*(divf/omega);
    }
  );

  // add radiation source terms if any
  if (psrc->source_terms_enabled) {
    if (psrc->beam_source)  psrc->AddBeamSource(ci0, i0, beta_dt);
  }

  return TaskStatus::complete;
}
} // namespace radiation
