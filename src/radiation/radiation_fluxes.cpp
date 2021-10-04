//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_fluxes.cpp
//  \brief Calculate spatial and angular fluxes for radiation

#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "radiation.hpp"
#include "eos/eos.hpp"
// include inlined reconstruction methods (yuck...)
#include "reconstruct/dc.cpp"
#include "reconstruct/plm.cpp"
#include "reconstruct/ppm.cpp"
#include "reconstruct/wenoz.cpp"
// include inlined flux calculation
#include "radiation/radiation_rsolver.cpp"

namespace radiation {
//----------------------------------------------------------------------------------------
//! \fn  void Radiation::CalcFluxes
//! \brief Compute radiation fluxes

TaskStatus Radiation::CalcFluxes(Driver *pdriver, int stage)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);

  auto &aindcs = amesh_indcs;
  int zs = aindcs.zs, ze = aindcs.ze;
  int ps = aindcs.ps, pe = aindcs.pe;
  
  int nvars = nangles;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  const auto recon_method_ = recon_method;
  auto &eos = peos->eos_data;
  auto &coord = pmy_pack->coord.coord_data;
  auto &i0_ = i0;

  auto n1_n_mu_ = n1_n_mu;
  auto n2_n_mu_ = n2_n_mu;
  auto n3_n_mu_ = n3_n_mu;
  auto na1_n_0_ = na1_n_0;
  auto na2_n_0_ = na2_n_0;

  //--------------------------------------------------------------------------------------
  // i-direction

  size_t scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 2;
  int scr_level = 0;
  auto flx1 = ciflx.x1f;

  par_for_outer("rflux_x1",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      ScrArray2D<Real> iil(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> iir(member.team_scratch(scr_level), nvars, ncells1);

      // Reconstruct qR[i] and qL[i+1]
      switch (recon_method_)
      {
        case ReconstructionMethod::dc:
          DonorCellX1(member, m, k, j, is-1, ie+1, i0_, iil, iir);
          break;
        // (TODO: @pdmullen) donor cell is easiest, so start with that.  Later add other
        // reconstruction algorithms
        default:
          break;
      }
      // Sync all threads in the team so that scratch memory is consistent
      member.team_barrier();

      // compute fluxes over [is,ie+1]
      SpatialFlux(member, eos, coord, m, k, j, is, ie+1, IVX,
                  n1_n_mu_, nvars, aindcs, iil, iir, flx1);
      member.team_barrier();

    }
  );

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmy_pack->pmesh->multi_d) {
    scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 3;
    auto flx2 = ciflx.x2f;

    par_for_outer("rflux_x2",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k)
      {
        ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, ncells1);
        ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, ncells1);
        ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, ncells1);

        for (int j=js-1; j<=je+1; ++j) {
          // Permute scratch arrays.
          auto iil     = scr1;
          auto iil_jp1 = scr2;
          auto iir     = scr3;
          if ((j%2) == 0) {
            iil     = scr2;
            iil_jp1 = scr1;
          }

          // Reconstruct qR[j] and qL[j+1]
          switch (recon_method_)
          {
            case ReconstructionMethod::dc:
              DonorCellX2(member, m, k, j, is, ie, i0_, iil_jp1, iir);
              break;
            // (TODO: @pdmullen) donor cell is easiest, so start with that.  Later add
            // other reconstruction algorithms
            default:
              break;
          }
          member.team_barrier();

          // compute fluxes over [js,je+1].  RS returns flux in input iir array
          if (j>(js-1)) {
            SpatialFlux(member, eos, coord, m, k, j, is, ie, IVY,
                        n2_n_mu_, nvars, aindcs, iil, iir, flx2);
            member.team_barrier();
          }
  
        } // end of loop over j
      }
    );
  }

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  if (pmy_pack->pmesh->three_d) {
    scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 3;
    auto flx3 = ciflx.x3f;

    par_for_outer("rflux_x3",DevExeSpace(), scr_size, scr_level, 0, nmb1, js, je,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j)
      {
        ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, ncells1);
        ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, ncells1);
        ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, ncells1);

        for (int k=ks-1; k<=ke+1; ++k) {
          // Permute scratch arrays.
          auto iil     = scr1;
          auto iil_kp1 = scr2;
          auto iir     = scr3;
          if ((k%2) == 0) {
            iil     = scr2;
            iil_kp1 = scr1;
          }

          // Reconstruct qR[k] and qL[k+1]
          switch (recon_method_)
          {
            case ReconstructionMethod::dc:
              DonorCellX3(member, m, k, j, is, ie, i0_, iil_kp1, iir);
              break;
            // (TODO: @pdmullen) donor cell is easiest, so start with that.  Later add
            // other reconstruction algorithms
            default:
              break;
          }
          member.team_barrier();

          // compute fluxes over [ks,ke+1].  RS returns flux in input iir array
          if (k>(ks-1)) {
            SpatialFlux(member, eos, coord, m, k, j, is, ie, IVZ,
                        n3_n_mu_, nvars, aindcs, iil, iir, flx3);
            member.team_barrier();
          }

        } // end loop over k
      }
    );
  }

  //--------------------------------------------------------------------------------------
  // z-direction.
  // Angular Flux
  auto flxa1 = cia1flx;

  par_for("rflux_a1", DevExeSpace(), 0, nmb1, ks, ke, js, je, zs, ze+1, ps, pe,
    KOKKOS_LAMBDA(int m, int k, int j, int z, int p)
    {
      // (TODO: @pdmullen) donor cell is easiest, so start with that.  Later add
      // other reconstruction algorithms
      int zp_l = AngleInd(z-1, p, false, false, aindcs);
      int zp_c = AngleInd(z, p, true, false, aindcs);
      int zp_r = AngleInd(z, p, false, false, aindcs);
      for (int i = is; i <= ie; ++i) {
        Real n_tmp = na1_n_0_(m,z,p,k,j,i);
        Real iil = i0_(m,zp_l,k,j,i);
        Real iir = i0_(m,zp_r,k,j,i);
        if (n_tmp < 0.0) {
          flxa1(m,zp_c,k,j,i) = n_tmp*iil;
        } else {
          flxa1(m,zp_c,k,j,i) = n_tmp*iir;
        }
      }
    }
  );

  //--------------------------------------------------------------------------------------
  // p-direction.
  // Angular Flux
  auto flxa2 = cia2flx;

  par_for("rflux_a2", DevExeSpace(), 0, nmb1, ks, ke, js, je, zs, ze, ps, pe+1,
    KOKKOS_LAMBDA(int m, int k, int j, int z, int p)
    {
      // (TODO: @pdmullen) donor cell is easiest, so start with that.  Later add
      // other reconstruction algorithms
      int zp_l = AngleInd(z, p-1, false, false, aindcs);
      int zp_c = AngleInd(z, p, false, true, aindcs);
      int zp_r = AngleInd(z, p, false, false, aindcs);
      for (int i = is; i <= ie; ++i) {
        Real n_tmp = na2_n_0_(m,z,p,k,j,i);
        Real iil = i0_(m,zp_l,k,j,i);
        Real iir = i0_(m,zp_r,k,j,i);
        if (n_tmp < 0.0) {
          flxa2(m,zp_c,k,j,i) = n_tmp*iil;
        } else {
          flxa2(m,zp_c,k,j,i) = n_tmp*iir;
        }
      }
    }
  );

  return TaskStatus::complete;
}

} // namespace radiation
