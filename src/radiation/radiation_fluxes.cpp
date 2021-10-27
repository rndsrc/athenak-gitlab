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
// include inlined reconstruction methods (yuck...)
#include "reconstruct/dc.cpp"
#include "reconstruct/plm.cpp"
#include "reconstruct/ppm.cpp"
#include "reconstruct/wenoz.cpp"

#include <Kokkos_Core.hpp>

namespace radiation {

KOKKOS_INLINE_FUNCTION
void SpatialFlux(TeamMember_t const &member,
     const int m, const int k, const int j,  const int il, const int iu,
     const DvceArray6D<Real> nn, struct AMeshIndcs aindcs,
     const ScrArray2D<Real> &iil, const ScrArray2D<Real> &iir, DvceArray5D<Real> flx);

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
  auto &coord = pmy_pack->coord.coord_data;
  auto &i0_ = i0;

  auto n1_n_0_ = n1_n_0;
  auto n2_n_0_ = n2_n_0;
  auto n3_n_0_ = n3_n_0;
  auto na1_n_0_ = na1_n_0;
  auto na2_n_0_ = na2_n_0;

  //--------------------------------------------------------------------------------------
  // i-direction

  size_t scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 2;
  int scr_level = 0;
  auto flx1 = iflx.x1f;

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
        case ReconstructionMethod::plm:
          PiecewiseLinearX1(member, m, k, j, is-1, ie+1, i0_, iil, iir);
          break;
        case ReconstructionMethod::ppm:
          PiecewiseParabolicX1(member, m, k, j, is-1, ie+1, i0_, iil, iir);
          break;
        case ReconstructionMethod::wenoz:
          WENOZX1(member, m, k, j, is-1, ie+1, i0_, iil, iir);
          break;
        default:
          break;
      }
      // Sync all threads in the team so that scratch memory is consistent
      member.team_barrier();

      // compute fluxes over [is,ie+1]
      SpatialFlux(member, m, k, j, is, ie+1,
                  n1_n_0_, aindcs, iil, iir, flx1);
      member.team_barrier();

    }
  );

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmy_pack->pmesh->multi_d) {
    scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 3;
    auto flx2 = iflx.x2f;

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
            case ReconstructionMethod::plm:
              PiecewiseLinearX2(member, m, k, j, is, ie, i0_, iil_jp1, iir);
              break;
            case ReconstructionMethod::ppm:
              PiecewiseParabolicX2(member, m, k, j, is, ie, i0_, iil_jp1, iir);
              break;
            case ReconstructionMethod::wenoz:
              WENOZX2(member, m, k, j, is-1, ie+1, i0_, iil_jp1, iir);
              break;
            default:
              break;
          }
          member.team_barrier();

          // compute fluxes over [js,je+1].  RS returns flux in input iir array
          if (j>(js-1)) {
            SpatialFlux(member, m, k, j, is, ie,
                        n2_n_0_, aindcs, iil, iir, flx2);
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
    auto flx3 = iflx.x3f;

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
            case ReconstructionMethod::plm:
              PiecewiseLinearX3(member, m, k, j, is, ie, i0_, iil_kp1, iir);
              break;
            case ReconstructionMethod::ppm:
              PiecewiseParabolicX3(member, m, k, j, is, ie, i0_, iil_kp1, iir);
              break;
            case ReconstructionMethod::wenoz:
              WENOZX3(member, m, k, j, is-1, ie+1, i0_, iil_kp1, iir);
              break;
            default:
              break;
          }
          member.team_barrier();

          // compute fluxes over [ks,ke+1].  RS returns flux in input iir array
          if (k>(ks-1)) {
            SpatialFlux(member, m, k, j, is, ie,
                        n3_n_0_, aindcs, iil, iir, flx3);
            member.team_barrier();
          }

        } // end loop over k
      }
    );
  }

  //--------------------------------------------------------------------------------------
  // z-direction.
  // Angular Flux
  auto flxa1 = ia1flx;
  auto zetaf_ = zetaf;
  auto zetav_ = zetav;

  par_for("rflux_a1", DevExeSpace(), 0, nmb1, ks, ke, js, je, zs, ze+1, ps, pe,
    KOKKOS_LAMBDA(int m, int k, int j, int z, int p)
    {
      int zpll = AngleInd(z-2, p, false, false, aindcs);
      int zpl  = AngleInd(z-1, p, false, false, aindcs);
      int zpc  = AngleInd(z  , p, true , false, aindcs);
      int zpr  = AngleInd(z  , p, false, false, aindcs);
      int zprr = AngleInd(z+1, p, false, false, aindcs);
      Real dxl = zetaf_.d_view(z) - zetav_.d_view(z-1);
      Real dxr = zetav_.d_view(z) - zetaf_.d_view(z);
      for (int i = is; i <= ie; ++i) {
        Real ill = i0_(m,zpll,k,j,i);
        Real il  = i0_(m,zpl, k,j,i);
        Real ir  = i0_(m,zpr, k,j,i);
        Real irr = i0_(m,zprr,k,j,i);
        Real dql = il - ill;
        Real dqc = ir - il;
        Real dqr = irr - ir;
        Real dq2l = dql*dqc;
        Real dq2r = dqc*dqr;
        Real dqml = (dq2l > 0.0) ? 2.0*dq2l / (dql + dqc) : 0.0;
        Real dqmr = (dq2r > 0.0) ? 2.0*dq2r / (dqc + dqr) : 0.0;
        Real n_tmp = na1_n_0_(m,z,p,k,j,i);
        Real iil = il + dxl*dqml;
        Real iir = ir - dxr*dqmr;
        if (n_tmp < 0.0) {
          flxa1(m,zpc,k,j,i) = n_tmp*iil;
        } else {
          flxa1(m,zpc,k,j,i) = n_tmp*iir;
        }
      }
    }
  );

  //--------------------------------------------------------------------------------------
  // p-direction.
  // Angular Flux
  auto flxa2 = ia2flx;
  auto psiv_ = psiv;
  auto psif_ = psif;

  par_for("rflux_a2", DevExeSpace(), 0, nmb1, ks, ke, js, je, zs, ze, ps, pe+1,
    KOKKOS_LAMBDA(int m, int k, int j, int z, int p)
    {
      int zpll = AngleInd(z, p-2, false, false, aindcs);
      int zpl  = AngleInd(z, p-1, false, false, aindcs);
      int zpc  = AngleInd(z, p  , false, true , aindcs);
      int zpr  = AngleInd(z, p  , false, false, aindcs);
      int zprr = AngleInd(z, p+1, false, false, aindcs);
      Real dxl = psif_.d_view(p) - psiv_.d_view(p-1);
      Real dxr = psiv_.d_view(p) - psif_.d_view(p  );
      for (int i = is; i <= ie; ++i) {
        Real ill = i0_(m,zpll,k,j,i);
        Real il  = i0_(m,zpl, k,j,i);
        Real ir  = i0_(m,zpr, k,j,i);
        Real irr = i0_(m,zprr,k,j,i);
        Real dql = il - ill;
        Real dqc = ir - il;
        Real dqr = irr - ir;
        Real dq2l = dql*dqc;
        Real dq2r = dqc*dqr;
        Real dqml = (dq2l > 0.0) ? 2.0*dq2l / (dql + dqc) : 0.0;
        Real dqmr = (dq2r > 0.0) ? 2.0*dq2r / (dqc + dqr) : 0.0;
        Real n_tmp = na2_n_0_(m,z,p,k,j,i);
        Real iil = il + dxl * dqml;
        Real iir = ir - dxr * dqmr;
        if (n_tmp < 0.0) {
          flxa2(m,zpc,k,j,i) = n_tmp*iil;
        } else {
          flxa2(m,zpc,k,j,i) = n_tmp*iir;
        }
      }
    }
  );

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void SpatialFlux
//! \brief Inline function for computing radiation spatial fluxes

KOKKOS_INLINE_FUNCTION
void SpatialFlux(TeamMember_t const &member,
     const int m, const int k, const int j,  const int il, const int iu,
     const DvceArray6D<Real> nn, struct AMeshIndcs aindcs,
     const ScrArray2D<Real> &iil, const ScrArray2D<Real> &iir, DvceArray5D<Real> flx)
{
  par_for_inner(member, il, iu, [&](const int i)
  {
    for (int z = aindcs.zs-aindcs.ng; z <= aindcs.ze+aindcs.ng; ++z) {
      for (int p = aindcs.ps-aindcs.ng; p <= aindcs.pe+aindcs.ng; ++p) {
        int zp = AngleInd(z, p, false, false, aindcs);
        flx(m,zp,k,j,i) = (nn(m,z,p,k,j,i)
                           * (nn(m,z,p,k,j,i) < 0.0 ? iil(zp,i) : iir(zp,i)));
      }
    }
  });

  return;
}

} // namespace radiation
