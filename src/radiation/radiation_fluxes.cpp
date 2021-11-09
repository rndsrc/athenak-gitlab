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
     const DvceArray5D<Real> nn, struct AMeshIndcs aindcs,
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
  
  int nvars = nangles;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  const auto recon_method_ = recon_method;
  auto &coord = pmy_pack->coord.coord_data;
  auto &i0_ = i0;

  auto n1_n_0_ = n1_n_0;
  auto n2_n_0_ = n2_n_0;
  auto n3_n_0_ = n3_n_0;
  auto na_n_0_ = na_n_0;

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
  // Angular Fluxes
  auto flxa_ = iaflx;
  auto eta_mn_ = eta_mn;
  auto xi_mn_ = xi_mn;

  par_for("rflux_a1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
        Real S_xi, S_eta;
        int nb_1, nb_2;
        
        for (int lm = 0; lm < nangles; ++lm){
        
          Real Im = i0_(m,lm,k,j,i);
          
          Real S_MAPR_av = 1.0e16;
          Real S_MAPR_xi, S_MAPR_eta;
         
          int neighbors[6];
          int num_neighbors = GetNeighbors(lm, neighbors); 
 
          for (int nb = 0; nb < num_neighbors; ++nb){
          
            nb_1 = nb;
            nb_2 = (nb+1)%num_neighbors;
            
            Real Imn   = i0_(m,neighbors[nb_1],k,j,i);
            Real Imnp1 = i0_(m,neighbors[nb_2],k,j,i);
            
            Real denom = 1.0/(eta_mn_.d_view(lm,nb_1)*xi_mn_.d_view(lm,nb_2)-xi_mn_.d_view(lm,nb_1)*eta_mn_.d_view(lm,nb_2));
            S_xi = (eta_mn_.d_view(lm,nb_1)*(Imnp1-Im)-eta_mn_.d_view(lm,nb_2)*(Imn-Im))*denom;
            S_eta = -(xi_mn_.d_view(lm,nb_1)*(Imnp1-Im)-xi_mn_.d_view(lm,nb_2)*(Imn-Im))*denom;
            
            if (std::sqrt(S_xi*S_xi+S_eta*S_eta) < S_MAPR_av) {
              S_MAPR_xi = S_xi;
              S_MAPR_eta = S_eta;
              S_MAPR_av = std::sqrt(S_xi*S_xi+S_eta*S_eta);
            }
          }
              
          for (int nb = 0; nb < num_neighbors; ++nb){
            Real I_edge = Im + 0.5*S_MAPR_xi*xi_mn_.d_view(lm,nb_1) + 0.5*S_MAPR_eta*eta_mn_.d_view(lm,nb_1);
            flxa_(m,lm,k,j,i,nb) = na_n_0_(m,lm,k,j,i,nb) * I_edge;
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
     const DvceArray5D<Real> nn, struct AMeshIndcs aindcs,
     const ScrArray2D<Real> &iil, const ScrArray2D<Real> &iir, DvceArray5D<Real> flx)
{
  par_for_inner(member, il, iu, [&](const int i)
  {
    // TODO FIXME  think about unified layout for angular fluxes
    for (int lm=0; lm < aindcs.nangles; ++lm) {
      flx(m,lm,k,j,i) = (nn(m,lm,k,j,i) * (nn(m,lm,k,j,i) < 0.0 ? iil(lm,i) : iir(lm,i)));
    }
  });

  return;
}

} // namespace radiation
