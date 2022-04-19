//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_fluxes.cpp
//  \brief Calculate 3D fluxes for radiation

#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "radiation.hpp"
// include inlined reconstruction methods (yuck...)
#include "reconstruct/dc.cpp"             // NOLINT(build/include)
#include "reconstruct/plm.cpp"            // NOLINT(build/include)
#include "reconstruct/ppm.cpp"            // NOLINT(build/include)
#include "reconstruct/wenoz.cpp"          // NOLINT(build/include)

namespace radiation {
//----------------------------------------------------------------------------------------
//! \fn  void Radiation::CalcFluxes
//! \brief Compute radiation fluxes

TaskStatus Radiation::CalcFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  int nang1 = nangles - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const auto recon_method_ = recon_method;
  auto &i0_ = i0;
  auto n1_n_0_ = n1_n_0;
  auto n2_n_0_ = n2_n_0;
  auto n3_n_0_ = n3_n_0;
  auto na_n_0_ = na_n_0;

  auto &coord = pmy_pack->pcoord->coord_data;
  auto &fc_mask_ = pmy_pack->pcoord->fc_mask;

  //--------------------------------------------------------------------------------------
  // i-direction

  auto flx1 = iflx.x1f;
  par_for("rflux_x1",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je,is,ie+1,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    // compute x1flux
    Real iil, iir, scr;
    switch (recon_method_) {
      case ReconstructionMethod::dc:
        iil = i0_(m,n,k,j,i-1);
        iir = i0_(m,n,k,j,i  );
        break;
      case ReconstructionMethod::plm:
        PLM(i0_(m,n,k,j,i-2), i0_(m,n,k,j,i-1), i0_(m,n,k,j,i  ), iil, scr);
        PLM(i0_(m,n,k,j,i-1), i0_(m,n,k,j,i  ), i0_(m,n,k,j,i+1), scr, iir);
        break;
      case ReconstructionMethod::ppm:
        PPM(i0_(m,n,k,j,i-3), i0_(m,n,k,j,i-2), i0_(m,n,k,j,i-1),
            i0_(m,n,k,j,i  ), i0_(m,n,k,j,i+1), iil, scr);
        PPM(i0_(m,n,k,j,i-2), i0_(m,n,k,j,i-1), i0_(m,n,k,j,i  ),
            i0_(m,n,k,j,i+1), i0_(m,n,k,j,i+2), scr, iir);
        break;
      case ReconstructionMethod::wenoz:
        WENOZ(i0_(m,n,k,j,i-3), i0_(m,n,k,j,i-2), i0_(m,n,k,j,i-1),
              i0_(m,n,k,j,i  ), i0_(m,n,k,j,i+1), iil, scr);
        WENOZ(i0_(m,n,k,j,i-2), i0_(m,n,k,j,i-1), i0_(m,n,k,j,i  ),
              i0_(m,n,k,j,i+1), i0_(m,n,k,j,i+2), scr, iir);
        break;
      default:
        break;
    }
    flx1(m,n,k,j,i) = (n1_n_0_(m,n,k,j,i)*(n1_n_0_(m,n,k,j,i) < 0.0 ? iil : iir));
    if (coord.bh_excise) {
      if (fc_mask_.x1f(m,k,j,i)) {
        flx1(m,n,k,j,i) = (n1_n_0_(m,n,k,j,i) *
                          (n1_n_0_(m,n,k,j,i) < 0.0 ? i0_(m,n,k,j,i-1) : i0_(m,n,k,j,i)));
      }
    }
  });

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmy_pack->pmesh->multi_d) {
    auto flx2 = iflx.x2f;
    par_for("rflux_x2",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je+1,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      // compute x2flux
      Real iil, iir, scr;
      switch (recon_method_) {
        case ReconstructionMethod::dc:
          iil = i0_(m,n,k,j-1,i);
          iir = i0_(m,n,k,j  ,i);
          break;
        case ReconstructionMethod::plm:
          PLM(i0_(m,n,k,j-2,i), i0_(m,n,k,j-1,i), i0_(m,n,k,j  ,i), iil, scr);
          PLM(i0_(m,n,k,j-1,i), i0_(m,n,k,j  ,i), i0_(m,n,k,j+1,i), scr, iir);
          break;
        case ReconstructionMethod::ppm:
          PPM(i0_(m,n,k,j-3,i), i0_(m,n,k,j-2,i), i0_(m,n,k,j-1,i),
              i0_(m,n,k,j  ,i), i0_(m,n,k,j+1,i), iil, scr);
          PPM(i0_(m,n,k,j-2,i), i0_(m,n,k,j-1,i), i0_(m,n,k,j  ,i),
              i0_(m,n,k,j+1,i), i0_(m,n,k,j+2,i), scr, iir);
          break;
        case ReconstructionMethod::wenoz:
          WENOZ(i0_(m,n,k,j-3,i), i0_(m,n,k,j-2,i), i0_(m,n,k,j-1,i),
                i0_(m,n,k,j  ,i), i0_(m,n,k,j+1,i), iil, scr);
          WENOZ(i0_(m,n,k,j-2,i), i0_(m,n,k,j-1,i), i0_(m,n,k,j  ,i),
                i0_(m,n,k,j+1,i), i0_(m,n,k,j+2,i), scr, iir);
          break;
        default:
          break;
      }
      flx2(m,n,k,j,i) = (n2_n_0_(m,n,k,j,i)*(n2_n_0_(m,n,k,j,i) < 0.0 ? iil : iir));
      if (coord.bh_excise) {
        if (fc_mask_.x2f(m,k,j,i)) {
          flx2(m,n,k,j,i) = (n2_n_0_(m,n,k,j,i) *
                            (n2_n_0_(m,n,k,j,i)<0.0 ? i0_(m,n,k,j-1,i) : i0_(m,n,k,j,i)));
        }
      }
    });
  }

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  if (pmy_pack->pmesh->three_d) {
    auto flx3 = iflx.x3f;
    par_for("rflux_x3",DevExeSpace(),0,nmb1,0,nang1,ks,ke+1,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      // compute x3flux
      Real iil, iir, scr;
      switch (recon_method_) {
        case ReconstructionMethod::dc:
          iil = i0_(m,n,k-1,j,i);
          iir = i0_(m,n,k  ,j,i);
          break;
        case ReconstructionMethod::plm:
          PLM(i0_(m,n,k-2,j,i), i0_(m,n,k-1,j,i), i0_(m,n,k  ,j,i), iil, scr);
          PLM(i0_(m,n,k-1,j,i), i0_(m,n,k  ,j,i), i0_(m,n,k+1,j,i), scr, iir);
          break;
        case ReconstructionMethod::ppm:
          PPM(i0_(m,n,k-3,j,i), i0_(m,n,k-2,j,i), i0_(m,n,k-1,j,i),
              i0_(m,n,k  ,j,i), i0_(m,n,k+1,j,i), iil, scr);
          PPM(i0_(m,n,k-2,j,i), i0_(m,n,k-1,j,i), i0_(m,n,k  ,j,i),
              i0_(m,n,k+1,j,i), i0_(m,n,k+2,j,i), scr, iir);
          break;
        case ReconstructionMethod::wenoz:
          WENOZ(i0_(m,n,k-3,j,i), i0_(m,n,k-2,j,i), i0_(m,n,k-1,j,i),
                i0_(m,n,k  ,j,i), i0_(m,n,k+1,j,i), iil, scr);
          WENOZ(i0_(m,n,k-2,j,i), i0_(m,n,k-1,j,i), i0_(m,n,k  ,j,i),
                i0_(m,n,k+1,j,i), i0_(m,n,k+2,j,i), scr, iir);
          break;
        default:
          break;
      }
      flx3(m,n,k,j,i) = (n3_n_0_(m,n,k,j,i)*(n3_n_0_(m,n,k,j,i) < 0.0 ? iil : iir));
      if (coord.bh_excise) {
        if (fc_mask_.x3f(m,k,j,i)) {
          flx3(m,n,k,j,i) = (n3_n_0_(m,n,k,j,i) *
                            (n3_n_0_(m,n,k,j,i)<0.0 ? i0_(m,n,k-1,j,i) : i0_(m,n,k,j,i)));
        }
      }
    });
  }

  //--------------------------------------------------------------------------------------
  // Angular Fluxes
  if (angular_fluxes) {
    auto nlev_ = nlevel;
    auto flxa_ = iaflx;
    auto eta_mn_ = eta_mn;
    auto xi_mn_ = xi_mn;
    auto amesh_indices_ = amesh_indices;
    par_for("rflux_a",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      int neighbors[6];
      int num_neighbors = DeviceGetNeighbors(n, nlev_, amesh_indices_, neighbors);
      Real s_av = HUGE_NUMBER, s_xi = HUGE_NUMBER, s_eta = HUGE_NUMBER;
      Real im = i0_(m,n,k,j,i);
      for (int nb=0; nb<num_neighbors; ++nb) {
        int nb_1 = nb;
        int nb_2 = (nb+1)%num_neighbors;
        Real imn = i0_(m,neighbors[nb_1],k,j,i);
        Real imnp1 = i0_(m,neighbors[nb_2],k,j,i);
        Real denom = (1.0/(eta_mn_.d_view(n,nb_1)*xi_mn_.d_view(n,nb_2)
                           - xi_mn_.d_view(n,nb_1)*eta_mn_.d_view(n,nb_2)));
        Real s_xi_tmp = (eta_mn_.d_view(n,nb_1)*(imnp1-im)
                         - eta_mn_.d_view(n,nb_2)*(imn-im))*denom;
        Real s_eta_tmp = -(xi_mn_.d_view(n,nb_1)*(imnp1-im)
                           - xi_mn_.d_view(n,nb_2)*(imn-im))*denom;
        if (sqrt(SQR(s_xi_tmp)+SQR(s_eta_tmp)) < s_av) {
          s_xi = s_xi_tmp;
          s_eta = s_eta_tmp;
          s_av = sqrt(SQR(s_xi)+SQR(s_eta));
        }
      }
      for (int nb=0; nb<num_neighbors; ++nb) {
        Real iedge = (im + 0.5*(s_xi*xi_mn_.d_view(n,nb) + s_eta*eta_mn_.d_view(n,nb)));
        flxa_(m,n,k,j,i,nb) = na_n_0_(m,n,k,j,i,nb)*iedge;
      }
    });
  }

  return TaskStatus::complete;
}

} // namespace radiation