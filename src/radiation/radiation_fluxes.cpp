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

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::CalcFluxes
//! \brief Compute radiation fluxes

TaskStatus Radiation::CalcFluxes(Driver *pdriver, int stage)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  int nangles_ = nangles;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  const auto recon_method_ = recon_method;
  auto &i0_ = i0;

  auto n1_n_0_ = n1_n_0;
  auto n2_n_0_ = n2_n_0;
  auto n3_n_0_ = n3_n_0;
  auto na_n_0_ = na_n_0;

  //--------------------------------------------------------------------------------------
  // i-direction
  auto flx1 = iflx.x1f;

  par_for("rflux_x1",DevExeSpace(),0,nmb1,0,nangles_-1,ks,ke,js,je,is,ie+1,
    KOKKOS_LAMBDA(int m, int lm, int k, int j, int i)
    {
      // compute x1flux
      Real iil, iir, scr;
      switch (recon_method_)
      {
        case ReconstructionMethod::dc:
          iil = i0_(m,lm,k,j,i-1);
          iir = i0_(m,lm,k,j,i  );
          break;
        case ReconstructionMethod::plm:
          PLM(i0_(m,lm,k,j,i-2), i0_(m,lm,k,j,i-1), i0_(m,lm,k,j,i  ), iil, scr);
          PLM(i0_(m,lm,k,j,i-1), i0_(m,lm,k,j,i  ), i0_(m,lm,k,j,i+1), scr, iir);
          break;
        case ReconstructionMethod::ppm:
          PPM(i0_(m,lm,k,j,i-3), i0_(m,lm,k,j,i-2), i0_(m,lm,k,j,i-1),
              i0_(m,lm,k,j,i  ), i0_(m,lm,k,j,i+1), iil, scr);
          PPM(i0_(m,lm,k,j,i-2), i0_(m,lm,k,j,i-1), i0_(m,lm,k,j,i  ),
              i0_(m,lm,k,j,i+1), i0_(m,lm,k,j,i+2), scr, iir);
          break;
        case ReconstructionMethod::wenoz:
          WENOZ(i0_(m,lm,k,j,i-3), i0_(m,lm,k,j,i-2), i0_(m,lm,k,j,i-1),
                i0_(m,lm,k,j,i  ), i0_(m,lm,k,j,i+1), iil, scr);
          WENOZ(i0_(m,lm,k,j,i-2), i0_(m,lm,k,j,i-1), i0_(m,lm,k,j,i  ),
                i0_(m,lm,k,j,i+1), i0_(m,lm,k,j,i+2), scr, iir);
          break;
        default:
          break;
      }
      flx1(m,lm,k,j,i) = (n1_n_0_(m,lm,k,j,i)*(n1_n_0_(m,lm,k,j,i) < 0.0 ? iil : iir));
    }
  );

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmy_pack->pmesh->multi_d) {
    auto flx2 = iflx.x2f;

    par_for("rflux_x2",DevExeSpace(),0,nmb1,0,nangles_-1,ks,ke,js,je+1,is,ie,
      KOKKOS_LAMBDA(int m, int lm, int k, int j, int i)
      {
        // compute x1flux
        Real iil, iir, scr;
        switch (recon_method_)
        {
          case ReconstructionMethod::dc:
            iil = i0_(m,lm,k,j-1,i);
            iir = i0_(m,lm,k,j  ,i);
            break;
          case ReconstructionMethod::plm:
            PLM(i0_(m,lm,k,j-2,i), i0_(m,lm,k,j-1,i), i0_(m,lm,k,j  ,i), iil, scr);
            PLM(i0_(m,lm,k,j-1,i), i0_(m,lm,k,j  ,i), i0_(m,lm,k,j+1,i), scr, iir);
            break;
          case ReconstructionMethod::ppm:
            PPM(i0_(m,lm,k,j-3,i), i0_(m,lm,k,j-2,i), i0_(m,lm,k,j-1,i),
                i0_(m,lm,k,j  ,i), i0_(m,lm,k,j+1,i), iil, scr);
            PPM(i0_(m,lm,k,j-2,i), i0_(m,lm,k,j-1,i), i0_(m,lm,k,j  ,i),
                i0_(m,lm,k,j+1,i), i0_(m,lm,k,j+2,i), scr, iir);
            break;
          case ReconstructionMethod::wenoz:
            WENOZ(i0_(m,lm,k,j-3,i), i0_(m,lm,k,j-2,i), i0_(m,lm,k,j-1,i),
                  i0_(m,lm,k,j  ,i), i0_(m,lm,k,j+1,i), iil, scr);
            WENOZ(i0_(m,lm,k,j-2,i), i0_(m,lm,k,j-1,i), i0_(m,lm,k,j  ,i),
                  i0_(m,lm,k,j+1,i), i0_(m,lm,k,j+2,i), scr, iir);
            break;
          default:
            break;
        }
        flx2(m,lm,k,j,i) = (n2_n_0_(m,lm,k,j,i)*(n2_n_0_(m,lm,k,j,i) < 0.0 ? iil : iir));
      }
    );
  }

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  if (pmy_pack->pmesh->three_d) {
    auto flx3 = iflx.x3f;

    par_for("rflux_x3",DevExeSpace(),0,nmb1,0,nangles_-1,ks,ke+1,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int lm, int k, int j, int i)
      {
        // compute x1flux
        Real iil, iir, scr;
        switch (recon_method_)
        {
          case ReconstructionMethod::dc:
            iil = i0_(m,lm,k-1,j,i);
            iir = i0_(m,lm,k  ,j,i);
            break;
          case ReconstructionMethod::plm:
            PLM(i0_(m,lm,k-2,j,i), i0_(m,lm,k-1,j,i), i0_(m,lm,k  ,j,i), iil, scr);
            PLM(i0_(m,lm,k-1,j,i), i0_(m,lm,k  ,j,i), i0_(m,lm,k+1,j,i), scr, iir);
            break;
          case ReconstructionMethod::ppm:
            PPM(i0_(m,lm,k-3,j,i), i0_(m,lm,k-2,j,i), i0_(m,lm,k-1,j,i),
                i0_(m,lm,k  ,j,i), i0_(m,lm,k+1,j,i), iil, scr);
            PPM(i0_(m,lm,k-2,j,i), i0_(m,lm,k-1,j,i), i0_(m,lm,k  ,j,i),
                i0_(m,lm,k+1,j,i), i0_(m,lm,k+2,j,i), scr, iir);
            break;
          case ReconstructionMethod::wenoz:
            WENOZ(i0_(m,lm,k-3,j,i), i0_(m,lm,k-2,j,i), i0_(m,lm,k-1,j,i),
                  i0_(m,lm,k  ,j,i), i0_(m,lm,k+1,j,i), iil, scr);
            WENOZ(i0_(m,lm,k-2,j,i), i0_(m,lm,k-1,j,i), i0_(m,lm,k  ,j,i),
                  i0_(m,lm,k+1,j,i), i0_(m,lm,k+2,j,i), scr, iir);
            break;
          default:
            break;
        }
        flx3(m,lm,k,j,i) = (n3_n_0_(m,lm,k,j,i)*(n3_n_0_(m,lm,k,j,i) < 0.0 ? iil : iir));
      }
    );
  }

  //--------------------------------------------------------------------------------------
  // Angular Fluxes
  auto flxa_ = iaflx;
  auto eta_mn_ = eta_mn;
  auto xi_mn_ = xi_mn;
  auto amesh_indices_ = amesh_indices;
  auto nlev_ = nlevels;

  par_for("rflux_a",DevExeSpace(),0,nmb1,0,nangles_-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int lm, int k, int j, int i)
    {
      Real im = i0_(m,lm,k,j,i);
      Real s_mapr_av = 1.0e16;
      Real s_mapr_xi, s_mapr_eta;
      int neighbors[6];
      int num_neighbors = DeviceGetNeighbors(lm, nlev_, amesh_indices_, neighbors);
      for (int nb=0; nb<num_neighbors; ++nb) {
        int nb_1 = nb;
        int nb_2 = (nb+1)%num_neighbors;
        Real imn   = i0_(m,neighbors[nb_1],k,j,i);
        Real imnp1 = i0_(m,neighbors[nb_2],k,j,i);
        Real denom = (1.0/(eta_mn_.d_view(lm,nb_1)*xi_mn_.d_view(lm,nb_2)
                           - xi_mn_.d_view(lm,nb_1)*eta_mn_.d_view(lm,nb_2)));
        Real s_xi = (eta_mn_.d_view(lm,nb_1)*(imnp1-im)
                     - eta_mn_.d_view(lm,nb_2)*(imn-im))*denom;
        Real s_eta = -(xi_mn_.d_view(lm,nb_1)*(imnp1-im)
                       - xi_mn_.d_view(lm,nb_2)*(imn-im))*denom;
        if (sqrt(SQR(s_xi)+SQR(s_eta)) < s_mapr_av) {
          s_mapr_xi = s_xi;
          s_mapr_eta = s_eta;
          s_mapr_av = sqrt(SQR(s_xi)+SQR(s_eta));
        }
      }
      for (int nb=0; nb<num_neighbors; ++nb) {
        Real i_edge = (im + 0.5*s_mapr_xi*xi_mn_.d_view(lm,nb)
                       + 0.5*s_mapr_eta*eta_mn_.d_view(lm,nb));
        flxa_(m,lm,k,j,i,nb) = na_n_0_(m,lm,k,j,i,nb)*i_edge;
      }
    }
  );

  return TaskStatus::complete;
}

} // namespace radiation
