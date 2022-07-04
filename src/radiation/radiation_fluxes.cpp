//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_fluxes.cpp
//  \brief Calculate 3D fluxes for radiation

#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "radiation.hpp"
#include "reconstruct/dc.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"

namespace radiation {
//----------------------------------------------------------------------------------------
//! \fn  void Radiation::CalculateFluxes
//! \brief Compute radiation fluxes

TaskStatus Radiation::CalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nang1 = prgeo->nangles - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  const auto &recon_method_ = recon_method;

  auto &i0_ = i0;
  auto &na_ = na;
  auto &nh_c_ = nh_c;

  //--------------------------------------------------------------------------------------
  // i-direction

  auto &tet_d1_x1f_ = tet_d1_x1f;
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
      case ReconstructionMethod::ppm4:
        PPM4(i0_(m,n,k,j,i-3), i0_(m,n,k,j,i-2), i0_(m,n,k,j,i-1),
             i0_(m,n,k,j,i  ), i0_(m,n,k,j,i+1), iil, scr);
        PPM4(i0_(m,n,k,j,i-2), i0_(m,n,k,j,i-1), i0_(m,n,k,j,i  ),
             i0_(m,n,k,j,i+1), i0_(m,n,k,j,i+2), scr, iir);
        break;
      case ReconstructionMethod::ppmx:
        PPMX(i0_(m,n,k,j,i-3), i0_(m,n,k,j,i-2), i0_(m,n,k,j,i-1),
             i0_(m,n,k,j,i  ), i0_(m,n,k,j,i+1), iil, scr);
        PPMX(i0_(m,n,k,j,i-2), i0_(m,n,k,j,i-1), i0_(m,n,k,j,i  ),
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
    Real n1 = 0.0;
    for (int d=0; d<4; ++d) { n1 += tet_d1_x1f_(m,d,k,j,i)*nh_c_.d_view(n,d); }
    flx1(m,n,k,j,i) = (n1 * ((n1 > 0.0) ? iil : iir));
  });

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmy_pack->pmesh->multi_d) {
    auto &tet_d2_x2f_ = tet_d2_x2f;
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
        case ReconstructionMethod::ppm4:
          PPM4(i0_(m,n,k,j-3,i), i0_(m,n,k,j-2,i), i0_(m,n,k,j-1,i),
               i0_(m,n,k,j  ,i), i0_(m,n,k,j+1,i), iil, scr);
          PPM4(i0_(m,n,k,j-2,i), i0_(m,n,k,j-1,i), i0_(m,n,k,j  ,i),
               i0_(m,n,k,j+1,i), i0_(m,n,k,j+2,i), scr, iir);
          break;
        case ReconstructionMethod::ppmx:
          PPMX(i0_(m,n,k,j-3,i), i0_(m,n,k,j-2,i), i0_(m,n,k,j-1,i),
               i0_(m,n,k,j  ,i), i0_(m,n,k,j+1,i), iil, scr);
          PPMX(i0_(m,n,k,j-2,i), i0_(m,n,k,j-1,i), i0_(m,n,k,j  ,i),
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
      Real n2 = 0.0;
      for (int d=0; d<4; ++d) { n2 += tet_d2_x2f_(m,d,k,j,i)*nh_c_.d_view(n,d); }
      flx2(m,n,k,j,i) = (n2 * ((n2 > 0.0) ? iil : iir));
    });
  }

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  if (pmy_pack->pmesh->three_d) {
    auto &tet_d3_x3f_ = tet_d3_x3f;
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
        case ReconstructionMethod::ppm4:
          PPM4(i0_(m,n,k-3,j,i), i0_(m,n,k-2,j,i), i0_(m,n,k-1,j,i),
               i0_(m,n,k  ,j,i), i0_(m,n,k+1,j,i), iil, scr);
          PPM4(i0_(m,n,k-2,j,i), i0_(m,n,k-1,j,i), i0_(m,n,k  ,j,i),
               i0_(m,n,k+1,j,i), i0_(m,n,k+2,j,i), scr, iir);
          break;
        case ReconstructionMethod::ppmx:
          PPMX(i0_(m,n,k-3,j,i), i0_(m,n,k-2,j,i), i0_(m,n,k-1,j,i),
               i0_(m,n,k  ,j,i), i0_(m,n,k+1,j,i), iil, scr);
          PPMX(i0_(m,n,k-2,j,i), i0_(m,n,k-1,j,i), i0_(m,n,k  ,j,i),
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
      Real n3 = 0.0;
      for (int d=0; d<4; ++d) { n3 += tet_d3_x3f_(m,d,k,j,i)*nh_c_.d_view(n,d); }
      flx3(m,n,k,j,i) = (n3 * ((n3 > 0.0) ? iil : iir));
    });
  }

  //--------------------------------------------------------------------------------------
  // Angular Fluxes

  if (angular_fluxes) {
    auto num_neighbors_ = prgeo->num_neighbors;
    auto indn_ = prgeo->ind_neighbors;
    auto arc_lengths_ = prgeo->arc_lengths;
    auto solid_angles_ = prgeo->solid_angles;
    auto divfa_ = divfa;
    par_for("rflux_a",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      divfa_(m,n,k,j,i) = 0.0;
      for (int nb=0; nb<num_neighbors_.d_view(n); ++nb) {
        Real flx_edge = (na_(m,n,k,j,i,nb) *
          ((na_(m,n,k,j,i,nb) < 0.0) ? i0_(m,indn_.d_view(n,nb),k,j,i) : i0_(m,n,k,j,i)));
        divfa_(m,n,k,j,i) += (arc_lengths_.d_view(n,nb)*flx_edge/solid_angles_.d_view(n));
      }
    });
  }

  return TaskStatus::complete;
}

} // namespace radiation
