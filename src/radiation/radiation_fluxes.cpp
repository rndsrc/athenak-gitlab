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
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  auto &aindcs = amesh_indcs;
  int &zs = aindcs.zs, &ze = aindcs.ze;
  int &ps = aindcs.ps, &pe = aindcs.pe;

  int nmb1 = pmy_pack->nmb_thispack - 1;

  const auto &recon_method_ = recon_method;

  auto &i0_ = i0;
  auto &n1_n_0_ = n1_n_0;
  auto &n2_n_0_ = n2_n_0;
  auto &n3_n_0_ = n3_n_0;
  auto &na1_n_0_ = na1_n_0;
  auto &na2_n_0_ = na2_n_0;

  auto &coord = pmy_pack->pcoord->coord_data;
  auto &fc_mask_ = pmy_pack->pcoord->fc_mask;

  //--------------------------------------------------------------------------------------
  // i-direction

  auto flx1 = iflx.x1f;
  par_for("rflux_x1",DevExeSpace(),0,nmb1,zs,ze,ps,pe,ks,ke,js,je,is,ie+1,
  KOKKOS_LAMBDA(int m, int z, int p, int k, int j, int i) {
    // compute x1flux
    int n = AngleInd(z,p,false,false,aindcs);
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
    flx1(m,n,k,j,i) = (n1_n_0_(m,z,p,k,j,i)*(n1_n_0_(m,z,p,k,j,i) < 0.0 ? iil : iir));
    if (coord.bh_excise) {
      if (fc_mask_.x1f(m,k,j,i)) {
        flx1(m,n,k,j,i) = (n1_n_0_(m,z,p,k,j,i) *
                          (n1_n_0_(m,z,p,k,j,i) < 0.0 ? i0_(m,n,k,j,i-1) : i0_(m,n,k,j,i)));
      }
    }
  });

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmy_pack->pmesh->multi_d) {
    auto flx2 = iflx.x2f;
    par_for("rflux_x2",DevExeSpace(),0,nmb1,zs,ze,ps,pe,ks,ke,js,je+1,is,ie,
    KOKKOS_LAMBDA(int m, int z, int p, int k, int j, int i) {
      // compute x2flux
      int n = AngleInd(z,p,false,false,aindcs);
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
      flx2(m,n,k,j,i) = (n2_n_0_(m,z,p,k,j,i)*(n2_n_0_(m,z,p,k,j,i) < 0.0 ? iil : iir));
      if (coord.bh_excise) {
        if (fc_mask_.x2f(m,k,j,i)) {
          flx2(m,n,k,j,i) = (n2_n_0_(m,z,p,k,j,i) *
                            (n2_n_0_(m,z,p,k,j,i)<0.0 ? i0_(m,n,k,j-1,i) : i0_(m,n,k,j,i)));
        }
      }
    });
  }

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  if (pmy_pack->pmesh->three_d) {
    auto flx3 = iflx.x3f;
    par_for("rflux_x3",DevExeSpace(),0,nmb1,zs,ze,ps,pe,ks,ke+1,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int z, int p, int k, int j, int i) {
      // compute x3flux
      int n = AngleInd(z,p,false,false,aindcs);
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
      flx3(m,n,k,j,i) = (n3_n_0_(m,z,p,k,j,i)*(n3_n_0_(m,z,p,k,j,i) < 0.0 ? iil : iir));
      if (coord.bh_excise) {
        if (fc_mask_.x3f(m,k,j,i)) {
          flx3(m,n,k,j,i) = (n3_n_0_(m,z,p,k,j,i) *
                            (n3_n_0_(m,z,p,k,j,i)<0.0 ? i0_(m,n,k-1,j,i) : i0_(m,n,k,j,i)));
        }
      }
    });
  }

  //--------------------------------------------------------------------------------------
  // Angular Fluxes
  if (angular_fluxes) {
    auto flxa1 = ia1flx;
    par_for("rflux_a1",DevExeSpace(),0,nmb1,zs,ze+1,ps,pe,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int z, int p, int k, int j, int i) {
      int nzl = AngleInd(z-1,p,false,false,aindcs);
      int nzf = AngleInd(z,p,true,false,aindcs);
      int nzr = AngleInd(z,p,false,false,aindcs);
      flxa1(m,nzf,k,j,i) = (na1_n_0_(m,z,p,k,j,i)*
                           (na1_n_0_(m,z,p,k,j,i)<0.0 ? i0_(m,nzl,k,j,i) : i0_(m,nzr,k,j,i)));
    });

    auto flxa2 = ia2flx;
    par_for("rflux_a1",DevExeSpace(),0,nmb1,zs,ze,ps,pe+1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int z, int p, int k, int j, int i) {
      int npl = AngleInd(z,p-1,false,false,aindcs);
      int npf = AngleInd(z,p,false,true,aindcs);
      int npr = AngleInd(z,p,false,false,aindcs);
      flxa2(m,npf,k,j,i) = (na2_n_0_(m,z,p,k,j,i)*
                           (na2_n_0_(m,z,p,k,j,i)<0.0 ? i0_(m,npl,k,j,i) : i0_(m,npr,k,j,i)));
    });
  }

  return TaskStatus::complete;
}

} // namespace radiation
