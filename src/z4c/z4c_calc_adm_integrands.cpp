//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_weyl_scalars.cpp
//  \brief implementation of functions in the Z4c class related to calculation
//  of Weyl scalars

// C++ standard headers
//#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "adm/adm.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/cell_locations.hpp"

namespace z4c {
//----------------------------------------------------------------------------------------
// \!fn void Z4c::Z4cWeyl(MeshBlockPack *pmbp)
// \brief compute the weyl scalars given the adm variables and matter state
//
// This function operates only on the interior points of the MeshBlock
template <int NGHOST>
void Z4c::Z4cAdmIntegrand(MeshBlockPack *pmbp) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;

  auto &z4c = pmbp->pz4c->z4c;
  auto &adm = pmbp->padm->adm;
  auto &adm_ints = pmbp->pz4c->adm_ints;
  auto &u_adm_ints = pmbp->pz4c->u_adm_ints;
  Kokkos::deep_copy(u_adm_ints, 0.);

  par_for("z4c_adm_integrand",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Simplify constants (2 & sqrt 2 factors) featured in re/im[psi4]
    const Real FR4 = 0.25;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    // Scalars
    Real detg = 0.0;         // det(g)
    Real R = 0.0;
    Real dotp1 = 0.0;
    Real dotp2 = 0.0;
    Real K = 0.0;            // trace of extrinsic curvature
    Real KK = 0.0;           // K^a_b K^b_a

    // Vectors
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> uvec;
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> uvec_d;

    for (int a = 0; a < 3; ++a) {
      uvec(a) = 0.0;
      uvec_d(a) = 0.0;
    }

    // Symmetric tensors
    // Rank 2
    // inverse of conf. metric
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> K_ud;        // extrinsic curvature

    // Rank 3
    AthenaScratchTensor<Real, TensorSymm::SYM2,  3, 3> dg_ddd;      // metric 1st drvts

    // Rank 4
    for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b) {
      g_uu(a,b) = 0.0;
      K_ud(a,b) = 0.0;
    }

    Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};
    // -----------------------------------------------------------------------------------
    // derivatives
    //
    // first derivatives of g and K
    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      dg_ddd(c,a,b) = Dx<NGHOST>(c, idx, adm.g_dd, m,a,b,k,j,i);
    }


    // -----------------------------------------------------------------------------------
    // inverse metric
    //
    detg = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                           adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                           adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
    adm::SpatialInv(1.0/detg,
                adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i),
                &g_uu(0,0), &g_uu(0,1), &g_uu(0,2),
                &g_uu(1,1), &g_uu(1,2), &g_uu(2,2));


    // -----------------------------------------------------------------------------------
    // Extrinsic curvature: traces and derivatives
    //

    for(int a = 0; a < 3; ++a) {
      for(int b = 0; b < 3; ++b) {
        for(int c = 0; c < 3; ++c) {
          K_ud(a,b) += g_uu(a,c) * adm.vK_dd(m,c,b,k,j,i);
        }
      }
      K += K_ud(a,a);
    }

    //------------------------------------------------------------------------------------
    //     Construct (approxmiate) normal vector
    uvec(0) = x1v;
    uvec(1) = x2v;
    uvec(2) = x3v;

    // (1) normalize radial vec
    for(int a = 0; a<3; ++a) {
      for(int b = 0; b<3; ++b) {
          dotp1 += adm.g_dd(m,a,b,k,j,i)*uvec(a)*uvec(b);
      }
    }
    for(int a =0; a<3; ++a) {
        uvec(a) = uvec(a)/std::sqrt(dotp1);
    }

    for(int a = 0; a < 3; ++a) {
      for(int b = 0; b < 3; ++b) {
        uvec_d(a) += adm.g_dd(m,a,b,k,j,i)*uvec(b);
      }
    }

    adm_ints.eadm(m,k,j,i) = 0;
    for(int a = 0; a < 3; ++a) {
      for(int b = 0; b < 3; ++b) {
        adm_ints.eadm(m,k,j,i) += (dg_ddd(b,a,b) - dg_ddd(a,b,b)) * uvec(a);
      }
    }
    //adm_ints.eadm(m,k,j,i) /= 16 * M_PI;
    adm_ints.eadm(m,k,j,i) = 1/(16 * M_PI);

    for(int c = 0; c < 3; ++ c) {
      adm_ints.padm(m,c,k,j,i) = 0;
      for(int a = 0; a < 3; ++a) {
        adm_ints.padm(m,c,k,j,i) += K_ud(a,c) * uvec_d(a) - K*uvec_d(c);
      }
      adm_ints.padm(m,c,k,j,i) /= 8 * M_PI;
    }
  });
}

template void Z4c::Z4cAdmIntegrand<2>(MeshBlockPack *pmbp);
template void Z4c::Z4cAdmIntegrand<3>(MeshBlockPack *pmbp);
template void Z4c::Z4cAdmIntegrand<4>(MeshBlockPack *pmbp);
} // namespace z4c

