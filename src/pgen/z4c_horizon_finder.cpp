//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_one_puncture.cpp
//  \brief Problem generator for finding the horizon for a single puncture placed at the origin of the domain
//

#include <algorithm>
#include <cmath>
#include <sstream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "athena_tensor.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "adm/adm.hpp"
#include "geodesic-grid/strahlkorper.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for Testing Horizon Finder
/*
// Function for inverting 3x3 symmetric matrices
void SpatialInv(Real const detginv,
                Real const gxx, Real const gxy, Real const gxz,
                Real const gyy, Real const gyz, Real const gzz,
                Real * uxx, Real * uxy, Real * uxz,
                Real * uyy, Real * uyz, Real * uzz) {
  *uxx = (-SQR(gyz) + gyy*gzz)*detginv;
  *uxy = (gxz*gyz  - gxy*gzz)*detginv;
  *uyy = (-SQR(gxz) + gxx*gzz)*detginv;
  *uxz = (-gxz*gyy + gxy*gyz)*detginv;
  *uyz = (gxy*gxz  - gxx*gyz)*detginv;
  *uzz = (-SQR(gxy) + gxx*gyy)*detginv;
  return;
}

// Function for determinant of 3x3 symmetric matrices
Real SpatialDet(Real const gxx, Real const gxy, Real const gxz,
                Real const gyy, Real const gyz, Real const gzz) {
  return - SQR(gxz)*gyy + 2*gxy*gxz*gyz
         - SQR(gyz)*gxx
         - SQR(gxy)*gzz +   gxx*gyy*gzz;
}
*/
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "One Puncture test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // One Puncture nitial data 
  pmbp->pz4c->ADMOnePuncture(pmbp, pin);
  pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  std::cout<<"OnePuncture initialized; Starting Horizon Finder"<<std::endl;

  // load in adm variables
  auto &adm = pmbp->padm->adm;
  auto &g_dd = adm.g_dd;
  auto &K_dd = adm.K_dd;

  // Initialize a surface with radius of 2 centered at the origin

  int nlev = 10;
  bool rotate_sphere = true;
  bool fluxes = true;

  Strahlkorper *S = nullptr;
  S = new Strahlkorper(pmbp, nlev, 2,25);
  // Real ctr[3] = {0,0,0};
  // DualArray1D<Real> rad_tmp;
  int nangles = S->nangles;
  auto surface_jacobian = S->surface_jacobian;
  // Kokkos::realloc(rad_tmp,nangles);

  // Container for surface tensors
  DualArray2D<Real> g_dd_surf;
  DualArray2D<Real> K_dd_surf;

  // Athena Tensor structure cannot easily adapt to the Strahlkorper Class. 
  // For now using DualArrays for Tensors and keeping track of the indices.
  // HostArray3D<Real> surf_tensors;
  // AthenaHostTensor<Real,TensorSymm::SYM2, 1, 2> g_dd_surf2;
  // AthenaHostTensor<Real,TensorSymm::SYM2, 1, 2> K_dd_surf2;
  // Kokkos::realloc(surf_tensors,    nmb, (N_Z4c), ncells3, ncells2, ncells1);

  Kokkos::realloc(g_dd_surf,nangles,6); // xx, xy, xz, yy, yz, zz
  Kokkos::realloc(K_dd_surf,nangles,6);

  // Interpolate g_dd and K_dd onto the surface
  g_dd_surf =  S->InterpolateTensorsToSphere(g_dd);
  K_dd_surf =  S->InterpolateTensorsToSphere(K_dd);


  // Evaluate Derivatives of F = r - h(theta,phi) in spherical components
  DualArray2D<Real> dF_d_surf;
  Kokkos::realloc(dF_d_surf,nangles,3);

  DualArray1D<Real> place_holder_for_partial_theta;
  DualArray1D<Real> place_holder_for_partial_phi;
  Kokkos::realloc(place_holder_for_partial_theta,nangles);
  Kokkos::realloc(place_holder_for_partial_phi,nangles);

  place_holder_for_partial_theta = S->ThetaDerivative(S->pointwise_radius);
  place_holder_for_partial_phi = S->PhiDerivative(S->pointwise_radius);

  for(int n=0; n<nangles; ++n) {
    // radial derivatives
    dF_d_surf.h_view(n,0) = 1;
    // theta and phi derivatives
    dF_d_surf.h_view(n,1) = place_holder_for_partial_theta.h_view(n);
    dF_d_surf.h_view(n,2) = place_holder_for_partial_phi.h_view(n);
  }

  // Evaluate Second Derivatives of F in spherical components
  DualArray2D<Real> ddF_dd_surf;
  Kokkos::realloc(ddF_dd_surf,nangles,6); // rr, rt, rp, tt, tp, pp

  DualArray1D<Real> place_holder_for_second_partials;
  Kokkos::realloc(place_holder_for_second_partials,nangles);

  // all second derivatives w.r.t. r vanishes as dr_F = 1
  for(int n=0; n<nangles; ++n) {
    ddF_dd_surf.h_view(n,0) = 0;
    ddF_dd_surf.h_view(n,1) = 0;
    ddF_dd_surf.h_view(n,2) = 0;
  }
  // tt
  place_holder_for_second_partials = S->ThetaDerivative(place_holder_for_partial_theta);
  for(int n=0; n<nangles; ++n) {
    ddF_dd_surf.h_view(n,3) = place_holder_for_second_partials.h_view(n);
  }
  // tp
  place_holder_for_second_partials = S->PhiDerivative(place_holder_for_partial_theta);
  for(int n=0; n<nangles; ++n) {
    ddF_dd_surf.h_view(n,4) = place_holder_for_second_partials.h_view(n);
  }
  // pp
  place_holder_for_second_partials = S->PhiDerivative(place_holder_for_partial_phi);
  for(int n=0; n<nangles; ++n) {
    ddF_dd_surf.h_view(n,5) = place_holder_for_second_partials.h_view(n);
  }

  // Calculating g_uu on the sphere
  DualArray2D<Real> g_uu_surf;
  Kokkos::realloc(g_uu_surf,nangles,6); // xx, xy, xz, yy, yz, zz
  for(int n=0; n<nangles; ++n) {
    Real detg = SpatialDet(g_dd_surf.h_view(n,0), g_dd_surf.h_view(n,1), g_dd_surf.h_view(n,2),
                           g_dd_surf.h_view(n,3), g_dd_surf.h_view(n,4), g_dd_surf.h_view(n,5));
    SpatialInv(1.0/detg,
            g_dd_surf.h_view(n,0), g_dd_surf.h_view(n,1), g_dd_surf.h_view(n,2),
            g_dd_surf.h_view(n,3), g_dd_surf.h_view(n,4), g_dd_surf.h_view(n,5),
            &g_uu_surf.h_view(n,0), &g_uu_surf.h_view(n,1), &g_uu_surf.h_view(n,2),
            &g_uu_surf.h_view(n,3), &g_uu_surf.h_view(n,4), &g_uu_surf.h_view(n,5));
  }

  // Covariant derivatives of F in cartesian basis
  DualArray2D<Real> dF_d_surf_cart;
  Kokkos::realloc(dF_d_surf_cart,nangles,3);
  for(int n=0; n<nangles; ++n) {
    for(int i=0; i<3;++i){
      dF_d_surf_cart.h_view(n,i) = 0;
      for(int u=0; u<3;++u) {
        dF_d_surf_cart.h_view(n,i) += surface_jacobian.h_view(n,u,i)*dF_d_surf.h_view(n,u);
      }
    }
  }

  return;
}
