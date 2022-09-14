//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file strahlkorper.cpp
//  \brief Initializes a deformed spherical grid to interpolate data onto

#include <cmath>
#include <iostream>
#include <list>
#include "athena_tensor.hpp"

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/coordinates.hpp"
#include "strahlkorper.hpp"
#include "utils/spherical_harm.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Strahlkorper::Strahlkorper(MeshBlockPack *ppack, int nlev, Real rad, int nfilter):
    SphericalGrid(ppack,nlev,rad),
    nfilt(nfilter),
    pointwise_radius("pointwise_radius",1),
    basis_functions("basis_functions",1,1,1), 
    tangent_vectors("tangent_vectors",1,1,1),
    normal_oneforms("normal_oneforms",1,1),
    surface_jacobian("surface_jacobian",1,1,1),
    d_surface_jacobian("d_surface_jacobian",1,1,1,1) {
  
  nlevel = nlev;
  // reallocate and set interpolation coordinates, indices, and weights
  // int &ng = pmy_pack->pmesh->mb_indcs.ng;
  Kokkos::realloc(pointwise_radius,nangles);
  Kokkos::realloc(surface_jacobian,nangles,3,3);
  Kokkos::realloc(d_surface_jacobian,nangles,3,3,3);
  Kokkos::realloc(tangent_vectors,2,nangles,3);
  Kokkos::realloc(basis_functions,3,nfilt,nangles);
  Kokkos::realloc(normal_oneforms,nangles,3);

  InitializeRadius();
  EvaluateSphericalHarm();
  EvaluateSurfaceJacobian();
  EvaluateSurfaceJacobianDerivative();
  return;
}

//----------------------------------------------------------------------------------------
//! \brief Strahlkorper destructor

Strahlkorper::~Strahlkorper() {
}

void Strahlkorper::InitializeRadius() {
  for (int n=0; n<nangles; ++n) {
    Real &theta = polar_pos.h_view(n,0);
    Real &phi = polar_pos.h_view(n,1);
    pointwise_radius.h_view(n) = radius;
  }
  pointwise_radius.template modify<HostMemSpace>();
  pointwise_radius.template sync<DevExeSpace>();
}

void Strahlkorper::SetPointwiseRadius(DualArray1D<Real> rad_tmp, Real ctr[3]) {
  for (int n=0; n<nangles; ++n) {
    Real &theta = polar_pos.h_view(n,0);
    Real &phi = polar_pos.h_view(n,1);
    interp_coord.h_view(n,0) = rad_tmp.h_view(n)*cos(phi)*sin(theta) + ctr[0];
    interp_coord.h_view(n,1) = rad_tmp.h_view(n)*sin(phi)*sin(theta) + ctr[1];
    interp_coord.h_view(n,2) = rad_tmp.h_view(n)*cos(theta) + ctr[2];
    pointwise_radius.h_view(n) = rad_tmp.h_view(n);
  }
  // sync dual arrays
  interp_coord.template modify<HostMemSpace>();
  interp_coord.template sync<DevExeSpace>();
  pointwise_radius.template modify<HostMemSpace>();
  pointwise_radius.template sync<DevExeSpace>();

  // reset interpolation indices and weights
  SetInterpolationIndices();
  SetInterpolationWeights();
  EvaluateSurfaceJacobian();
  // EvaluateTangentVectors();
  return;
}

void Strahlkorper::EvaluateSphericalHarm() {
  for (int a=0; a<nfilt; a++) {
    int l = (int) sqrt(a);
    int m = (int) (a-l*l-l);
    for (int n=0; n<nangles; ++n) {
      Real &theta = polar_pos.h_view(n,0);
      Real &phi = polar_pos.h_view(n,1);
      basis_functions.h_view(0,a,n) = RealSphericalHarm(l, m, theta, phi);
      basis_functions.h_view(1,a,n) = RealSphericalHarm_dtheta(l, m, theta, phi);
      basis_functions.h_view(2,a,n) = RealSphericalHarm_dphi(l, m, theta, phi);
    }
  }
  // sync dual arrays
  basis_functions.template modify<HostMemSpace>();
  basis_functions.template sync<DevExeSpace>();
  return;
}

// integral assuming unit sphere, for spectral differentiation
Real Strahlkorper::Integrate(DualArray1D<Real> integrand) {
  Real value = 0.;
  for (int n=0; n<nangles; ++n) {
    value += integrand.h_view(n)*solid_angles.h_view(n);
  }
  return value;
}

// calculate spectral representation, maybe change into inline function later
DualArray1D<Real> Strahlkorper::SpatialToSpectral(DualArray1D<Real> scalar_function) {
  DualArray1D<Real> spectral;
  DualArray1D<Real> integrand;
  Kokkos::realloc(spectral,nfilt);
  Kokkos::realloc(integrand,nangles);

  for (int i=0; i<nfilt; ++i) {
    for (int n=0; n<nangles; ++n) {
      integrand.h_view(n) = scalar_function.h_view(n)*basis_functions.h_view(0,i,n);
    }
    spectral.h_view(i) = Integrate(integrand);
  }
  return spectral;
}

DualArray1D<Real> Strahlkorper::SpectralToSpatial(DualArray1D<Real> scalar_spectrum) {
  DualArray1D<Real> scalar_function;
  Kokkos::realloc(scalar_function,nangles);
  for (int i=0; i<nfilt;++i) {
    for (int n=0; n<nangles; ++n) {
      scalar_function.h_view(n) += scalar_spectrum.h_view(i)*basis_functions.h_view(0,i,n);
    }
  }
  return scalar_function;
}

DualArray1D<Real> Strahlkorper::ThetaDerivative(DualArray1D<Real> scalar_function) {
  // first find spectral representation
  DualArray1D<Real> spectral;
  Kokkos::realloc(spectral,nfilt);
  spectral = SpatialToSpectral(scalar_function);
  // calculate theta derivative
  DualArray1D<Real> scalar_function_dtheta;
  Kokkos::realloc(scalar_function_dtheta,nangles);
  for (int i=0; i<nfilt; ++i) {
    for (int n=0; n<nangles; ++n) {
      scalar_function_dtheta.h_view(n) += spectral.h_view(i)*basis_functions.h_view(1,i,n);
    }
  }
  return scalar_function_dtheta;
}

DualArray1D<Real> Strahlkorper::PhiDerivative(DualArray1D<Real> scalar_function) {
  // first find spectral representation
  DualArray1D<Real> spectral;
  Kokkos::realloc(spectral,nfilt);
  spectral = SpatialToSpectral(scalar_function);
  // calculate theta derivative
  DualArray1D<Real> scalar_function_dphi;
  Kokkos::realloc(scalar_function_dphi,nangles);
  for (int i=0; i<nfilt; ++i) {
    for (int n=0; n<nangles; ++n) {
      scalar_function_dphi.h_view(n) += spectral.h_view(i)*basis_functions.h_view(2,i,n);
    }
  }
  return scalar_function_dphi;
}

void Strahlkorper::EvaluateTangentVectors() {
  // tangent vectors are theta and phi derivatives of the x,y,z coordinate
  DualArray1D<Real> spatial_coord;
  Kokkos::realloc(spatial_coord,nangles);

  // i iterates over x, y, and z
  for (int i=0; i<3; ++i) {
    for (int n=0; n<nangles; ++n) {
      spatial_coord.h_view(n) = interp_coord.h_view(n,i);
    }
    DualArray1D<Real> dx;
    Kokkos::realloc(dx,nangles);
    
    // theta component
    dx = ThetaDerivative(spatial_coord);
    for (int n=0; n<nangles; ++n) {
      tangent_vectors.h_view(0,n,i) = dx.h_view(n);
    }
    
    // phi component
    dx = PhiDerivative(spatial_coord);
    for (int n=0; n<nangles; ++n) {
      tangent_vectors.h_view(1,n,i) = dx.h_view(n);
    }
  }

  // sync dual arrays
  tangent_vectors.template modify<HostMemSpace>();
  tangent_vectors.template sync<DevExeSpace>();
}


void Strahlkorper::EvaluateNormalOneForms() {

  // i iterates over x, y, and z components of the one form
  for (int n=0; n<nangles; ++n) {
    normal_oneforms.h_view(n,0) = tangent_vectors.h_view(0,n,1)*tangent_vectors.h_view(1,n,2)
                                - tangent_vectors.h_view(0,n,2)*tangent_vectors.h_view(1,n,1);
    normal_oneforms.h_view(n,1) = tangent_vectors.h_view(0,n,2)*tangent_vectors.h_view(1,n,0)
                                - tangent_vectors.h_view(0,n,0)*tangent_vectors.h_view(1,n,2);
    normal_oneforms.h_view(n,2) = tangent_vectors.h_view(0,n,0)*tangent_vectors.h_view(1,n,1)
                                - tangent_vectors.h_view(0,n,1)*tangent_vectors.h_view(1,n,0);
  }

  // sync dual arrays
  normal_oneforms.template modify<HostMemSpace>();
  normal_oneforms.template sync<DevExeSpace>();
}

// Jacobian matrix to transform vector to Cartesian basis
// first index r theta phi, second index x,y,z
void Strahlkorper::EvaluateSurfaceJacobian() {
  for (int n=0; n<nangles; ++n) {
    Real x = interp_coord.h_view(n,0);
    Real y = interp_coord.h_view(n,1);
    Real z = interp_coord.h_view(n,2);
    Real r = pointwise_radius.h_view(n);
    Real x2plusy2 = x*x + y*y;
    Real sqrt_x2plusy2 = sqrt(x2plusy2);

    // for x component
    surface_jacobian.h_view(n,0,0) = x/r;
    surface_jacobian.h_view(n,1,0) = x*z/sqrt_x2plusy2/r/r;
    surface_jacobian.h_view(n,2,0) = -y/x2plusy2;
  
    // for y component
    surface_jacobian.h_view(n,0,1) = y/r;
    surface_jacobian.h_view(n,1,1) = y*z/sqrt_x2plusy2/r/r;
    surface_jacobian.h_view(n,2,1) = x/x2plusy2;

    // for z component
    surface_jacobian.h_view(n,0,2) = z/r;
    surface_jacobian.h_view(n,1,2) = -sqrt_x2plusy2/r/r;
    surface_jacobian.h_view(n,2,2) = 0;
  }
  surface_jacobian.template modify<HostMemSpace>();
  surface_jacobian.template sync<DevExeSpace>();
}
// Analytical derivative of the Jacobian matrix
// first index x,y,z, second index r,theta,phi, third index x,y,z
void Strahlkorper::EvaluateSurfaceJacobianDerivative() {
  for (int n=0; n<nangles; ++n) {
    Real x = interp_coord.h_view(n,0);
    Real y = interp_coord.h_view(n,1);
    Real z = interp_coord.h_view(n,2);
    Real r = pointwise_radius.h_view(n);

    Real x2 = x*x;
    Real y2 = y*y;
    Real z2 = z*z;

    Real r2 = r*r;
    Real r3 = r2*r;
    Real r4 = r3*r;
    Real rxy2 = x*x + y*y;
    Real rxy = sqrt(rxy2);
    Real rxy3 = rxy2*rxy;
    Real rxy4 = rxy3*rxy;
    //****************** Partial x ********************
    // for x component
    d_surface_jacobian.h_view(n,0,0,0) = (r2-x2)/r3;
    d_surface_jacobian.h_view(n,0,1,0) = (-2*rxy2*x2 + r2*(rxy2-x2))*z/(r4*rxy3);
    d_surface_jacobian.h_view(n,0,2,0) = 2*x*y/rxy4;
  
    // for y component
    d_surface_jacobian.h_view(n,0,0,1) = -x*y/r3;
    d_surface_jacobian.h_view(n,0,1,1) = -(r2 + 2*rxy2)*x*y*z/(r4*rxy3);
    d_surface_jacobian.h_view(n,0,2,1) = (rxy2 - 2*x2)/rxy4;

    // for z component
    d_surface_jacobian.h_view(n,0,0,2) = -x*z/r3;
    d_surface_jacobian.h_view(n,0,1,2) = -x/(r2*rxy)+2*rxy*x/r4;
    d_surface_jacobian.h_view(n,0,2,2) = 0;

    //****************** Partial y ********************
    // for x component
    d_surface_jacobian.h_view(n,1,0,0) = -x*y/r3;
    d_surface_jacobian.h_view(n,1,1,0) = -(r2 + 2*rxy2)*x*y*z/(r4*rxy3);
    d_surface_jacobian.h_view(n,1,2,0) = -(rxy2 - 2*y2)/rxy4;

    // for y component
    d_surface_jacobian.h_view(n,1,0,1) = (r2-y2)/r3;
    d_surface_jacobian.h_view(n,1,1,1) = (-2*rxy2*y2 + r2*(rxy2-y2))*z/(r4*rxy3);
    d_surface_jacobian.h_view(n,1,2,1) = - 2*x*y/rxy4;

    // for z component
    d_surface_jacobian.h_view(n,1,0,2) = -y*z/r3;
    d_surface_jacobian.h_view(n,1,1,2) = -y/(r2*rxy) +2*rxy*y/r4;
    d_surface_jacobian.h_view(n,1,2,2) = 0;

    //****************** Partial z ********************
    // for x component
    d_surface_jacobian.h_view(n,2,0,0) = -x*z/r3;
    d_surface_jacobian.h_view(n,2,1,0) = x*(r2-2*z2)/(r4*rxy);
    d_surface_jacobian.h_view(n,2,2,0) = 0;

    // for y component
    d_surface_jacobian.h_view(n,2,0,1) = -y*z/r3;
    d_surface_jacobian.h_view(n,2,1,1) = y*(r2-2*z2)/(r4*rxy);
    d_surface_jacobian.h_view(n,2,2,1) = 0;

    // for z component
    d_surface_jacobian.h_view(n,2,0,2) = (r2-z2)/r3;
    d_surface_jacobian.h_view(n,2,1,2) = 2*rxy*z/r4;
    d_surface_jacobian.h_view(n,2,2,2) = 0;
  }
  d_surface_jacobian.template modify<HostMemSpace>();
  d_surface_jacobian.template sync<DevExeSpace>();
}