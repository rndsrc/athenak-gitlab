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
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Strahlkorper::Strahlkorper(MeshBlockPack *ppack, int nlev, Real rad):
    SphericalGrid(ppack,nlev,rad),
    pointwise_radius("pointwise_radius",1),
    basis_functions("basis_functions",1,1,1), 
    tangent_vectors("tangent_vectors",1,1,1),
    normal_oneforms("normal_oneforms",1,1) {
  
  nlevel = nlev;
  // reallocate and set interpolation coordinates, indices, and weights
  // int &ng = pmy_pack->pmesh->mb_indcs.ng;
  Kokkos::realloc(pointwise_radius,nangles);

  // SetPointwiseRadius();

  return;
}

//----------------------------------------------------------------------------------------
//! \brief Strahlkorper destructor

Strahlkorper::~Strahlkorper() {
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
  return;
}

KOKKOS_INLINE_FUNCTION
double fac(int i) {
  double result = 1;
  if (i>0) {
    while (i>0) {
      result*=i;
      i-=1;
    }
  }
  return(result);
}

//Calculate spin-weighted spherical harmonics using Wigner-d matrix notation see e.g. Eq II.7, II.8 in 0709.0093
std::pair<double,double> Strahlkorper::SWSphericalHarm(int l, int m, int s, Real theta, Real phi) {
  Real wignerd = 0;
  int k1,k2,k;
  k1 = std::max(0, m-s);
  k2 = std::min(l+m,l-s);

  for (k = k1; k<=k2; ++k) {
    wignerd += pow((-1),k)*pow(cos(theta/2.0),2*l+m-s-2*k)*pow(sin(theta/2.0),2*k+s-m)/(fac(l+m-k)*fac(l-s-k)*fac(k)*fac(k+s-m));
  }
  wignerd *= pow((-1),s)*sqrt((2*l+1)/(4*M_PI))*sqrt(fac(l+m))*sqrt(fac(l-m))*sqrt(fac(l+s))*sqrt(fac(l-s));
  return std::make_pair(wignerd*cos(m*phi), wignerd*sin(m*phi));
}

// theta derivative of the s=0 spherical harmonics
std::pair<double,double> Strahlkorper::SphericalHarm_dtheta(int l, int m, Real theta, Real phi) {
  std::pair<double,double> value;
  if (l==m) {
    std::pair<double,double> value2 = Strahlkorper::SWSphericalHarm(l,m,0,theta,phi);
    value.first = m/tan(theta)*value2.first; 
    value.second = m/tan(theta)*value2.second; 
  } else {
    std::pair<double,double> value2 = Strahlkorper::SWSphericalHarm(l,m,0,theta,phi);
    std::pair<double,double> value3 = Strahlkorper::SWSphericalHarm(l,m+1,0,theta,phi);
    value.first = m/tan(theta)*value2.first + sqrt((l-m)*(l+m+1))*( cos(phi)*value3.first - sin(-phi)*value3.second);
    value.second = m/tan(theta)*value2.second + sqrt((l-m)*(l+m+1))*( cos(phi)*value3.second + sin(-phi)*value3.first);
  }
  return value;
}

// phi derivative of the s=0 spherical harmonics
std::pair<double,double> Strahlkorper::SphericalHarm_dphi(int l, int m, Real theta, Real phi) {
  std::pair<double,double> value;
  std::pair<double,double> value2 = Strahlkorper::SWSphericalHarm(l,m,0,theta,phi);
  value.first = -m*value2.second;
  value.second = m*value2.first;
  return value;
}

Real Strahlkorper::RealSphericalHarm(int l, int m, Real theta, Real phi) {
  double value;
  if (m==0) {
    value = Strahlkorper::SWSphericalHarm(l,m,0,theta,phi).first;
  } else if (m>0) {
    value = sqrt(2)*pow((-1),m)*Strahlkorper::SWSphericalHarm(l,m,0,theta,phi).first;
  } else {
    value = sqrt(2)*pow((-1),(-m))*Strahlkorper::SWSphericalHarm(l,m,0,theta,phi).second;
  }
  return value;
}

Real Strahlkorper::RealSphericalHarm_dtheta(int l, int m, Real theta, Real phi) {
  double value;
  if (m==0) {
    value = Strahlkorper::SphericalHarm_dtheta(l,m,theta,phi).first;
  } else if (m>0) {
    value = sqrt(2)*pow((-1),m)*Strahlkorper::SphericalHarm_dtheta(l,m,theta,phi).first;
  } else {
    value = sqrt(2)*pow((-1),(-m))*Strahlkorper::SphericalHarm_dtheta(l,m,theta,phi).second;
  }
  return value;
}

Real Strahlkorper::RealSphericalHarm_dphi(int l, int m, Real theta, Real phi) {
  double value;
  if (m==0) {
    value = Strahlkorper::SphericalHarm_dphi(l,m,theta,phi).first;
  } else if (m>0) {
    value = sqrt(2)*pow((-1),m)*Strahlkorper::SphericalHarm_dphi(l,m,theta,phi).first;
  } else {
    value = sqrt(2)*pow((-1),(-m))*Strahlkorper::SphericalHarm_dphi(l,m,theta,phi).second;
  }
  return value;
}

Real Strahlkorper::MakeReal(int l, int m, Real theta, Real phi, std::pair<double,double> (*func)(int, int, int, Real, Real)) {
  double value;
  if (m==0) {
    value = func(l,m,0,theta,phi).first;
  } else if (m>0) {
    value = sqrt(2)*pow((-1),m)*func(l,m,0,theta,phi).first;
  } else {
    value = sqrt(2)*pow((-1),(-m))*func(l,m,0,theta,phi).second;
  }
  return value;
}

void Strahlkorper::EvaluateSphericalHarm() {
  Kokkos::realloc(basis_functions,3,4*nlevel*nlevel,nangles);
  for (int a=0; a<4*nlevel*nlevel; a++) {
    int l = (int) sqrt(a);
    int m = (int) (a-l*l-l);
    for (int n=0; n<nangles; ++n) {
      Real &theta = polar_pos.h_view(n,0);
      Real &phi = polar_pos.h_view(n,1);
      basis_functions.h_view(0,a,n) = Strahlkorper::RealSphericalHarm(l, m, theta, phi);
      basis_functions.h_view(1,a,n) = Strahlkorper::RealSphericalHarm_dtheta(l, m, theta, phi);
      basis_functions.h_view(2,a,n) = Strahlkorper::RealSphericalHarm_dphi(l, m, theta, phi);
      //basis_functions.h_view(1,a,n) = MakeReal(l, m, theta, phi, SphericalHarm_dtheta);
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
  Kokkos::realloc(spectral,4*nlevel*nlevel);
  Kokkos::realloc(integrand,nangles);

  for (int i=0; i<4*nlevel*nlevel; ++i) {
    for (int n=0; n<nangles; ++n) {
      integrand.h_view(n) = scalar_function.h_view(n)*basis_functions.h_view(0,i,n);
    }
    spectral.h_view(i) = Integrate(integrand);
  }
  return spectral;
}

DualArray1D<Real> Strahlkorper::ThetaDerivative(DualArray1D<Real> scalar_function) {
  // first find spectral representation
  DualArray1D<Real> spectral;
  Kokkos::realloc(spectral,4*nlevel*nlevel);
  spectral = SpatialToSpectral(scalar_function);
  // calculate theta derivative
  DualArray1D<Real> scalar_function_dtheta;
  Kokkos::realloc(scalar_function_dtheta,nangles);
  for (int i=0; i<4*nlevel*nlevel; ++i) {
    for (int n=0; n<nangles; ++n) {
      scalar_function_dtheta.h_view(n) += spectral.h_view(i)*basis_functions.h_view(1,i,n);
    }
  }
  return scalar_function_dtheta;
}

DualArray1D<Real> Strahlkorper::PhiDerivative(DualArray1D<Real> scalar_function) {
  // first find spectral representation
  DualArray1D<Real> spectral;
  Kokkos::realloc(spectral,4*nlevel*nlevel);
  spectral = SpatialToSpectral(scalar_function);
  // calculate theta derivative
  DualArray1D<Real> scalar_function_dphi;
  Kokkos::realloc(scalar_function_dphi,nangles);
  for (int i=0; i<4*nlevel*nlevel; ++i) {
    for (int n=0; n<nangles; ++n) {
      scalar_function_dphi.h_view(n) += spectral.h_view(i)*basis_functions.h_view(2,i,n);
    }
  }
  return scalar_function_dphi;
}

void Strahlkorper::EvaluateTangentVectors() {
  Kokkos::realloc(tangent_vectors,2,nangles,3);
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
  Kokkos::realloc(normal_oneforms,nangles,3);

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

// Now evaluate normal vectors, indices raised with spatial metric

// void Strahlkorper::EvaluateNormalUnitVectors() {
//
// }