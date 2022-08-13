//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_grid.cpp
//  \brief Initializes a spherical grid to interpolate data onto

#include <cmath>
#include <iostream>
#include <list>

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
    basis_functions("basis_functions",1,1,1) {
  
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
int fac(int i) {
  int result = 1;
  if (i>0) {
    while (i>0) {
      result*=i;
      i-=1;
    }
  }
  return(result);
}
/*
Real Strahlkorper::SphericalHarm(int l, int m, Real theta, Real phi) {
    Real value;
    if (m<0) {
        value = pow((-1),m)*sqrt(2)*sqrt((2*l+1)*fac(l-abs(m))/4/M_PI/fac(l+abs(m)))*std::tr1::sph_legendre(l,abs(m),cos(theta))*sin(abs(m)*phi);
    } else if (m=0) {
        value = sqrt((2*l+1)/4/M_PI)*std::assoc_legendre(l,m,cos(theta));
    } else (m>0) {
        value = pow((-1),m)*sqrt(2)*sqrt((2*l+1)*fac(l-m)/4/M_PI/fac(l+m))*std::sph_legendre(l,m,cos(theta))*cos(m*phi);
    }

    return value;
}
*/




//Calculate spin-weighted spherical harmonics using Wigner-d matrix notation see e.g. Eq II.7, II.8 in 0709.0093
std::pair<double,double> Strahlkorper::SWSphericalHarm(int l, int m, int s, Real theta, Real phi) {
  Real wignerd = 0;
  int k1,k2,k;
  k1 = std::max(0, m-s);
  k2 = std::min(l+m,l-s);
  for (k = k1; k<k2+1; ++k) {
    wignerd += pow((-1),k)*sqrt(fac(l+m)*fac(l-m)*fac(l+s)*fac(l-s))*pow(cos(theta/2.0),2*l+m-s-2*k)*pow(sin(theta/2.0),2*k+s-m)/(fac(l+m-k)*fac(l-s-k)*fac(k)*fac(k+s-m));
  }
  wignerd *= pow((-1),s)*sqrt((2*l+1)/(4*M_PI));
  return std::make_pair(wignerd*cos(m*phi), wignerd*sin(m*phi));
}

// theta derivative of the spherical harmonics
std::pair<double,double> Strahlkorper::SphericalHarm_t(int l, int m, Real theta, Real phi) {
    std::pair<double,double> value;
    if (l==m) {
        std::pair<double,double> value2 = Strahlkorper::SWSphericalHarm(l,m,0,theta,phi);
        value.first = m/tan(theta)*value2.first; 
        value.second = m/tan(theta)*value2.second; 
    } else {
        std::pair<double,double> value2 = Strahlkorper::SWSphericalHarm(l,m,0,theta,phi);
        std::pair<double,double> value3 = Strahlkorper::SWSphericalHarm(l,m+1,0,theta,phi);
        value.first = m/tan(theta)*value2.first + sqrt((n-m)*(n+m+1))*exp(-1j*phi)*sph_harm(m+1,n,phi,theta)
    }
}

// phi derivative of the spherical harmonics
std::pair<double,double> Strahlkorper::SphericalHarm_p(int l, int m, Real theta, Real phi) {
  Real wignerd = 0;
  int k1,k2,k;
  k1 = std::max(0, m-s);
  k2 = std::min(l+m,l-s);
  for (k = k1; k<k2+1; ++k) {
    wignerd += pow((-1),k)*sqrt(fac(l+m)*fac(l-m)*fac(l+s)*fac(l-s))*pow(cos(theta/2.0),2*l+m-s-2*k)*pow(sin(theta/2.0),2*k+s-m)/(fac(l+m-k)*fac(l-s-k)*fac(k)*fac(k+s-m));
  }
  wignerd *= pow((-1),s)*sqrt((2*l+1)/(4*M_PI));
  return std::make_pair(wignerd*cos(m*phi), wignerd*sin(m*phi));
}

Real Strahlkorper::MakeReal(std::pair<double,double> (*func)(int, int, int, Real, Real), int l, int m, Real theta, Real phi) {
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

    return;
}

/*
KOKKOS_INLINE_FUNCTION
int max(int i, int j) {
  if (i>=j) {
    return(i);
  } else {
    return(j);
  }
}

KOKKOS_INLINE_FUNCTION
int min(int i, int j) {
  if (i<=j) {
    return(i);
  } else {
    return(j);
  }
}
*/