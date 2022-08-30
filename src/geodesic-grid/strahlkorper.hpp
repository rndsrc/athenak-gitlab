#ifndef GEODESIC_GRID_strahlkorper_HPP_
#define GEODESIC_GRID_strahlkorper_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file strahlkorper.hpp
//  \brief definitions for strahlkorper (deformed spherical grid) class

#include "athena.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "athena_tensor.hpp"

//----------------------------------------------------------------------------------------
//! \class Strahlkorper

class Strahlkorper: public SphericalGrid {
 public:
   // Creates a geodetic grid with nlev levels and radius rad
   Strahlkorper(MeshBlockPack *pmy_pack, int nlev, Real rad);
   ~Strahlkorper();

   DualArray1D<Real> pointwise_radius;
   DualArray3D<Real> basis_functions;
   DualArray3D<Real> tangent_vectors;
   DualArray2D<Real> normal_oneforms;

   int nlevel;
   void SetPointwiseRadius(DualArray1D<Real> rad_tmp, Real ctr[3]);  // set indexing for interpolation
   void EvaluateSphericalHarm();
   void EvaluateTangentVectors();
   void EvaluateNormalOneForms();

   Real Integrate(DualArray1D<Real> integrand);
   DualArray1D<Real> ThetaDerivative(DualArray1D<Real> scalar_function);
   DualArray1D<Real> PhiDerivative(DualArray1D<Real> scalar_function);
   DualArray1D<Real> SpatialToSpectral(DualArray1D<Real> scalar_function);
   // AthenaTensor<Real, TensorSymm::NONE, 3, 1> dtheta_u;     // theta tangent vector 
   // AthenaTensor<Real, TensorSymm::NONE, 3, 1> dphi_u;     // phi tangent vector 
   

   std::pair<double,double> SWSphericalHarm(int l, int m, int s, Real theta, Real phi);
   Real RealSphericalHarm(int l, int m, Real theta, Real phi);
   Real RealSphericalHarm_dtheta(int l, int m, Real theta, Real phi);
   Real RealSphericalHarm_dphi(int l, int m, Real theta, Real phi);
   Real MakeReal(int l, int m, Real theta, Real phi, std::pair<double,double> (*func)(int, int, int, Real, Real));
   std::pair<double,double> SphericalHarm_dtheta(int l, int m, Real theta, Real phi);
   std::pair<double,double> SphericalHarm_dphi(int l, int m, Real theta, Real phi);

 private:
};

#endif // GEODESIC_GRID_SPHERICAL_GRID_HPP_
