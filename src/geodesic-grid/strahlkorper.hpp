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
   Strahlkorper(MeshBlockPack *pmy_pack, int nlev, Real rad, int nfilt);
   ~Strahlkorper();
   
   DualArray1D<Real> pointwise_radius;
   DualArray3D<Real> basis_functions;
   DualArray3D<Real> tangent_vectors;
   DualArray2D<Real> normal_oneforms;
   DualArray3D<Real> surface_jacobian;
   DualArray4D<Real> d_surface_jacobian;

   int nfilt;
   int nlevel;
   void InitializeRadius();
   void SetPointwiseRadius(DualArray1D<Real> rad_tmp, Real ctr[3]);  // set indexing for interpolation
   void EvaluateSphericalHarm();
   void EvaluateTangentVectors();
   void EvaluateNormalOneForms();
   void EvaluateSurfaceJacobian();
   void EvaluateSurfaceJacobianDerivative();
   Real Integrate(DualArray1D<Real> integrand);
   Real Integrate(AthenaSurfaceTensor<Real,TensorSymm::NONE,3,0> integrand);
   DualArray1D<Real> ThetaDerivative(DualArray1D<Real> scalar_function);
   DualArray1D<Real> PhiDerivative(DualArray1D<Real> scalar_function);
   DualArray1D<Real> SpatialToSpectral(DualArray1D<Real> scalar_function);
   DualArray1D<Real> SpatialToSpectral(AthenaSurfaceTensor<Real,TensorSymm::NONE,3,0> scalar_function);
   DualArray1D<Real> SpectralToSpatial(DualArray1D<Real> scalar_spectrum);

 private:
};

#endif // GEODESIC_GRID_SPHERICAL_GRID_HPP_
