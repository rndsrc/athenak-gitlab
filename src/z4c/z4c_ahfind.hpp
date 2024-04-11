#ifndef Z4C_AHFIND_HPP
#define Z4C_AHFIND_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_ahfind.hpp
//  \brief definitions for the AHF class

#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c_macros.hpp"
#include "z4c.hpp"
#include "geodesic-grid/gauss_legendre.hpp"


// Forward declaration
class Mesh;
class ParameterInput;
class GaussLegendreGrid;

namespace z4c {
//! \class AHF
//! \brief Apparent Horizon Finder
class AHF {

public:
  //! Creates the AHF object
  AHF(Mesh * pmesh, ParameterInput * pin, int n);
  //! Destructor (will close output file)
  ~AHF();

  //! Evaluate surface null expansion
  void SurfaceNullExpansion(MeshBlockPack *pmbp, GaussLegendreGrid *S);
  //! Analytical expansion for Schwarzschild, for testing purposes.
  void AnalyticSurfaceNullExpansion(GaussLegendreGrid *S);
  //! Fast flow Iterator
  void FastFlow();


  //! Gauss-Legendre defining the horizon
  GaussLegendreGrid *S;
  //! Horizon found
  bool ah_found;
  //! Initial guess
  Real initial_radius;
  //! Center
  Real center[3];

  //! Fast flow parameters
  Real hmean_tol;
  Real mass_tol;
  int flow_iterations;
  Real flow_alpha_beta_const;
  bool verbose;

  //! Grid
  int nlev, nfilt;
  DualArray1D<Real> r_spectral;
  DualArray1D<Real> r_spectral_np1;
  DualArray1D<Real> H_spectral;

  //! Maximum number of iteration;
  int maxit;
  AthenaSurfaceTensor<Real,TensorSymm::NONE,3,0> null_expansion;

  Mesh const * pmesh;
  FILE * pofile;
};

}// end namespace z4c
#endif
