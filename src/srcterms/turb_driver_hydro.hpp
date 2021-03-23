#ifndef SRCTERMS_TURB_DRIVER_HYDRO_HPP_
#define SRCTERMS_TURB_DRIVER_HYDRO_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_driver_hydro.hpp
//  \brief defines turbulence driver class, which implements data and functions for
//  randomly forced turbulence which evolves via an Ornstein-Uhlenbeck stochastic process
//  This derived class implements routines for non-relativistic  hydrodynamics

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "turb_driver.hpp"
#include "srcterms.hpp"

//----------------------------------------------------------------------------------------
//! \class TurbulenceDriverHydro

class TurbulenceDriverHydro : public TurbulenceDriver
{
 public:
  TurbulenceDriverHydro(MeshBlockPack *pp, ParameterInput *pin);
  ~TurbulenceDriverHydro() = default;

  // function to compute/apply forcing
  virtual void ApplyForcing(int stage) override;

  virtual void ImplicitEquation(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI,
      DvceArray5D<Real> &Ru) override;

  void ImplicitEquationRK2(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI,
      DvceArray5D<Real> &Ru);

  void ImplicitEquationRK3(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI,
      DvceArray5D<Real> &Ru);

  void ApplyForcingImplicit(DvceArray5D<Real> &force_, DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI);
  void ComputeImplicitSources(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI, DvceArray5D<Real> &Ru);

  void ApplyForcingSourceTermsExplicit(DvceArray5D<Real> &u);
  array_sum::GlobalSum ComputeNetMomentum(DvceArray5D<Real> &u, DvceArray5D<Real> &ftmp);
  array_sum::GlobalSum ComputeNetForce(DvceArray5D<Real> &u, DvceArray5D<Real> &ftmp);
  array_sum::GlobalSum ComputeNetEnergyInjection(DvceArray5D<Real> &w, DvceArray5D<Real> &ftmp);

};


#endif // SRCTERMS_TURB_DRIVER_HYDRO_HPP_
