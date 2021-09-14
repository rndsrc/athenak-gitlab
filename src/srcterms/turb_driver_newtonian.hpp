#ifndef SRCTERMS_TURB_DRIVER_HYDRO_HPP_
#define SRCTERMS_TURB_DRIVER_HYDRO_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_driver_newtonian.hpp
//  \brief defines turbulence driver class, which implements data and functions for
//  randomly forced turbulence which evolves via an Ornstein-Uhlenbeck stochastic process
//  This derived class implements routines for non-relativistic  hydrodynamics

#include "turb_driver.hpp"

//----------------------------------------------------------------------------------------
//! \class TurbulenceDriverHydro

class TurbulenceDriverNewtonian : public TurbulenceDriver
{
 public:
  TurbulenceDriverNewtonian(MeshBlockPack *pp, ParameterInput *pin);
  ~TurbulenceDriverNewtonian() = default;

  // function to compute/apply forcing
  virtual TaskStatus AddForcing(Driver *pdrive, int stage) override;

  auto ComputeNetMomentum(DvceArray5D<Real> &ftmp);
  auto ComputeNetEnergyInjection(DvceArray5D<Real> &w, DvceArray5D<Real> &ftmp);

  virtual void GlobalNormalization(DvceArray5D<Real> &ftmp) override;

};


#endif // SRCTERMS_TURB_DRIVER_HYDRO_HPP_
