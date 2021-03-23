#ifndef SRCTERMS_TURB_DRIVER_HPP_
#define SRCTERMS_TURB_DRIVER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_driver.hpp
//  \brief defines turbulence driver class, which implements data and functions for
//  randomly forced turbulence which evolves via an Ornstein-Uhlenbeck stochastic process

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "imex.hpp"

//----------------------------------------------------------------------------------------
//! \class TurbulenceDriver

class TurbulenceDriver : public ImEx
{
 public:
  TurbulenceDriver(MeshBlockPack *pp, ParameterInput *pin);
  ~TurbulenceDriver() = default;

  // function to compute/apply forcing
  virtual void ApplyForcing(int stage)=0;

  // data
  DvceArray5D<Real> force;        // forcing for driving hydro variables
  DvceArray5D<Real> force_tmp;    // second force register for OU evolution

  DvceArray3D<Real> x1sin;   // array for pre-computed sin(k x)
  DvceArray3D<Real> x1cos;   // array for pre-computed cos(k x)
  DvceArray3D<Real> x2sin;   // array for pre-computed sin(k y)
  DvceArray3D<Real> x2cos;   // array for pre-computed cos(k y)
  DvceArray3D<Real> x3sin;   // array for pre-computed sin(k z)
  DvceArray3D<Real> x3cos;   // array for pre-computed cos(k z)

  DvceArray3D<Real> amp1;
  DvceArray3D<Real> amp2;
  DvceArray3D<Real> amp3;

  DvceArray2D<int64_t> seeds; // random seeds

  // parameters of driving
  int nlow,nhigh,ntot,nwave;
  Real tcorr,dedt;
  Real expo;


  bool initialized = false;
  // functions
  void InitializeModes();
  void NewRandomForce(DvceArray5D<Real> &ftmp);

};


#endif // SRCTERMS_TURB_DRIVER_HPP_
