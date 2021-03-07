#ifndef IMEX_HPP_
#define IMEX_HPP_
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

//----------------------------------------------------------------------------------------
//! \class TurbulenceDriver

class ImEx
{
 public:
  ImEx(MeshBlockPack *pp, ParameterInput *pin);
  ~ImEx() = default;



  // Implicit source terms
  DvceArray3D<Real> Ru1;
  DvceArray3D<Real> Ru2;
  DvceArray3D<Real> Ru3;

  int nimplicit, noff;

  // function to compute/apply forcing
  void ApplySourceTermsImplicit(DvceArray5D<Real> &u, DvceArray5D<Real> &w, int stage);
  void ApplySourceTermsImplicitPreStage(DvceArray5D<Real> &u,DvceArray5D<Real> &u);

  virtual void ImplicitKernel(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI,
      DvceArray5D<Real> &Ru) =0;

  // function to compute/apply forcing
  void ApplySourceTermsImplicitRK3(DvceArray5D<Real> &u, DvceArray5D<Real> &w, int stage);
  void ApplySourceTermsImplicitPreStageRK3(DvceArray5D<Real> &u,DvceArray5D<Real> &u);

protected:

  Real ceff[4];

  enum method {RKexplicit, RK1, RK2, RK3};
  int this_imex = method::RKexplicit;
  int nimplicit=0;
  int noff=0;

  int current_stage=0;


  void allocate_storage(int _noff, int _nimplicit);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this TurbulenceDriver

};


#endif // SRCTERMS_TURB_DRIVER_HPP_
