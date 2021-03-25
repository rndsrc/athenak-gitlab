#ifndef SRCTERMS_TURB_DRIVER_HYDRO_REL_HPP_
#define SRCTERMS_TURB_DRIVER_HYDRO_REL_HPP_
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

class TurbulenceDriverHydroRel : public TurbulenceDriver
{
 public:
  TurbulenceDriverHydroRel(MeshBlockPack *pp, ParameterInput *pin);
  ~TurbulenceDriverHydroRel() = default;

  // function to compute/apply forcing
  virtual void ApplyForcing(int stage) override;

  virtual void ImplicitEquation(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI,
      DvceArray5D<Real> &Ru) override;

  void ImplicitEquationRK2(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI,
      DvceArray5D<Real> &Ru);

  void ImplicitEquationRK3(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI,
      DvceArray5D<Real> &Ru);

  void ApplyForcingImplicit(DvceArray5D<Real> &force_, DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI);
  void ApplyForcingImplicitNew(DvceArray5D<Real> &force_, DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI);
  void ComputeImplicitSources(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI, DvceArray5D<Real> &Ru);

  void ApplyForcingSourceTermsExplicit(DvceArray5D<Real> &u);
  array_sum::GlobalSum ComputeNetMomentum(DvceArray5D<Real> &u, DvceArray5D<Real> &ftmp);
  array_sum::GlobalSum ComputeNetForce(DvceArray5D<Real> &u, DvceArray5D<Real> &ftmp);
  array_sum::GlobalSum ComputeNetEnergyInjection(DvceArray5D<Real> &u, DvceArray5D<Real> &ftmp);

KOKKOS_INLINE_FUNCTION
Real GetEpsfromTauWithCoolingKernel(Real u_e, Real SdotF, Real Lambda, Real lorentz, Real gamma_adi, Real eps){
	auto hW = (1. + gamma_adi*eps)*lorentz;
	auto pud = eps*(gamma_adi-1.)/(lorentz);
	auto cool = Lambda*eps*eps*eps*eps;
	auto fv = SdotF/(hW + cool);
	return hW - pud - 1. - u_e + cool -fv; 
};

KOKKOS_INLINE_FUNCTION
Real GetEpsfromTauWithCooling(Real u_e, Real SdotF, Real Lambda, Real lorentz, Real gamma_adi){


      Real const tol = 1.e-10;
      int const max_iterations = 200;

      Real zm = 0.;
      Real zp = 2.e1; // FIXME
	
      auto fm = GetEpsfromTauWithCoolingKernel(u_e,SdotF,Lambda,lorentz,gamma_adi,zm);
      auto fp = GetEpsfromTauWithCoolingKernel(u_e,SdotF,Lambda,lorentz,gamma_adi,zp);

      Real z;
      z=0.5*(zm+zp);

      //For simplicity on the GPU, use the false position method
	int iterations = max_iterations;
	if((fabs(zm-zp) < tol ) || ((fabs(fm) + fabs(fp)) < 2.*tol )){
	    iterations = -1;
	}
      for(int ii=0; ii< iterations; ++ii){

	z =  (zm*fp - zp*fm)/(fp-fm);

	auto f = GetEpsfromTauWithCoolingKernel(u_e,SdotF,Lambda,lorentz,gamma_adi,z);


	if(f * fp < 0.){
	   zm = zp;
	   fm = fp;
	   zp = z;
	   fp = f;
	}
	else{
	   fm = 0.5*fm;
	   zp = z;
	   fp = f;
	}

	// NOTE: both z and f are of order unity
	if((fabs(zm-zp) < tol*max(fabs(zm),fabs(zp)) ) || (fabs(f) < tol )){
	    break;
	}

      }

      return z;
}; 

};


#endif // SRCTERMS_TURB_DRIVER_HYDRO_HPP_
