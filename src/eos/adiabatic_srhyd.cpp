//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adiabatic_srhyd.cpp
//  \brief implements EOS functions in derived class for special relativistic ad. hydro
// Conserved to primitive variable inversion implements algorithm described in Appendix C
// of Galeazzi et al., PhysRevD, 88, 064009 (2013). Equation references are to this paper.

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor
    
AdiabaticSRHydro::AdiabaticSRHydro(MeshBlockPack *pp, ParameterInput *pin)
  : EquationOfState(pp, pin)
{      
  eos_data.is_adiabatic = true;
  eos_data.gamma = pin->GetReal("eos","gamma");
  eos_data.iso_cs = 0.0;
}  

//----------------------------------------------------------------------------------------
// \!fn Real EquationC22()
// \brief Inline function to compute function f(z) defined in eq. C22 of Galeazzi et al.
// The ConsToPRim algorithms finds the root of this function f(z)=0

KOKKOS_INLINE_FUNCTION
Real EquationC22(Real z, Real &u_d, Real q, Real r, Real gm1, Real pfloor)
{
  Real const w = sqrt(1.0 + z*z);         // (C15)
  Real const wd = u_d/w;                  // (C15)
  Real eps = w*q - z*r + (z*z)/(1.0 + w); // (C16)

  //NOTE: The following generalizes to ANY equation of state
  eps = fmax(pfloor/(wd*gm1), eps);                          // (C18)
  Real const h = (1.0 + eps) * (1.0 + (gm1*eps)/(1.0+eps));   // (C1) & (C21)

  return (z - r/h); // (C22)
}

//----------------------------------------------------------------------------------------
// \!fn void ConsToPrim()
// \brief Converts conserved into primitive variables in nonrelativistic adiabatic hydro.
// Implementation follows Wolfgang Kastaun's algorithm described in Appendix C of
// Galeazzi et al., PhysRevD, 88, 064009 (2013).  Roots of "master function" (eq. C22) 
// found by false position method.
//
// In SR hydrodynamics, the conserved variables are: (D, E - D, m^i),
// where D = \gamma \rho is the density in the lab frame, \gamma = (1 + u^2)^{1/2} is the//  Lorentz factor, u^i = \gamma v^i are the spatial components of the 4-velocity (v^i is
// the 3-velocity), \rho is the comoving/fluid/rest frame mass density, 
// E = \gamma^2 w - P_gas is the total energy, w = \rho + [\Gamma / (\Gamma - 1)] P_gas
// is the total enthalpy, \Gamma is the adiabatic index, P_gas is the gas pressure, and
// m^i = \gamma w u^i are components of the momentum in the lab frame.
//
// The primitive variables are: (\rho, P_gas, u^i).
// Note we store components of the 4-velocity (not 3-velocity) in the primitive variables.
//
// This function operates over entire MeshBlock, including ghost cells.  

void AdiabaticSRHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = eos_data.gamma - 1.0; 
  Real pfloor_ = eos_data.pressure_floor;
  Real &dfloor_ = eos_data.density_floor;

  // Parameters
  int const max_iterations = 25;
  Real const tol = 1.0e-12;
  Real const v_sq_max = 1.0 - tol;

  par_for("hyd_con2prim", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real& u_d  = cons(m, IDN,k,j,i);
      Real& u_e  = cons(m, IEN,k,j,i);
      const Real& u_m1 = cons(m, IM1,k,j,i);
      const Real& u_m2 = cons(m, IM2,k,j,i);
      const Real& u_m3 = cons(m, IM3,k,j,i);

      Real& w_d  = prim(m, IDN,k,j,i);
      Real& w_p  = prim(m, IPR,k,j,i);
      Real& w_vx = prim(m, IVX,k,j,i);
      Real& w_vy = prim(m, IVY,k,j,i);
      Real& w_vz = prim(m, IVZ,k,j,i);

      // apply density floor, without changing momentum or energy
      u_d = (u_d > dfloor_) ?  u_d : dfloor_;

      // apply energy floor
//      Real ee_min = pfloor_/gm1;
//      u_e = (u_e > ee_min) ?  u_e : ee_min;


      // Recast all variables (eq C2)
      // Variables q and r defined in anonymous namspace: global this file
      Real q = u_e/u_d;
      Real r = sqrt(SQR(u_m1) + SQR(u_m2) + SQR(u_m3))/u_d;
      Real kk = r/(1.+q);

      // Enforce lower velocity bound (eq. C13). This bound combined with a floor on
      // the value of p will guarantee "some" result of the inversion
      kk = fmin(2.* sqrt(v_sq_max)/(1.0 + v_sq_max), kk);

      // Compute bracket (C23)
      auto zm = 0.5*kk/sqrt(1.0 - 0.25*kk*kk);
      auto zp = kk/sqrt(1.0 - kk*kk);

      // Evaluate master function (eq C22) at bracket values
      Real fm = EquationC22(zm, u_d, q, r, gm1, pfloor_);
      Real fp = EquationC22(zp, u_d, q, r, gm1, pfloor_);

      // For simplicity on the GPU, find roots using the false position method
      int iterations = max_iterations;
      // If bracket within tolerances, don't bother doing any iterations
      if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
        iterations = -1;
      }
      Real z = 0.5*(zm + zp);

      for (int ii=0; ii < iterations; ++ii) {
	z =  (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
        Real f = EquationC22(z, u_d, q, r, gm1, pfloor_);

        // Quit if convergence reached
	// NOTE: both z and f are of order unity
	if ((fabs(zm-zp) < tol ) || (fabs(f) < tol )){
/**
std::cout << "|zm-zp|=" <<fabs(zm-zp)<<" |f|="<< fabs(f) << "for i=" <<  ii << std::endl;
**/
	    break;
	}

        // assign zm-->zp if root bracketed by [z,zp]
	if (f * fp < 0.0) {
	   zm = zp;
	   fm = fp;
	   zp = z;
	   fp = f;

        // assign zp-->z if root bracketed by [zm,z]
	} else {
	   fm = 0.5*fm; // 1/2 comes from "Illinois algorithm" to accelerate convergence
	   zp = z;
	   fp = f;
	}
      }

      // iterations ended, compute primitives from resulting value of z
      Real const w = sqrt(1.0 + z*z); // (C15)
      w_d = u_d/w;                    // (C15)

      //NOTE: The following generalizes to ANY equation of state
      Real eps = w*q - z*r + (z*z)/(1.0 + w); // (C16)
      eps = fmax(pfloor_/w_d/gm1, eps);                 // (C18)
      Real h = (1. + eps) * (1.0 + (gm1*eps)/(1.+eps)); // (C1) & (C21)
      w_p = w_d*gm1*eps;

      Real const conv = 1.0/(h*u_d); // (C26)
      w_vx = conv * u_m1;           // (C26)
      w_vy = conv * u_m2;           // (C26)
      w_vz = conv * u_m3;           // (C26)

      // convert scalars (if any)
      for (int n=nhyd; n<(nhyd+nscal); ++n) {
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u_d;
      }

      // TODO error handling
//
//      if (false)
//      {
//	Real gamma_adi = gm1+1.;
//	Real rho_eps = w_p / gm1;
//	//FIXME ERM: Only ideal fluid for now
//        Real wgas = w_d + gamma_adi / gm1 *w_p;
//	
//	auto gamma = sqrt(1. +z*z);
//        cons(m,IDN,k,j,i) = w_d * gamma;
//        cons(m,IEN,k,j,i) = wgas*gamma*gamma - w_p - w_d * gamma; 
//        cons(m,IM1,k,j,i) = wgas * gamma * w_vx;
//        cons(m,IM2,k,j,i) = wgas * gamma * w_vy;
//        cons(m,IM3,k,j,i) = wgas * gamma * w_vz;
//      }

    }
  );

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void PrimToCons()
// \brief Converts primitive into conserved variables.  Operates only over active cells.
//  Recall in SR hydrodynamics the conserved variables are: (D, E-D, m^i), and the
//  primitive variables are: (\rho, P_gas, u^i).

void AdiabaticSRHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gamma_prime = eos_data.gamma/(eos_data.gamma - 1.0); 

  par_for("hyd_prim2cons", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real& u_d  = cons(m, IDN,k,j,i);
      Real& u_e  = cons(m, IEN,k,j,i);
      Real& u_m1 = cons(m, IM1,k,j,i);
      Real& u_m2 = cons(m, IM2,k,j,i);
      Real& u_m3 = cons(m, IM3,k,j,i);

      const Real& w_d  = prim(m, IDN,k,j,i);
      const Real& w_p  = prim(m, IPR,k,j,i);
      const Real& w_vx = prim(m, IVX,k,j,i);
      const Real& w_vy = prim(m, IVY,k,j,i);
      const Real& w_vz = prim(m, IVZ,k,j,i);

      // Calculate Lorentz factor
      Real u0 = sqrt(1.0 + SQR(w_vx) + SQR(w_vy) + SQR(w_vz));
      Real wgas_u0 = (w_d + gamma_prime * w_p) * u0;

      // Set conserved quantities
      u_d  = w_d * u0;
      u_e  = wgas_u0 * u0 - w_p - u_d; // In SR, evolve E - D
      u_m1 = wgas_u0 * w_vx;           // In SR, w_vx/y/z are 4-velocity
      u_m2 = wgas_u0 * w_vy;
      u_m3 = wgas_u0 * w_vz;

      // convert scalars (if any)
      for (int n=nhyd; n<(nhyd+nscal); ++n) {
        cons(m,n,k,j,i) = prim(m,n,k,j,i)*u_d;
      }

    }
  );

  return;
}