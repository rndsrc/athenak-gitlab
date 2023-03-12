#ifndef EOS_IDEAL_C2P_MHD_HPP_
#define EOS_IDEAL_C2P_MHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_mhd.hpp
//! \brief Various inline functions that transform a single state of conserved variables
//! into primitive variables (and the reverse, primitive to conserved) for MHD
//! with an isothermal EOS. Versions for both non-relativistic and relativistic fluids are
//! provided.


//----------------------------------------------------------------------------------------
//! \fn Real Equation49_Isothermal()
//! \brief Inline function to compute function fa(mu) defined in eq. 49 of Kastaun et al.
//! The root fa(mu)==0 of this function corresponds to the upper bracket for
//! solving Equation44_Isothermal

KOKKOS_INLINE_FUNCTION
Real Equation49_Isothermal(const Real mu, const Real b2, const Real rp, const Real r) {
  Real const x = 1.0/(1.0 + mu*b2);             // (26)
  Real rbar = (x*x*r*r + mu*x*(1.0 + x)*rp*rp); // (38)
  return mu*sqrt(1.0 + rbar) - 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn Real Equation44_Isothermal()
//! \brief Inline function to compute function f(mu) defined in eq. 44 of Kastaun et al.
//! The ConsToPRim algorithms finds the root of this function f(mu)=0

KOKKOS_INLINE_FUNCTION
Real Equation44_Isothermal(const Real mu, const Real b2, const Real rpar, const Real r,
                const Real u_d,  EOS_Data eos) {
  Real const x = 1./(1.+mu*b2);                  // (26)
  Real rbar = (x*x*r*r + mu*x*(1.+x)*rpar*rpar); // (38)

  Real z2 = (mu*mu*rbar/(fabs(1.- SQR(mu)*rbar))); // (32)
  Real w = sqrt(1.+z2);

  Real const wd = u_d/w;                           // (34)
  // Here we use p = rho T = rho eps cs_rel_lim^2
  // and T = cs^2/ (1- (cs/cs_rel_lim)^2)  (=const)
  
  Real temperature = SQR(eos.iso_cs)/(1.0 - SQR(eos.iso_cs/eos.iso_cs_rel_lim));
  Real eps = temperature/SQR(eos.iso_cs_rel_lim);

  Real const h = 1.0 + eps*(1. + SQR(eos.iso_cs));   
  return mu - 1./(h/w + rbar*mu);         // (45)
}

//----------------------------------------------------------------------------------------
//! \fn void SingleC2P_IsothermalSRMHD()
//! \brief Converts single state of conserved variables into primitive variables for
//! special relativistic MHD with an ideal gas EOS. Note input CONSERVED state contains
//! cell-centered magnetic fields, but PRIMITIVE state returned via arguments does not.

KOKKOS_INLINE_FUNCTION
void SingleC2P_IsothermalSRMHD(MHDCons1D &u, const EOS_Data &eos, Real s2, Real b2, Real rpar,
                          HydPrim1D &w, bool &dfloor_used, bool &efloor_used,
                          bool &c2p_failure, int &max_iter) {
  // Parameters
  const int max_iterations = 25;
  const Real tol = 1.0e-12;

  // apply density floor, without changing momentum or energy
  if (u.d < eos.dfloor) {
    u.d = eos.dfloor;
    dfloor_used = true;
  }

  // Recast all variables (eq 22-24)
  Real r = sqrt(s2)/u.d;
  Real isqrtd = 1.0/sqrt(u.d);
  Real bx = u.bx*isqrtd;
  Real by = u.by*isqrtd;
  Real bz = u.bz*isqrtd;

  // normalize b2 and rpar as well since they contain b
  b2 /= u.d;
  rpar *= isqrtd;

  // Need to find initial bracket. Requires separate solve
  Real zm=0.;
  Real zp=1.; // This is the lowest specific enthalpy admitted by the EOS

  // Evaluate master function (eq 49) at bracket values
  Real fm = Equation49_Isothermal(zm, b2, rpar, r );
  Real fp = Equation49_Isothermal(zp, b2, rpar, r );

  // For simplicity on the GPU, find roots using the false position method
  int iterations = max_iterations;
  // If bracket within tolerances, don't bother doing any iterations
  if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
    iterations = -1;
  }
  Real z = 0.5*(zm + zp);

  int iter;
  for (iter=0; iter<iterations; ++iter) {
    z =  (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
    Real f = Equation49_Isothermal(z, b2, rpar, r, q);
    // Quit if convergence reached
    // NOTE(@ermost): both z and f are of order unity
    if ((fabs(zm-zp) < tol) || (fabs(f) < tol)) {
      break;
    }
    // assign zm-->zp if root bracketed by [z,zp]
    if (f*fp < 0.0) {
      zm = zp;
      fm = fp;
      zp = z;
      fp = f;
    } else {  // assign zp-->z if root bracketed by [zm,z]
      fm = 0.5*fm; // 1/2 comes from "Illinois algorithm" to accelerate convergence
      zp = z;
      fp = f;
    }
  }
  max_iter = (iter > max_iter) ? iter : max_iter;

  // Found brackets. Now find solution in bounded interval, again using the
  // false position method
  zm= 0.;
  zp= z;

  // Evaluate master function (eq 44) at bracket values
  fm = Equation44_Isothermal(zm, b2, rpar, r, u.d, eos);
  fp = Equation44_Isothermal(zp, b2, rpar, r, u.d, eos);

  iterations = max_iterations;
  if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
    iterations = -1;
  }
  z = 0.5*(zm + zp);

  for (iter=0; iter<iterations; ++iter) {
    z = (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
    Real f = Equation44_Isothermal(z, b2, rpar, r, u.d, eos);
    // Quit if convergence reached
    // NOTE: both z and f are of order unity
    if ((fabs(zm-zp) < tol) || (fabs(f) < tol)) {
      break;
    }
    // assign zm-->zp if root bracketed by [z,zp]
    if (f*fp < 0.0) {
      zm = zp;
      fm = fp;
      zp = z;
      fp = f;
    } else {  // assign zp-->z if root bracketed by [zm,z]
      fm = 0.5*fm; // 1/2 comes from "Illinois algorithm" to accelerate convergence
      zp = z;
      fp = f;
    }
  }
  max_iter = (iter > max_iter) ? iter : max_iter;

  // check if convergence is established within max_iterations.  If not, trigger a C2P
  // failure and return floored density, pressure, and primitive velocities. Future
  // development may trigger averaging of (successfully inverted) neighbors in the event
  // of a C2P failure.
  if (max_iter==max_iterations) {
    w.d = eos.dfloor;
    w.vx = 0.0;
    w.vy = 0.0;
    w.vz = 0.0;
    c2p_failure = true;
    return;
  }

  // iterations ended, compute primitives from resulting value of z
  Real &mu = z;
  Real const x = 1./(1.+mu*b2);                               // (26)
  Real rbar = (x*x*r*r + mu*x*(1.+x)*rpar*rpar);              // (38)
  Real z2 = (mu*mu*rbar/(fabs(1.- SQR(mu)*rbar)));            // (32)
  Real lor = sqrt(1.0 + z2);

  // compute density then apply floor
  Real dens = u.d/lor;
  if (dens < eos.dfloor) {
    dens = eos.dfloor;
    dfloor_used = true;
  }

  // compute specific internal energy density then apply floor
  Real temperature = SQR(eos.iso_cs)/(1.0 - SQR(eos.iso_cs/eos.iso_cs_rel_lim));
  Real eps = temperature/SQR(eos.iso_cs_rel_lim);

  // set parameters required for velocity inversion
  Real const h = 1.0 + eps*(1. + SQR(eos.iso_cs));     // (C1) & (C21)
  Real const conv = lor/(h*lor + b2);  // (C26)

  // set primitive variables
  w.d  = dens;
  w.vx = conv*(u.mx/u.d + bx*rpar/(h*lor));  // (C26)
  w.vy = conv*(u.my/u.d + by*rpar/(h*lor));  // (C26)
  w.vz = conv*(u.mz/u.d + bz*rpar/(h*lor));  // (C26)

  return;
}

//--------------------------------------------------------------------------------------
//! \fn void SingleP2C_IsothermalSRMHD()
//! \brief Converts single set of primitive into conserved variables in SRMHD.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IsothermalSRMHD(const MHDPrim1D &w, const Real iso_cs, const Real iso_cs_rel_lim, HydCons1D &u) {
  // Calculate Lorentz factor
  Real u0 = sqrt(1.0 + SQR(w.vx) + SQR(w.vy) + SQR(w.vz));

  // compute specific internal energy density then apply floor
  Real temperature = SQR(eos.iso_cs)/(1.0 - SQR(eos.iso_cs/eos.iso_cs_rel_lim));
  Real eps = temperature/SQR(eos.iso_cs_rel_lim);
  Real const h = 1.0 + eps*(1. + SQR(eos.iso_cs));     // (C1) & (C21)

  // Calculate 4-magnetic field
  Real b0 = w.bx*w.vx + w.by*w.vy + w.bz*w.vz;
  Real b1 = (w.bx + b0 * w.vx) / u0;
  Real b2 = (w.by + b0 * w.vy) / u0;
  Real b3 = (w.bz + b0 * w.vz) / u0;
  Real b_sq = -SQR(b0) + SQR(b1) + SQR(b2) + SQR(b3);

  // Set conserved quantities
  Real wtot_u02 = (w.d * h + b_sq) * u0 * u0;
  u.d  = w.d * u0;
  u.mx = wtot_u02 * w.vx / u0 - b0 * b1;
  u.my = wtot_u02 * w.vy / u0 - b0 * b2;
  u.mz = wtot_u02 * w.vz / u0 - b0 * b3;
  return;
}

//--------------------------------------------------------------------------------------
//! \fn voidSingleP2C_IsothermalGRMHD()
//! \brief Converts single set of primitive into conserved variables in GRMHD.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IsothermalGRMHD(const Real glower[][4], const Real gupper[][4],
                          const MHDPrim1D &w, const Real iso_cs, const Real iso_cs_rel_lim, HydCons1D &u) {

  // compute specific internal energy density then apply floor
  Real temperature = SQR(eos.iso_cs)/(1.0 - SQR(eos.iso_cs/eos.iso_cs_rel_lim));
  Real eps = temperature/SQR(eos.iso_cs_rel_lim);
  Real const h = 1.0 + eps*(1. + SQR(eos.iso_cs));     // (C1) & (C21)
						       //
  // Calculate 4-velocity (exploiting symmetry of metric)
  Real q = glower[1][1]*w.vx*w.vx +2.0*glower[1][2]*w.vx*w.vy +2.0*glower[1][3]*w.vx*w.vz
         + glower[2][2]*w.vy*w.vy +2.0*glower[2][3]*w.vy*w.vz
         + glower[3][3]*w.vz*w.vz;
  Real alpha = sqrt(-1.0/gupper[0][0]);
  Real gamma = sqrt(1.0 + q);
  Real u0 = gamma / alpha;
  Real u1 = w.vx - alpha * gamma * gupper[0][1];
  Real u2 = w.vy - alpha * gamma * gupper[0][2];
  Real u3 = w.vz - alpha * gamma * gupper[0][3];

  // lower vector indices
  Real u_0 = glower[0][0]*u0 + glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
  Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
  Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
  Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

  // Calculate 4-magnetic field
  Real b0 = u_1*w.bx + u_2*w.by + u_3*w.bz;
  Real b1 = (w.bx + b0 * u1) / u0;
  Real b2 = (w.by + b0 * u2) / u0;
  Real b3 = (w.bz + b0 * u3) / u0;

  // lower vector indices
  Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
  Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
  Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
  Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
  Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

  Real wtot = w.d*h + b_sq;
  u.d  = w.d * u0;
  u.mx = wtot * u0 * u_1 - b0 * b_1;
  u.my = wtot * u0 * u_2 - b0 * b_2;
  u.mz = wtot * u0 * u_3 - b0 * b_3;
  return;
}

#endif // EOS_IDEAL_C2P_MHD_HPP_
