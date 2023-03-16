#ifndef EOS_IDEAL_C2P_HYD_HPP_
#define EOS_IDEAL_C2P_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file isothermal_c2p_hyd.hpp
//! \brief Various inline functions that transform a single state of conserved variables
//! into primitive variables (and the reverse, primitive to conserved) for hydrodynamics
//! with an isothermal EOS. Versions for both non-relativistic and relativistic fluids are
//! provided.
//

//----------------------------------------------------------------------------------------
//! \fn void SingleC2P_IsothermalSRHyd()
//! \brief Converts single state of conserved variables into primitive variables for
//! special relativistic hydrodynamics with an ideal gas EOS.

KOKKOS_INLINE_FUNCTION
void SingleC2P_IsothermalSRHyd(HydCons1D &u, const EOS_Data &eos, const Real s2, HydPrim1D &w,
                          bool &dfloor_used, bool &c2p_failure,
                          int &iter_used) {
  // Parameters
  const int max_iterations = 25;
  const Real tol = 1.0e-12;
  const Real v_max = 0.9999999999995;  // NOTE(@pdmullen): SQR(v_max) = 1.0 - tol;
  const Real kmax = 2.0*v_max/(1.0 + v_max*v_max);
  const Real gm1 = eos.gamma - 1.0;

  // apply density floor, without changing momentum or energy
  if (u.d < eos.dfloor) {
    u.d = eos.dfloor;
    dfloor_used = true;
  }

  // Here we use p = rho T = rho eps cs_rel_lim^2
  // and T = cs^2/ (1- (cs/cs_rel_lim)^2)  (=const)
  
  Real temperature = SQR(eos.iso_cs)/(1.0 - SQR(eos.iso_cs/eos.iso_cs_rel_lim));
  Real eps = temperature/SQR(eos.iso_cs_rel_lim);

  // Note that h is a constant for isothermal
  Real const h = 1.0 + eps*(1. + SQR(eos.iso_cs));     // (C1) & (C21)

  // Recast all variables (eq C2)
  Real r = sqrt(s2)/u.d;

  Real z = r/h;

  // iterations ended, compute primitives from resulting value of z
  Real const lor = sqrt(1.0 + z*z);  // (C15)

  // compute density then apply floor
  Real dens = u.d/lor;
  if (dens < eos.dfloor) {
    dens = eos.dfloor;
    dfloor_used = true;
  }

  Real const conv = 1.0/h;             // (C26)

  // set primitive variables
  w.d  = dens;
  w.vx = conv*(u.mx/u.d);  // (C26)
  w.vy = conv*(u.my/u.d);  // (C26)
  w.vz = conv*(u.mz/u.d);  // (C26)

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleP2C_IsothermalSRHyd()
//! \brief Converts single state of primitive variables into conserved variables for
//! special relativistic hydrodynamics with an isothermal EOS.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IsothermalSRHyd(const HydPrim1D &w, const Real iso_cs, const Real iso_cs_rel_lim, HydCons1D &u) {

  Real temperature = SQR(iso_cs)/(1.0 - SQR(iso_cs/iso_cs_rel_lim));
  Real eps = temperature/SQR(iso_cs_rel_lim);

  // Calculate Lorentz factor
  Real u0 = sqrt(1.0 + SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
  Real wgas_u0 = (1. + eps*(1. + SQR(iso_cs_rel_lim)))*w.d*u0;

  // Set conserved quantities
  u.d  = w.d * u0;
  u.mx = wgas_u0 * w.vx;            // In SR, vx/y/z are 4-velocity
  u.my = wgas_u0 * w.vy;
  u.mz = wgas_u0 * w.vz;
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void SingleP2C_IsothermalGRHyd()
//! \brief Converts single set of primitive into conserved variables.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IsothermalGRHyd(const Real glower[][4], const Real gupper[][4],
                          const HydPrim1D &w, const Real &iso_cs, const Real &iso_cs_rel_lim, HydCons1D &u) {
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

  Real temperature = SQR(iso_cs)/(1.0 - SQR(iso_cs/iso_cs_rel_lim));
  Real eps = temperature/SQR(iso_cs_rel_lim);

  Real wgas_u0 = (1. + eps* (1. + SQR(iso_cs_rel_lim))) * w.d* u0;

  // set conserved quantities
  u.d  = w.d * u0;
  u.mx = wgas_u0 * u_1;
  u.my = wgas_u0 * u_2;
  u.mz = wgas_u0 * u_3;
  return;
}

#endif // EOS_IDEAL_C2P_HYD_HPP_
