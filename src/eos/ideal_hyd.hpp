#ifndef EOS_IDEAL_HYD_HPP_
#define EOS_IDEAL_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_hyd.hpp
//! \brief Various inline functions that transform a single state of conserved variables
//! into primitive variables (and the reverse, primitive to conserved) for hydrodynamics
//! with an ideal gas EOS. Versions for both non-relativistic and relativistic fluids are
//! provided.

//----------------------------------------------------------------------------------------
//! \fn void SingleC2P_IdealHyd()
//! \brief Converts single state of conserved variables into primitive variables for
//! non-relativistic hydrodynamics with an ideal gas EOS.
//! Conserved = (d,M1,M2,M3,E), Primitive = (d,vx,vy,vz,e)
//! where E=total energy density and e=internal energy density

KOKKOS_INLINE_FUNCTION
void SingleC2P_IdealHyd(HydCons1D &u, const EOS_Data &eos,
                        HydPrim1D &w, bool &dfloor_used, bool &efloor_used) {
  const Real &dfloor_ = eos.dfloor;
  Real efloor = eos.pfloor/(eos.gamma - 1.0);

  // apply density floor, without changing momentum or energy
  if (u.d < dfloor_) {
    u.d = dfloor_;
    dfloor_used = true;
  }
  w.d = u.d;

  // compute velocities
  Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;

  // set internal energy, apply floor, correct total energy (if needed)
  Real e_k = 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz));
  w.e = (u.e - e_k);
  if (w.e < efloor) {
    w.e = efloor;
    u.e = efloor + e_k;
    efloor_used = true;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleP2C_IdealHyd()
//! \brief Converts single state of primitive variables into conserved variables for
//! non-relativistic hydrodynamics with an ideal gas EOS.
//! Conserved = (d,M1,M2,M3,E), Primitive = (d,vx,vy,vz,e)
//! where E=total energy density and e=internal energy density

KOKKOS_INLINE_FUNCTION
void SingleP2C_IdealHyd(const HydPrim1D &w, HydCons1D &u) {
  u.d  = w.d;
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  u.e = w.e + 0.5*w.d*(SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationC22()
//! \brief Inline function to compute function f(z) defined in eq. C22 of Galeazzi et al.
//! used to convert conserved to primitive variables for relativistic hydrodynamics
//! The ConsToPrim algorithm finds the root of this function f(z)=0

KOKKOS_INLINE_FUNCTION
Real EquationC22(Real z, Real &u_d, Real q, Real r, EOS_Data eos) {
  Real const gm1 = eos.gamma - 1.0;
  Real const w = sqrt(1.0 + z*z);         // (C15)
  Real const wd = u_d/w;                  // (C15)
  Real eps = w*q - z*r + (z*z)/(1.0 + w); // (C16)

  //NOTE: The following generalizes to ANY equation of state
  eps = fmax(eos.pfloor/(wd*gm1), eps);   // (C18)
  Real const h = 1.0 + eos.gamma*eps;     // (C1) & (C21)
  return (z - r/h); // (C22)
}

//----------------------------------------------------------------------------------------
//! \fn void SingleC2P_IdealSRHyd()
//! \brief Converts single state of conserved variables into primitive variables for
//! special relativistic hydrodynamics with an ideal gas EOS.

KOKKOS_INLINE_FUNCTION
void SingleC2P_IdealSRHyd(HydCons1D &u, const EOS_Data &eos, const Real s2, HydPrim1D &w,
                          bool &dfloor_used, bool &efloor_used, int &iter_used) {
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

  // apply energy floor
  if (u.e < eos.pfloor/gm1) {
    u.e = eos.pfloor/gm1;
    efloor_used = true;
  }

  // Recast all variables (eq C2)
  Real q = u.e/u.d;
  Real r = sqrt(s2)/u.d;
  Real kk = r/(1.+q);

  // Enforce lower velocity bound (eq. C13). This bound combined with a floor on
  // the value of p will guarantee "some" result of the inversion
  kk = fmin(kmax, kk);

  // Compute bracket (C23)
  Real zm = 0.5*kk/sqrt(1.0 - 0.25*kk*kk);
  Real zp = kk/sqrt(1.0 - kk*kk);

  // Evaluate master function (eq C22) at bracket values
  Real fm = EquationC22(zm, u.d, q, r, eos);
  Real fp = EquationC22(zp, u.d, q, r, eos);

  // For simplicity on the GPU, find roots using the false position method
  int iterations = max_iterations;
  // If bracket within tolerances, don't bother doing any iterations
  if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
    iterations = -1;
  }
  Real z = 0.5*(zm + zp);

  for (iter_used=0; iter_used < iterations; ++iter_used) {
    z =  (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
    Real f = EquationC22(z, u.d, q, r, eos);

    // Quit if convergence reached
    // NOTE: both z and f are of order unity
    if ((fabs(zm-zp) < tol ) || (fabs(f) < tol )) {
      break;
    }

    // assign zm-->zp if root bracketed by [z,zp]
    if (f * fp < 0.0) {
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

  // iterations ended, compute primitives from resulting value of z
  Real const lor = sqrt(1.0 + z*z);       // (C15)
  w.d = u.d/lor;                    // (C15)

  // NOTE(@ermost): The following generalizes to ANY equation of state
  Real eps = lor*q - z*r + (z*z)/(1.0 + lor);   // (C16)
  Real epsmin = eos.pfloor/(w.d*gm1);
  if (eps <= epsmin) {                      // C18
    eps = epsmin;
    efloor_used = true;
  }

  Real const conv = 1.0/((1.0 + eos.gamma*eps)*u.d); // (C1 and C21)
  w.vx = conv * u.mx;            // (C26)
  w.vy = conv * u.my;            // (C26)
  w.vz = conv * u.mz;            // (C26)

  w.e = w.d*eps;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleP2C_IdealSRHyd()
//! \brief Converts single state of primitive variables into conserved variables for
//! special relativistic hydrodynamics with an ideal gas EOS.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IdealSRHyd(const HydPrim1D &w, const Real gam, HydCons1D &u) {
  // Calculate Lorentz factor
  Real u0 = sqrt(1.0 + SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
  Real wgas_u0 = (w.d + gam*w.e)*u0;

  // Set conserved quantities
  u.d  = w.d * u0;
  u.e  = wgas_u0 * u0 - (gam-1.0)*w.e - u.d;  // In SR, evolve E - D
  u.mx = wgas_u0 * w.vx;            // In SR, vx/y/z are 4-velocity
  u.my = wgas_u0 * w.vy;
  u.mz = wgas_u0 * w.vz;
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void SingleP2C_IdealGRHyd()
//! \brief Converts single set of primitive into conserved variables.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IdealGRHyd(const Real glower[][4], const Real gupper[][4],
                          const HydPrim1D &w, const Real &gam, HydCons1D &u) {
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
  Real wgas_u0 = (w.d + gam * w.e) * u0;

  // set conserved quantities
  u.d  = w.d * u0;
  u.e  = wgas_u0 * u_0 + (gam-1.0)*w.e + u.d;  // evolve T^t_t + D
  u.mx = wgas_u0 * u_1;
  u.my = wgas_u0 * u_2;
  u.mz = wgas_u0 * u_3;
  return;
}

#endif // EOS_IDEAL_HYD_HPP_
