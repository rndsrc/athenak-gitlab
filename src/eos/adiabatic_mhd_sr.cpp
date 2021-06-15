
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adiabatic_hydro.cpp
//  \brief implements EOS functions in derived class for nonrelativistic adiabatic hydro

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor
    
AdiabaticMHDRel::AdiabaticMHDRel(MeshBlockPack *pp, ParameterInput *pin)
  : EquationOfState(pp, pin)
{      
  eos_data.is_adiabatic = true;
  eos_data.gamma = pin->GetReal("eos","gamma");
  eos_data.iso_cs = 0.0;
}  

//----------------------------------------------------------------------------------------
// \!fn void ConservedToPrimitive()
// \brief No-Op version of MHD cons to prim functions.  Never used in MHD.

void AdiabaticMHDRel::ConsToPrimMHD(const DvceArray5D<Real> &cons,
         const DvceFaceFld4D<Real> &b, DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc)
{

  auto ncells = pmy_pack->mb_cells;
  int ng = ncells.ng;
  int n1 = ncells.nx1 + 2*ng;
  int n2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*ng) : 1;
  int n3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*ng) : 1;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = eos_data.gamma - 1.0;
  Real gamma_adi = eos_data.gamma;

  Real &dfloor_ = eos_data.density_floor;
  Real &pfloor_ = eos_data.pressure_floor;
  Real ee_min = pfloor_/gm1;

  Real mm_sq_ee_sq_max = 1.0 - 1.0e-12;  // max. of squared momentum over energy

    // Parameters
    int const max_iterations = 15;
    Real const tol = 1.0e-12;
    Real const pgas_uniform_min = 1.0e-12;
    Real const a_min = 1.0e-12;
    Real const v_sq_max = 1.0 - 1.0e-12;
    Real const rr_max = 1.0 - 1.0e-12;


  par_for("mhd_con2prim", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real& u_d  = cons(m, IDN,k,j,i);
      Real& u_m1 = cons(m, IM1,k,j,i);
      Real& u_m2 = cons(m, IM2,k,j,i);
      Real& u_m3 = cons(m, IM3,k,j,i);
      Real& u_e  = cons(m, IEN,k,j,i);

      Real& w_d  = prim(m, IDN,k,j,i);
      Real& w_vx = prim(m, IVX,k,j,i);
      Real& w_vy = prim(m, IVY,k,j,i);
      Real& w_vz = prim(m, IVZ,k,j,i);
      Real& w_p  = prim(m, IPR,k,j,i);

      // cell-centered fields are simple linear average of face-centered fields
      Real& w_bx = bcc(m,IBX,k,j,i);
      Real& w_by = bcc(m,IBY,k,j,i);
      Real& w_bz = bcc(m,IBZ,k,j,i);
      w_bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));  
      w_by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      w_bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));

      // apply density floor, without changing momentum or energy
///      u_d = (u_d > dfloor_) ?  u_d : dfloor_;
//      w_d = u_d;

      // apply energy floor
//      u_e = (u_e > ee_min) ?  u_e : ee_min;
//      w_p = pfloor_;

      Real ee = u_d + u_e;

      Real mm_sq = SQR(u_m1) + SQR(u_m2) + SQR(u_m3);

      Real bb_sq = w_bx*w_bx + w_by*w_by + w_bz*w_bz;

      Real tt = u_m1 * w_bx + u_m2 * w_by + u_m3 * w_bz;

      Real m2_max = mm_sq_ee_sq_max * SQR(ee);
      if( mm_sq > m2_max){
	Real factor = std::sqrt(m2_max/mm_sq);
	u_m1*= factor;
	u_m2*= factor;
	u_m3*= factor;

	tt*= factor;
      }

    bool failed= false;


    // Calculate functions of conserved quantities
    Real d = 0.5 * (mm_sq * bb_sq - tt*tt);                  // (NH 5.7)
    d = fmax(d, 0.0);
    Real pgas_min = cbrt(27.0/4.0 * d) - ee - 0.5*bb_sq;
    pgas_min = fmax(pfloor_, pgas_uniform_min);

    // Iterate until convergence
    Real pgas[3];
    pgas[0] =  pgas_min; // Do we have a previous step
    int n;
    for (n = 0; n < max_iterations; ++n) {
      // Step 1: Calculate cubic coefficients
      Real a;
      if (n%3 != 2) {
        a = ee + pgas[n%3] + 0.5*bb_sq;  // (NH 5.7)
	a = fmax(a, a_min);
      }

      // Step 2: Calculate correct root of cubic equation
      Real phi, eee, ll, v_sq;
      if (n%3 != 2) {
	phi = acos(1.0/a * sqrt(27.0*d/(4.0*a)));                     // (NH 5.10)
	eee = a/3.0 - 2.0/3.0 * a * cos(2.0/3.0 * (phi+M_PI));               // (NH 5.11)
	ll = eee - bb_sq;                                                       // (NH 5.5)
	v_sq = ll * (bb_sq+ll);
	v_sq = (mm_sq*ll*ll + tt*tt*(bb_sq+2.0*ll)) / (v_sq*v_sq); // (NH 5.2)
	v_sq = fmin(fmax(v_sq, 0.0), v_sq_max);
	Real gamma_sq = 1.0/(1.0-v_sq);                                         // (NH 3.1)
	Real gamma = sqrt(gamma_sq);                                       // (NH 3.1)
	Real wgas = ll/gamma_sq;                                                // (NH 5.1)
	Real rho = u_d/gamma;                                                    // (NH 4.5)
	pgas[(n+1)%3] = (gamma_adi-1.0)/gamma_adi * (wgas - rho);               // (NH 4.1)
	pgas[(n+1)%3] = std::max(pgas[(n+1)%3], pgas_min);
      }

      // Step 3: Check for convergence
      if (n%3 != 2) {
	if (pgas[(n+1)%3] > pgas_min && fabs(pgas[(n+1)%3]-pgas[n%3]) < tol) {
	  break;
	}
      }

      // Step 4: Calculate Aitken accelerant and check for convergence
      if (n%3 == 2) {
	Real rr = (pgas[2] - pgas[1]) / (pgas[1] - pgas[0]);  // (NH 7.1)
	if ((rr!=rr) || fabs(rr) > rr_max) {
	  continue;
	}
	pgas[0] = pgas[1] + (pgas[2] - pgas[1]) / (1.0 - rr);  // (NH 7.2)
	pgas[0] = fmax(pgas[0], pgas_min);
	if (pgas[0] > pgas_min && fabs(pgas[0]-pgas[2]) < tol) {
	  break;
	}
      }
    }

    // Step 5: Set primitives
    if (n == max_iterations) {
      failed = true;
    }
    w_p = pgas[(n+1)%3];
//    if (!std::isfinite(w_p)) {
//      failed=true;
//    }
    Real a = ee + prim(m,IPR,k,j,i) + 0.5*bb_sq;                      // (NH 5.7)
    a = std::max(a, a_min);
    Real phi = std::acos(1.0/a * std::sqrt(27.0*d/(4.0*a)));        // (NH 5.10)
    Real eee = a/3.0 - 2.0/3.0 * a * std::cos(2.0/3.0 * (phi+M_PI));  // (NH 5.11)
    Real ll = eee - bb_sq;                                          // (NH 5.5)
    Real v_sq = (mm_sq*SQR(ll) + SQR(tt)*(bb_sq+2.0*ll))
		/ SQR(ll * (bb_sq+ll));                             // (NH 5.2)
    v_sq = std::min(std::max(v_sq, 0.0), v_sq_max);
    Real gamma_sq = 1.0/(1.0-v_sq);                                 // (NH 3.1)
    Real gamma = std::sqrt(gamma_sq);                               // (NH 3.1)

    w_d = u_d/gamma;                      // (NH 4.5)
//    if (!std::isfinite(w_d)) {
//      failed=true;
//    }
    Real ss = tt/ll;                          // (NH 4.8)
    w_vx = (u_m1 + ss*w_bx) / (ll + bb_sq);  // (NH 4.6)
    w_vy = (u_m2 + ss*w_by) / (ll + bb_sq);  // (NH 4.6)
    w_vz = (u_m3 + ss*w_bz) / (ll + bb_sq);  // (NH 4.6)
    w_vx *= gamma;           // (NH 4.6)
    w_vy *= gamma;           // (NH 4.6)
    w_vz *= gamma;           // (NH 4.6)

//    if (!std::isfinite(w_vx) || !std::isfinite(w_vy) || !std::isfinite(w_vz)) {
//      failed = true;
//    }

      // apply pressure floor, correct total energy
//      u_e = (w_p > pfloor_) ?  u_e : ((pfloor_/gm1) + e_k);
//      w_p = (w_p > pfloor_) ?  w_p : pfloor_;


    // TODO error handling

      if (false)
      {
	Real gamma_adi = gm1+1.;
	Real rho_eps = w_p / gm1;
	//FIXME ERM: Only ideal fluid for now
        Real wgas = w_d + gamma_adi / gm1 *w_p;

        Real b0 = w_bx * w_vx + w_by * w_vy + w_bz * w_vz;
        Real b1 = (w_bx + b0 * w_vx) / gamma;
        Real b2 = (w_by + b0 * w_vy) / gamma;
        Real b3 = (w_bz + b0 * w_vz) / gamma;
        Real b_sq = -SQR(b0) + SQR(b1) + SQR(b2) + SQR(b3);

	wgas += b_sq;
	
        cons(m,IDN,k,j,i) = w_d * gamma;
        cons(m,IEN,k,j,i) = wgas*gamma*gamma - (w_p + 0.5*b_sq) - w_d * gamma - b0*b0; //rho_eps * gamma_sq + (w_p + cons(IDN,k,j,i)/(gamma+1.))*(v_sq*gamma_sq);
        cons(m,IM1,k,j,i) = wgas * gamma * w_vx - b0*b1;
        cons(m,IM2,k,j,i) = wgas * gamma * w_vy - b0*b2;
        cons(m,IM3,k,j,i) = wgas * gamma * w_vz - b0*b3;
      }

      // convert scalars (if any)
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u_d;
      }
    }
  );

  return;
}
