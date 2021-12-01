//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file therm_instab_linear.cpp
//! \brief Problem generator for linear thermal instability

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"
#include "srcterms/srcterms.hpp"
#include "globals.hpp"
#include "utils/units.hpp" 


// derivate of the cooling function for temperature
static Real DLnLambdaDLnT(Real rho, Real temp);

// calculate growth rate of perturbation
static Real SolveCubic(const Real b, const Real c, const Real d);
static Real ThermalInstabilityGrowthRate(const Real rho, const Real pgas, const Real k);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Problem Generator for linear thermal instability

void ProblemGenerator::UserProblem(MeshBlockPack *pmbp, ParameterInput *pin)
{
  if (pmbp->phydro == nullptr and pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Thermal instability problem generator can only be run with Hydro and/or MHD, "
       << "but no <hydro> or <mhd> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  int nmb1 = (pmbp->nmb_thispack-1);

  // Set Units
  // mean particle mass in unit of hydrogen atom mass
  Real mu = pin->GetOrAddReal("problem","mu",0.618);
  // density unit in unit of number density
  Real dunit = mu*physical_constants::m_hydrogen; 
  // length unit in unit of parsec
  Real lunit = pin->GetOrAddReal("problem","lunit",1.0)*physical_constants::pc; 
  // velocity unit in unit of km/s
  Real vunit = pin->GetOrAddReal("problem","vunit",1.0)*physical_constants::kms; 
  units::punit->UpdateUnits(dunit, lunit, vunit, mu);

  // Get temperature in Kelvin
  Real temp = pin->GetOrAddReal("problem","temp",1.0);
  
  //Find the equilibrium point of the cooling curve by n*Lambda-Gamma=0
  Real number_density=2.0e-26/CoolFn(temp);
  Real rho_0 = number_density*units::punit->mu*
               physical_constants::m_hydrogen/units::punit->density;
  Real cs_iso = std::sqrt(temp/units::punit->temperature);  

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    auto &u0 = pmbp->phydro->u0;
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real pgas_0 = rho_0*cs_iso*cs_iso;
    Real cs = std::sqrt(eos.gamma*pgas_0/rho_0);
    
    // Print info
    if (global_variable::my_rank == 0) {
      std::cout << "============== Check Initialization ===============" << std::endl;
      std::cout << "  rho_0 (code) = " << rho_0 << std::endl;
      std::cout << "  sound speed (code) = " << cs << std::endl;
      std::cout << "  mu = " << units::punit->mu << std::endl;
      std::cout << "  temperature (c.g.s) = " << temp << std::endl;
      std::cout << "  cooling function (c.g.s) = " << CoolFn(temp) << std::endl;
    }
    // End print info

    // Read pertubation parameters
    Real wave_len = pin->GetOrAddReal("problem","wave_len",1.0);
    Real amp = pin->GetOrAddReal("problem","amplitude",1.0e-3);
    Real kx = 2.0*M_PI/(wave_len);
    // Calculate growth rate, omega in the linear perturbation theory
    Real om = ThermalInstabilityGrowthRate(rho_0,pgas_0,kx);

    // Print info
    if (global_variable::my_rank == 0) {
      std::cout << "  k = " << kx << std::endl;
      std::cout << "  omega = " << om << std::endl;
      std::cout << "============= omega for different k  ==============" << std::endl;
      for (int i=0; i<100; i++){
        Real kx_tmp = 2.0*M_PI*i*i/100.0;
        Real om_tmp = ThermalInstabilityGrowthRate(rho_0,pgas_0,kx_tmp);
        std::cout << " k = " << kx_tmp << " omega = " << om_tmp << std::endl;
      }
    }
    // End print info

    // Set initial conditions
    par_for("pgen_thermal_instability", DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i)
      {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

        Real sn = std::sin(kx*x1v);
        Real cn = std::cos(kx*x1v);
        
        u0(m,IDN,k,j,i) = rho_0 * (1.0 + amp * cn);
        u0(m,IM1,k,j,i) = -rho_0 * amp * (om/kx) * sn;
        u0(m,IM2,k,j,i) = 0.0;
        u0(m,IM3,k,j,i) = 0.0;
        if (eos.is_ideal) {
          u0(m,IEN,k,j,i) = pgas_0/gm1 - amp*rho_0/gm1*(om/kx)*(om/kx)*cn + 
            0.5*rho_0*amp*amp*(om/kx)*(om/kx)*sn*sn;
        }
      }
    );
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn static Real DLnLambdaDLnT(Real rho, Real temp)
//! \brief derivate of the cooling function for temperature

static Real DLnLambdaDLnT(Real rho, Real temp) {
  // original data from Shure et al. paper, covers 4.12 < logt < 8.16
  const float lhd[102] = {
      -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497, -21.6211, -21.6595,
      -21.6426, -21.5688, -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
      -20.8986, -20.8281, -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051,
      -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749, -20.6709, -20.6748,
      -20.7089, -20.8031, -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
      -21.4538, -21.5055, -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073,
      -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243, -21.8875, -21.9738,
      -22.0671, -22.1537, -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
      -22.3590, -22.3512, -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595,
      -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353, -22.5659, -22.5895,
      -22.6059, -22.6161, -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
      -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844,
      -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928};

  Real logt = std::log10(temp);

  //  for temperatures less than 10^4 K, use Koyama & Inutsuka
  if (logt <= 4.2) {
    Real lambda_cooling  = (2.0e-19*std::exp(-1.184e5/(temp + 1.0e3)) + 
                           2.8e-28*std::sqrt(temp)*std::exp(-92.0/temp));
    Real dlambda_dtemp_1 = 2.0e-19*std::exp(-1.184e5/(temp + 1.0e3))*
                           1.184e5/(temp + 1.0e3)/(temp + 1.0e3);
    Real dlambda_dtemp_2 = 2.8e-28*std::sqrt(temp)*std::exp(-92.0/temp)*
                           (0.5/temp+92.0/temp/temp);
    return (Real) temp / lambda_cooling * (dlambda_dtemp_1 + dlambda_dtemp_2);
  }

  // for temperatures above 10^8.15 use CGOLS fit
  if (logt > 8.15) {
    return (Real) 0.45;
  }

  // in between values of 4.2 < log(T) < 8.15
  // linear interpolation of tabulated SPEX cooling rate

  int ipps  = static_cast<int>(25.0*logt) - 103;
  ipps = (ipps < 100)? ipps : 100;
  ipps = (ipps > 0 )? ipps : 0;

  Real d_ln_lambda_d_ln_temp = (lhd[ipps+1] - lhd[ipps])*25.0;
  return d_ln_lambda_d_ln_temp;
}

//----------------------------------------------------------------------------------------
//! \fn static Real SolveCubic(const Real a, const Real b, const Real c, const Real d)
//! \brief solve a cubic equation
//! \note
//! - input coeffs of eq. x^3 + b*x^2 + c*x + d = 0
//! - output greatest real solution to the cubic equation (due to Cardano 1545)
//! - reference https://mathworld.wolfram.com/CubicFormula.html

static Real SolveCubic(const Real b, const Real c, const Real d) {
  Real q,r,discriminant,s,t,res;
  Real theta,z1,z2,z3;
  // variables of use in solution Eqs. 22, 23 of reference
  q = (3*c - b*b)/9;
  r = (9*b*c - 27*d - 2*b*b*b)/54;
  // calculate the polynomial discriminant
  discriminant = q*q*q + r*r;
  // if the dsicriminant is positive there is one real root
  if (discriminant>0) {
    s = std::cbrt(r + std::sqrt(discriminant));
    t = std::cbrt(r - std::sqrt(discriminant));
    res =  s + t - b/3;
  } else {
  // if the discriminant is zero all roots are real and at least two are equal
  // if the discriminant is negative there are three, distinct, real roots
    theta = std::acos(r/std::sqrt( -1*q*q*q ));
    // calculate the three real roots
    z1 = 2*std::sqrt(-1*q)*std::cos(theta/3) - b/3;
    z2 = 2*std::sqrt(-1*q)*std::cos((theta + 2*M_PI)/3) - b/3;
    z3 = 2*std::sqrt(-1*q)*std::cos((theta + 4*M_PI)/3) - b/3;
    if ((z1>z2)&&(z1>z3))
      res = z1;
    else if(z2>z3)
      res = z2;
    else
      res = z3;
  }
  return res;
}

//----------------------------------------------------------------------------------------
//! \fn static Real ThermalInstabilityGrowthRate(const Real rho, const Real temp, 
//!       const Real k)
//! \brief growth rate of instability
//! \note
//! - input rho, pgas, and k are in code Units
//! - output growth rate of instability in code Units

static Real ThermalInstabilityGrowthRate(const Real rho, const Real pgas, const Real k) {
  Real gamma_adiabatic = 5.0/3.0;
  Real gm1 = gamma_adiabatic-1;
  // get Temperature in Kelvin
  Real temp = pgas/rho*units::punit->temperature;
  // number density in c.g.s
  Real n_hydrogen = rho*units::punit->density/units::punit->mu/
                    physical_constants::m_hydrogen;
  // pressure in c.g.s
  Real press = pgas*units::punit->pressure;
  // sounds spped in c.g.s
  Real cs = std::sqrt(gamma_adiabatic*pgas/rho)*units::punit->velocity;
  // krho in c.g.s
  Real k_rho = gm1*n_hydrogen*n_hydrogen*CoolFn(temp)/(press*cs);
  // krho in code units
  k_rho *= units::punit->length;
  // cs in code units 
  cs /= units::punit->velocity; 

  Real d_ln_lambda_d_ln_temp = DLnLambdaDLnT(rho,temp);
  // k_temp in code units based on derivative
  Real k_temp = k_rho*d_ln_lambda_d_ln_temp;
  // get coefficients in cubic dispersion relation
  Real b = cs*k_temp;
  Real c = cs*cs*k*k;
  Real d = cs*cs*cs*k*k*(k_temp - k_rho)/gamma_adiabatic;
  // solve  dispersion relation for the growth rate
  Real growth_rate = SolveCubic(b,c,d);
  return growth_rate;
}
