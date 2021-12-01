//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file therm_instab_nonlinear.cpp
//! \brief Problem generator for nonlinear thermal instability

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


//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Problem Generator for nonlinear thermal instability

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

  // Find the equilibrium point of the cooling curve by n*Lambda-Gamma=0
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

    // Set initial conditions
    par_for("pgen_thermal_instability", DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i)
      {
        u0(m,IDN,k,j,i) = rho_0;
        u0(m,IM1,k,j,i) = 0.0;
        u0(m,IM2,k,j,i) = 0.0;
        u0(m,IM3,k,j,i) = 0.0;
        if (eos.is_ideal) {
          u0(m,IEN,k,j,i) = pgas_0/gm1;
        }
      }
    );    
  }

  return;
}
