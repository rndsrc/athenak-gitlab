//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cooling.cpp
//  \brief Problem generator for cooling

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
//#include "mesh/mesh_positions.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Problem Generator for cooling

void ProblemGenerator::UserProblem(MeshBlockPack *pmbp, ParameterInput *pin)
{
  if (pmbp->phydro == nullptr and pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Cooling problem generator can only be run with Hydro and/or MHD, but no "
       << "<hydro> or <mhd> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  Real cs = pin->GetOrAddReal("eos","iso_sound_speed",1.0);
  Real dens = pin->GetOrAddReal("problem","dens",1.0);

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    auto &u0 = pmbp->phydro->u0;
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = dens*cs*cs/eos.gamma;

    // Set initial conditions
    par_for("pgen_cooling", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i)
      {
        u0(m,IDN,k,j,i) = dens;
        u0(m,IM1,k,j,i) = 0.0;
        u0(m,IM2,k,j,i) = 0.0;
        u0(m,IM3,k,j,i) = 0.0;
        if (eos.is_ideal) {
          u0(m,IEN,k,j,i) = p0/gm1;
        }
      }
    );
  }

  return;
}

