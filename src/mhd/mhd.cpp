//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd.cpp
//  \brief implementation of MHD class constructor and assorted functions

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "mhd/mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

MHD::MHD(MeshBlockPack *ppack, ParameterInput *pin) :
  pmy_pack(ppack),
  u0("cons",1,1,1,1,1),
  w0("prim",1,1,1,1,1),
  b0("B_fc",1,1,1,1),
  bcc0("B_cc",1,1,1,1,1),
  u1("cons1",1,1,1,1,1),
  b1("B_fc1",1,1,1,1),
  uflx("uflx",1,1,1,1,1),
  efld("efld",1,1,1,1),
  e3x1("e3x1",1,1,1,1),
  e2x1("e2x1",1,1,1,1),
  e1x2("e1x2",1,1,1,1),
  e3x2("e3x2",1,1,1,1),
  e2x3("e2x3",1,1,1,1),
  e1x3("e1x3",1,1,1,1),
  e1_cc("e1_cc",1,1,1,1),
  e2_cc("e2_cc",1,1,1,1),
  e3_cc("e3_cc",1,1,1,1)
{
  // (1) Start by selecting physics for this MHD:

  // construct EOS object (no default)
  {std::string eqn_of_state = pin->GetString("mhd","eos");
  if (eqn_of_state.compare("adiabatic") == 0) {

    // FIXME : Should this only be switched via the riemann solver flag?
    std::string rsolver = pin->GetString("mhd","rsolver");
    if (rsolver.compare("llf_rel") == 0){
        relativistic = true;
    	peos = new AdiabaticMHDRel(ppack, pin);
    }
    else{
    	peos = new AdiabaticMHD(ppack, pin);
    }
    nmhd = 5;
  } else if (eqn_of_state.compare("isothermal") == 0) {
    peos = new IsothermalMHD(ppack, pin);
    nmhd = 4;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<mhd> eos = '" << eqn_of_state << "' not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }}

  // Initialize number of scalars
  nscalars = pin->GetOrAddInteger("mhd","nscalars",0);

  // read time-evolution option [already error checked in driver constructor]
  std::string evolution_t = pin->GetString("time","evolution");

  // (2) Now initialize memory/algorithms

  // allocate memory for conserved and primitive variables
  int nmb = ppack->nmb_thispack;
  auto &ncells = ppack->mb_cells;
  int ncells1 = ncells.nx1 + 2*(ncells.ng);
  int ncells2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
  int ncells3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;
  Kokkos::realloc(u0,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
  Kokkos::realloc(w0,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);

  // allocate memory for face-centered and cell-centered magnetic fields
  Kokkos::realloc(bcc0,   nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(b0.x1f, nmb, ncells3, ncells2, ncells1+1);
  Kokkos::realloc(b0.x2f, nmb, ncells3, ncells2+1, ncells1);
  Kokkos::realloc(b0.x3f, nmb, ncells3+1, ncells2, ncells1);
  
  // allocate boundary buffers for conserved (cell-centered) variables
  pbval_u = new BoundaryValueCC(ppack, pin);
  pbval_u->AllocateBuffersCC((nmhd+nscalars));

  // allocate boundary buffers for face-centered magnetic field
  pbval_b = new BoundaryValueFC(ppack, pin);
  pbval_b->AllocateBuffersFC();

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("stationary") != 0) {

    // select reconstruction method (default PLM)
    {std::string xorder = pin->GetOrAddString("mhd","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method_ = ReconstructionMethod::dc;

    } else if (xorder.compare("plm") == 0) {
      recon_method_ = ReconstructionMethod::plm;

    } else if (xorder.compare("ppm") == 0) {
      // check that nghost > 2
      if (ncells.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "PPM reconstruction requires at least 3 ghost zones, "
            << "but <mesh>/nghost=" << ncells.ng << std::endl;
        std::exit(EXIT_FAILURE); 
      }                
      recon_method_ = ReconstructionMethod::ppm;

    } else if (xorder.compare("wenoz") == 0) {
      // check that nghost > 2
      if (ncells.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "WENOZ reconstruction requires at least 3 ghost zones, "
            << "but <mesh>/nghost=" << ncells.ng << std::endl;
        std::exit(EXIT_FAILURE); 
      }                
      recon_method_ = ReconstructionMethod::wenoz;

    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }}

    // select Riemann solver (no default).  Test for compatibility of options
    {std::string rsolver = pin->GetString("mhd","rsolver");
    if (rsolver.compare("advect") == 0) {
      if (evolution_t.compare("dynamic") == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/rsolver = '" << rsolver
                  << "' cannot be used with dynamic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      } else {
        rsolver_method_ = MHD_RSolver::advect;
      }

    } else  if (evolution_t.compare("dynamic") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/rsolver = '" << rsolver
                << "' cannot be used with non-dynamic problems" << std::endl;
      std::exit(EXIT_FAILURE);

    } else if (rsolver.compare("llf") == 0) {
      rsolver_method_ = MHD_RSolver::llf;

    } else if (rsolver.compare("llf_rel") == 0) {
      rsolver_method_ = MHD_RSolver::llf_rel;

    } else if (rsolver.compare("hlld") == 0) {
      if (peos->eos_data.is_adiabatic) {
        rsolver_method_ = MHD_RSolver::hlld;
      } else { 
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/rsolver = '" << rsolver
                  << "' cannot be used with isothermal EOS" << std::endl;
        std::exit(EXIT_FAILURE); 
      }  

//    } else if (rsolver.compare("roe") == 0) {
//      rsolver_method_ = MHD_RSolver::roe;

    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/rsolver = '" << rsolver << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE); 
    }}

    // allocate second registers
    Kokkos::realloc(u1,     nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(b1.x1f, nmb, ncells3, ncells2, ncells1+1);
    Kokkos::realloc(b1.x2f, nmb, ncells3, ncells2+1, ncells1);
    Kokkos::realloc(b1.x3f, nmb, ncells3+1, ncells2, ncells1);

    // allocate fluxes, electric fields
    Kokkos::realloc(uflx.x1f, nmb, (nmhd+nscalars), ncells3, ncells2, ncells1+1);
    Kokkos::realloc(uflx.x2f, nmb, (nmhd+nscalars), ncells3, ncells2+1, ncells1);
    Kokkos::realloc(uflx.x3f, nmb, (nmhd+nscalars), ncells3+1, ncells2, ncells1);
    Kokkos::realloc(efld.x1e, nmb, ncells3+1, ncells2+1, ncells1);
    Kokkos::realloc(efld.x2e, nmb, ncells3+1, ncells2, ncells1+1);
    Kokkos::realloc(efld.x3e, nmb, ncells3, ncells2+1, ncells1+1);

    // allocate scratch arrays for face- and cell-centered E used in CornerE
    Kokkos::realloc(e3x1, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e2x1, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e1x2, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e3x2, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e2x3, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e1x3, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e1_cc, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e2_cc, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e3_cc, nmb, ncells3, ncells2, ncells1);
  }
}

//----------------------------------------------------------------------------------------
// destructor
  
MHD::~MHD()
{
  delete peos;
  delete pbval_u;
  delete pbval_b;
}

} // namespace mhd
