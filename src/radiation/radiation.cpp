//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation.cpp
//  \brief implementation of functions in class Radiation

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "radiation/radiation.hpp"

namespace radiation {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Radiation::Radiation(MeshBlockPack *ppack, ParameterInput *pin) :
  pmy_pack(ppack),

  // intensities and fluxes
  i0("prim",1,1,1,1,1),
  i1("cons1",1,1,1,1,1),
  iflx("ciflx",1,1,1,1,1),
  ia1flx("cia1flx",1,1,1,1,1),
  ia2flx("cia2flx",1,1,1,1,1),

  // angle and lengths...yuck...(TODO: @pdmullen) be smarter about this
  zetaf("zetaf",1),
  zetav("zetav",1),
  dzetaf("dzetaf",1),
  psif("psif",1),
  psiv("psiv",1),
  dpsif("dpsif",1),
  zeta_length("zlen",1,1),
  psi_length("plen",1,1),
  solid_angle("solidang",1,1),

  // coordinate frame quantities...yuck...(TODO: @pdmullen) be smarter about this
  nh_cc("nh_cc",1,1,1),
  nh_fc("nh_fc",1,1,1),
  nh_cf("nh_cf",1,1,1),
  n0("n0",1,1,1,1,1,1),
  n0_n_0("n0_n_0",1,1,1,1,1,1),
  n1_n_0("n1_n_0",1,1,1,1,1,1),
  n2_n_0("n2_n_0",1,1,1,1,1,1),
  n3_n_0("n3_n_0",1,1,1,1,1,1),
  na1_n_0("na1_n_0",1,1,1,1,1,1),
  na2_n_0("na2_n_0",1,1,1,1,1,1),

  // moments of the radiation field
  moments_coord("moments",1,1,1,1,1)
{
  // Check for hydrodynamics or mhd
  is_hydro_enabled = pin->DoesBlockExist("hydro");
  is_mhd_enabled = pin->DoesBlockExist("mhd");

  // initialize metric (GR only)
  pmy_pack->coord.InitMetric(pin);

  // no-op, passed to boundary functions
  peos = new EquationOfState(ppack, pin);

  // Source terms (constructor parses input file to initialize only srcterms needed)
  psrc = new SourceTerms("radiation", ppack, pin);

  // read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");

  // allocate memory for conserved and primitive variables,
  // angles, and coordinate frame data
  amesh_indcs.nzeta = pin->GetInteger("radiation", "nzeta");
  amesh_indcs.npsi = pin->GetInteger("radiation", "npsi");

  int nmb = ppack->nmb_thispack;
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  amesh_indcs.ng = indcs.ng;
  int ncellsa1 = amesh_indcs.nzeta + 2*(amesh_indcs.ng);
  int ncellsa2 = amesh_indcs.npsi + 2*(amesh_indcs.ng);
  nangles = ncellsa1*ncellsa2;

  amesh_indcs.zs = amesh_indcs.ng;
  amesh_indcs.ze = amesh_indcs.nzeta + amesh_indcs.ng - 1;
  amesh_indcs.ps = amesh_indcs.ng;
  amesh_indcs.pe = amesh_indcs.npsi + amesh_indcs.ng - 1;

  Kokkos::realloc(i0,nmb,nangles,ncells3,ncells2,ncells1);
  Kokkos::realloc(moments_coord,nmb,1,ncells3,ncells2,ncells1);

  // Setup angular mesh and coordinate frame quantities
  Kokkos::realloc(zetaf,ncellsa1+1);
  Kokkos::realloc(zetav,ncellsa1);
  Kokkos::realloc(dzetaf,ncellsa1);
  Kokkos::realloc(psif,ncellsa2+1);
  Kokkos::realloc(psiv,ncellsa2);
  Kokkos::realloc(dpsif,ncellsa2);
  Kokkos::realloc(zeta_length,ncellsa1,ncellsa2+1);
  Kokkos::realloc(psi_length,ncellsa1+1,ncellsa2);
  Kokkos::realloc(solid_angle,ncellsa1,ncellsa2);
  Kokkos::realloc(nh_cc,4,ncellsa1,ncellsa2);
  Kokkos::realloc(nh_fc,4,ncellsa1+1,ncellsa2);
  Kokkos::realloc(nh_cf,4,ncellsa1,ncellsa2+1);
  Kokkos::realloc(n0,nmb,ncellsa1,ncellsa2,ncells3,ncells2,ncells1);
  Kokkos::realloc(n0_n_0,nmb,ncellsa1,ncellsa2,ncells3,ncells2,ncells1);
  Kokkos::realloc(n1_n_0,nmb,ncellsa1,ncellsa2,ncells3,ncells2,ncells1+1);
  Kokkos::realloc(n2_n_0,nmb,ncellsa1,ncellsa2,ncells3,ncells2+1,ncells1);
  Kokkos::realloc(n3_n_0,nmb,ncellsa1,ncellsa2,ncells3+1,ncells2,ncells1);
  Kokkos::realloc(na1_n_0,nmb,ncellsa1+1,ncellsa2,ncells3,ncells2,ncells1);
  Kokkos::realloc(na2_n_0,nmb,ncellsa1,ncellsa2+1,ncells3,ncells2,ncells1);
  InitAngularMesh();
  InitCoordinateFrame();

  // allocate boundary buffers for conserved (cell-centered) variables
  pbval_ci = new BoundaryValueCC(ppack, pin);
  pbval_ci->AllocateBuffersCC(nangles);

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("static") != 0) {
    // select reconstruction method (default PLM)
    {std::string xorder = pin->GetOrAddString("radiation","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method = ReconstructionMethod::dc;
    } else if (xorder.compare("plm") == 0) {
      recon_method = ReconstructionMethod::plm;
    } else if (xorder.compare("ppm") == 0) {
      // check that nghost > 2
      if (indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "PPM reconstruction requires at least 3 ghost zones, "
          << "but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE); 
      }                
      recon_method = ReconstructionMethod::ppm;
    } else if (xorder.compare("wenoz") == 0) {
      // check that nghost > 2
      if (indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "WENOZ reconstruction requires at least 3 ghost zones, "
            << "but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE); 
      }                
      recon_method = ReconstructionMethod::wenoz;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<radiation> recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }}

    // allocate second registers, fluxes
    Kokkos::realloc(i1,nmb,nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x1f,nmb,nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x2f,nmb,nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x3f,nmb,nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(ia1flx,nmb,((ncellsa1+1)*ncellsa2),ncells3,ncells2,ncells1);
    Kokkos::realloc(ia2flx,nmb,(ncellsa1*(ncellsa2+1)),ncells3,ncells2,ncells1);
  }

}

//----------------------------------------------------------------------------------------
// destructor
  
Radiation::~Radiation()
{
  delete peos;
  delete psrc;
  delete pbval_ci;
}

} // namespace radiation
