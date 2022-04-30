//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.cpp
//! \brief implementation of Hydro class constructor and assorted other functions

#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/coordinates.hpp"
#include "radiation/radiation.hpp"

namespace radiation {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Radiation::Radiation(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack),
    i0("i0",1,1,1,1,1),
    i1("i1",1,1,1,1,1),
    iflx("ciflx",1,1,1,1,1),
    ia1flx("ciaflx",1,1,1,1,1),
    ia2flx("ciaflx",1,1,1,1,1),
    zetaf("zetaf",1),
    zetav("zetav",1),
    dzetaf("dzetaf",1),
    psif("psif",1),
    psiv("psiv",1),
    dpsif("dpsif",1),
    zeta_length("zlen",1,1),
    psi_length("plen",1,1),
    solid_angle("solidang",1,1),
    nh_c("nh_c",1,1,1),
    nh_zf("nh_zf",1,1,1),
    nh_pf("nh_pf",1,1,1),
    tet_c("tet_c",1,1,1,1,1,1),
    tetcov_c("tetcov_c",1,1,1,1,1,1),
    tet_d1_x1f("tet_d1_x1f",1,1,1,1,1),
    tetcov_d0_x1f("tetcov_d0_x1f",1,1,1,1,1),
    tet_d2_x2f("tet_d2_x2f",1,1,1,1,1),
    tetcov_d0_x2f("tetcov_d0_x2f",1,1,1,1,1),
    tet_d3_x3f("tet_d3_x3f",1,1,1,1,1),
    tetcov_d0_x3f("tetcov_d0_x3f",1,1,1,1,1),
    ricci("ricci",1,1,1,1,1,1,1),
    norm_to_tet("norm_to_tet",1,1,1,1,1,1) {
  // Check for general relativity
  if (!(pmy_pack->pcoord->is_general_relativistic)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "Radiation requires general relativity" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Check for hydrodynamics or mhd
  is_hydro_enabled = pin->DoesBlockExist("hydro");
  is_mhd_enabled = pin->DoesBlockExist("mhd");
  if (is_hydro_enabled && is_mhd_enabled) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "Radiation does not support two fluid calculations, yet "
      << "both <hydro> and <mhd> blocks exist in input file" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Radiation source term (radiation+(M)HD) and opacities
  rad_source = pin->GetOrAddBoolean("radiation","rad_source",true);
  if (rad_source && (!is_hydro_enabled && !is_mhd_enabled)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "Radiation source term requires hydro or mhd, but "
      << "neither <hydro> nor <mhd> block exist" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  fixed_fluid = pin->GetOrAddBoolean("radiation","fixed_fluid",false);
  affect_fluid = pin->GetOrAddBoolean("radiation","affect_fluid",true);
  zero_radiation_force = pin->GetOrAddBoolean("radiation","zero_radiation_force",false);
  arad = pin->GetOrAddReal("radiation","arad",1.0);
  kappa_a = pin->GetOrAddReal("radiation","kappa_a",0.0);
  kappa_s = pin->GetOrAddReal("radiation","kappa_s",0.0);
  kappa_p = pin->GetOrAddReal("radiation","kappa_p",0.0);
  constant_opacity = pin->GetOrAddBoolean("radiation","constant_opacity",true);
  power_opacity = pin->GetOrAddBoolean("radiation","power_opacity",false);
  if (constant_opacity==power_opacity) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "One formulation for opacities must be selected, yet "
      << "constant_opacity and power_opacity booleans are the same" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Other rad source terms (constructor parses input file to init only srcterms needed)
  psrc = new SourceTerms("radiation", ppack, pin);

  // Setup angular mesh and radiation frame data
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  amesh_indcs.ng = 1;
  amesh_indcs.nzeta = pin->GetInteger("radiation", "nzeta");
  amesh_indcs.npsi = pin->GetInteger("radiation", "npsi");
  amesh_indcs.zs = amesh_indcs.ng;
  amesh_indcs.ze = amesh_indcs.nzeta + amesh_indcs.ng - 1;
  amesh_indcs.ps = amesh_indcs.ng;
  amesh_indcs.pe = amesh_indcs.npsi + amesh_indcs.ng - 1;
  angular_fluxes = pin->GetOrAddBoolean("radiation","angular_fluxes",true);

  int nmb = ppack->nmb_thispack;
  int ncellsa1 = amesh_indcs.nzeta + 2*(amesh_indcs.ng);
  int ncellsa2 = amesh_indcs.npsi + 2*(amesh_indcs.ng);
  nangles = ncellsa1*ncellsa2;
  {
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(zetaf,(ncellsa1+1));
  Kokkos::realloc(zetav,ncellsa1);
  Kokkos::realloc(dzetaf,ncellsa1);
  Kokkos::realloc(psif,(ncellsa2+1));
  Kokkos::realloc(psiv,ncellsa2);
  Kokkos::realloc(dpsif,ncellsa2);
  Kokkos::realloc(zeta_length,ncellsa1,(ncellsa2+1));
  Kokkos::realloc(psi_length,(ncellsa1+1),ncellsa2);
  Kokkos::realloc(solid_angle,ncellsa1,ncellsa2);
  Kokkos::realloc(nh_c,ncellsa1,ncellsa2,4);
  Kokkos::realloc(nh_zf,(ncellsa1+1),ncellsa2,4);
  Kokkos::realloc(nh_pf,ncellsa1,(ncellsa2+1),4);
  Kokkos::realloc(tet_c,nmb,4,4,ncells3,ncells2,ncells1);
  Kokkos::realloc(tetcov_c,nmb,4,4,ncells3,ncells2,ncells1);
  Kokkos::realloc(tet_d1_x1f,nmb,4,ncells3,ncells2,ncells1+1);
  Kokkos::realloc(tetcov_d0_x1f,nmb,4,ncells3,ncells2,ncells1+1);
  Kokkos::realloc(tet_d2_x2f,nmb,4,ncells3,ncells2+1,ncells1);
  Kokkos::realloc(tetcov_d0_x2f,nmb,4,ncells3,ncells2+1,ncells1);
  Kokkos::realloc(tet_d3_x3f,nmb,4,ncells3+1,ncells2,ncells1);
  Kokkos::realloc(tetcov_d0_x3f,nmb,4,ncells3+1,ncells2,ncells1);
  Kokkos::realloc(ricci,nmb,4,4,4,ncells3,ncells2,ncells1);
  Kokkos::realloc(norm_to_tet,nmb,4,4,ncells3,ncells2,ncells1);
  }
  InitAngularMesh();
  SetOrthonormalTetrad();

  // (3) read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");

  // allocate memory for intensities
  {
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(i0,nmb,nangles,ncells3,ncells2,ncells1);
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_i0,nmb,nangles,nccells3,nccells2,nccells1);
  }

  // allocate boundary buffers for conserved (cell-centered) variables
  pbval_i = new BoundaryValuesCC(ppack, pin);
  pbval_i->InitializeBuffers(nangles);

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("stationary") != 0) {
    // select reconstruction method (default PLM)
    {std::string xorder = pin->GetOrAddString("hydro","reconstruct","plm");
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
                << std::endl << "<hydro> recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    }

    // allocate second registers, fluxes, moments
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(i1,      nmb,nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x1f,nmb,nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x2f,nmb,nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x3f,nmb,nangles,ncells3,ncells2,ncells1);
    if (angular_fluxes) {
      Kokkos::realloc(ia1flx,nmb,((ncellsa1+1)*ncellsa2),ncells3,ncells2,ncells1);
      Kokkos::realloc(ia2flx,nmb,(ncellsa1*(ncellsa2+1)),ncells3,ncells2,ncells1);
    }
  }
}

//----------------------------------------------------------------------------------------
// destructor

Radiation::~Radiation() {
  delete pbval_i;
  if (psrc != nullptr) {delete psrc;}
}

} // namespace radiation
