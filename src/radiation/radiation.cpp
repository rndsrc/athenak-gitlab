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

void NoOpOpacity(const Real rho, const Real temp,
                 Real& kappa_a, Real& kappa_s, Real& kappa_p);

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Radiation::Radiation(MeshBlockPack *ppack, ParameterInput *pin) :
  pmy_pack(ppack),

  // intensities and fluxes
  i0("i0",1,1,1,1,1),
  i1("i1",1,1,1,1,1),
  iflx("ciflx",1,1,1,1,1),
  iaflx("ciaflx",1,1,1,1,1,1),

  // TODO(@gnwong, @pdmullen) in the future, it might make sense to get rid
  // of some of these and compute them on the fly
  nh_c("nh_c",1,1),
  nh_f("nh_f",1,1,1),
  xi_mn("xi_mn",1,1),
  eta_mn("eta_mn",1,1),
  arc_lengths("arclen",1,1),
  solid_angle("solidang",1),

  amesh_normals("ameshnorm",1,1,1,1),
  ameshp_normals("ameshpnorm",1,1),
  amesh_indices("ameshind",1,1,1),
  ameshp_indices("ameshpind",1),

  num_neighbors("numneigh",1),
  ind_neighbors("indneigh",1,1),

  // coordinate frame quantities
  nmu("nmu",1,1,1,1,1,1),
  n_mu("n_mu",1,1,1,1,1,1),
  n1_n_0("n1_n_0",1,1,1,1,1),
  n2_n_0("n2_n_0",1,1,1,1,1),
  n3_n_0("n3_n_0",1,1,1,1,1),
  na_n_0("na_n_0",1,1,1,1,1,1),
  norm_to_tet("norm_to_tet",1,1,1,1,1,1),

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
  if (psrc->rad_source && not is_hydro_enabled) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "Radiation source term requires phydro, but "
      << "<hydro> block does not exist" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (psrc->rad_source) {
    OpacityFunc = NoOpOpacity;
  }

  // read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");

  // allocate memory for conserved and primitive variables,
  // angles, and coordinate frame data
  nlevels = pin->GetInteger("radiation", "nlevel");
  nangles = 5 * 2*SQR(nlevels) + 2;
  rotate_geo = pin->GetOrAddBoolean("radiation", "rotate_geo", true);

  int nmb = ppack->nmb_thispack;
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  Kokkos::realloc(i0,nmb,nangles,ncells3,ncells2,ncells1);
  Kokkos::realloc(moments_coord,nmb,10,ncells3,ncells2,ncells1);

  // Setup angular mesh and coordinate frame quantities
  Kokkos::realloc(solid_angle, nangles);

  Kokkos::realloc(amesh_normals, 5, 2+nlevels, 2+2*nlevels, 3);
  Kokkos::realloc(ameshp_normals, 2, 3);

  Kokkos::realloc(amesh_indices, 5, 2+nlevels, 2+2*nlevels);
  Kokkos::realloc(ameshp_indices, 2);

  Kokkos::realloc(num_neighbors, nangles);
  Kokkos::realloc(ind_neighbors, nangles, 6);
  Kokkos::realloc(arc_lengths, nangles, 6);

  Kokkos::realloc(nh_c, nangles, 4);
  Kokkos::realloc(nh_f, nangles, 6, 4);
  Kokkos::realloc(xi_mn, nangles, 6);
  Kokkos::realloc(eta_mn, nangles, 6);

  Kokkos::realloc(nmu, nmb, nangles, ncells3, ncells2, ncells1, 4);
  Kokkos::realloc(n_mu, nmb, nangles, ncells3, ncells2, ncells1, 4);
  Kokkos::realloc(n1_n_0, nmb, nangles, ncells3, ncells2, ncells1+1);
  Kokkos::realloc(n2_n_0, nmb, nangles, ncells3, ncells2+1, ncells1);
  Kokkos::realloc(n3_n_0, nmb, nangles, ncells3+1, ncells2, ncells1);
  Kokkos::realloc(na_n_0, nmb, nangles, ncells3, ncells2, ncells1, 6);
  Kokkos::realloc(norm_to_tet, nmb, 4, 4, ncells3, ncells2, ncells1);

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
    Kokkos::realloc(iaflx,nmb,nangles,ncells3,ncells2,ncells1,6);
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

//----------------------------------------------------------------------------------------
//! \fn EnrollOpacityFunction(BValFunc my_bc)
//! \brief Enroll a user-defined boundary function

void Radiation::EnrollOpacityFunction(OpacityFnPtr my_opacityfunc) {
  OpacityFunc = my_opacityfunc;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn NoOpOpacity
//! \brief

void NoOpOpacity(const Real rho, const Real temp,
                 Real& kappa_a, Real& kappa_s, Real& kappa_p)
{
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
    << std::endl << "Radiation source term requires specifying opacity function, but "
    << "no UserOpacityFunction enrolled" << std::endl;
  std::exit(EXIT_FAILURE);
  return;
}

} // namespace radiation
