//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geogrid_test.cpp
//  \brief Tests the geodesic sphere implementation

#include <iostream>

#include "athena.hpp"

#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "radiation/radiation.hpp"
#include "pgen.hpp"

#include "Kokkos_Core.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem
//  \brief Problem Generator for geodesic grid test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &nmb = pmbp->nmb_thispack;

  bool is_radiation_enabled; int nlevel;
  if (pmbp->prad != nullptr) {
    is_radiation_enabled = true;
    nlevel = pin->GetInteger("radiation","nlevel");
  } else {
    is_radiation_enabled = false;
    nlevel = pin->GetInteger("problem","nlevel");
  }

  // Construct geodesic mesh
  GeodesicGrid *pmy_geo = nullptr;
  pmy_geo = new GeodesicGrid(nlevel, true, true, 1.0);

  // print number of angles
  printf("\nnangles: %d\n", pmy_geo->nangles);

  // guarantee sum of solid angles is 4 pi
  Real sum_angles = 0.0;
  int nang1 = pmy_geo->nangles - 1;
  auto &solid_angles_ = pmy_geo->solid_angles;
  for (int n=0; n<=nang1; ++n) {
    sum_angles += solid_angles_.h_view(n);
  }
  printf("|sum_angles-four_pi|/four_pi: %24.16e\n\n",
         fabs(sum_angles-4.0*M_PI)/(4.0*M_PI));

  // check that arc lengths are equivalent after computed separately for each angle
  auto posm = pmy_geo->cart_pos_mid;
  auto numn = pmy_geo->num_neighbors;
  auto indn = pmy_geo->ind_neighbors;
  auto arcl = pmy_geo->arc_lengths;
  for (int n=0; n<=nang1; ++n) {
    bool not_equal = false;
    for (int nb=0; nb<numn.h_view(n); ++nb) {
      Real this_arc = arcl.h_view(n,nb);
      bool match_not_found = true;
      for (int nnb=0; nnb<numn.h_view(indn.h_view(n,nb)); ++nnb) {
        Real neigh_arc = arcl.h_view(indn.h_view(n,nb),nnb);
        if (posm.h_view(n,nb,0) == posm.h_view(indn.h_view(n,nb),nnb,0) &&
            posm.h_view(n,nb,1) == posm.h_view(indn.h_view(n,nb),nnb,1) &&
            posm.h_view(n,nb,2) == posm.h_view(indn.h_view(n,nb),nnb,2)) {
          if (this_arc != neigh_arc) { not_equal = true; }
          match_not_found = false;
        }
      }
      if (not_equal || match_not_found) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Error in geodesic grid arc lengths" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  // check that arc lengths are equivalent after computed separately for each angle
  auto uvec = pmy_geo->unit_flux;
  for (int n=0; n<=nang1; ++n) {
    bool not_equal = false;
    for (int nb=0; nb<numn.h_view(n); ++nb) {
      Real this_uzeta = uvec.h_view(n,nb,0);
      Real this_upsi  = uvec.h_view(n,nb,1);
      bool match_not_found = true;
      for (int nnb=0; nnb<numn.h_view(indn.h_view(n,nb)); ++nnb) {
        Real neigh_uzeta = uvec.h_view(indn.h_view(n,nb),nnb,0);
        Real neigh_upsi  = uvec.h_view(indn.h_view(n,nb),nnb,1);
        if (posm.h_view(n,nb,0) == posm.h_view(indn.h_view(n,nb),nnb,0) &&
            posm.h_view(n,nb,1) == posm.h_view(indn.h_view(n,nb),nnb,1) &&
            posm.h_view(n,nb,2) == posm.h_view(indn.h_view(n,nb),nnb,2)) {
          if (this_uzeta != -1*neigh_uzeta) { not_equal = true; }
          if (this_upsi != -1*neigh_upsi) { not_equal = true; }
          match_not_found = false;
        }
      }
      if (not_equal || match_not_found) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Error in geodesic grid arc lengths" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  if (is_radiation_enabled) {
    auto nh_f_ = pmbp->prad->nh_f;
    auto na_ = pmbp->prad->na;
    par_for("rad_na_n_0",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Guarantee that n^angle at shared faces are identical
      for (int n=0; n<=nang1; ++n) {
        bool not_equal = false;
        for (int nb=0; nb<numn.d_view(n); ++nb) {
          Real this_na = na_(m,n,k,j,i,nb);
          bool match_not_found = true;
          for (int nnb=0; nnb<numn.d_view(indn.d_view(n,nb)); ++nnb) {
            Real neigh_na = na_(m,indn.d_view(n,nb),k,j,i,nnb);
            if (nh_f_.d_view(n,nb,1) == nh_f_.d_view(indn.d_view(n,nb),nnb,1) &&
                nh_f_.d_view(n,nb,2) == nh_f_.d_view(indn.d_view(n,nb),nnb,2) &&
                nh_f_.d_view(n,nb,3) == nh_f_.d_view(indn.d_view(n,nb),nnb,3)) {
              if (this_na != -1*neigh_na) { not_equal = true; }
              match_not_found = false;
            }
          }
        if (not_equal || match_not_found) {
          Kokkos::abort("Error in radiation initialization\n");
        }
        }
      }
    });
  }

  delete pmy_geo;
  return;
}
