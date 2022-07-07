//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_grid_test.cpp
//  \brief Tests the geodesic sphere and spherical implementation

#include <iostream>

#include "athena.hpp"

#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "pgen.hpp"

#include "Kokkos_Core.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem
//  \brief Problem Generator for geodesic grid test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int nmb1 = pmbp->nmb_thispack - 1;

  // geodesic mesh
  GeodesicGrid *pmy_geo = nullptr;
  int nlevel = pin->GetInteger("problem","nlevel");
  pmy_geo = new GeodesicGrid(nlevel, true, true);

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

  // check that unit flux lengths are equivalent after computed separately for each angle
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
  delete pmy_geo;

  // test mass flux
  auto w0_ = pmbp->phydro->w0;
  par_for("set_w0", DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
    Real vr = 1.0/SQR(rad);

    w0_(m,IDN,k,j,i) = 1.0;
    w0_(m,IEN,k,j,i) = 1.0;
    w0_(m,IVX,k,j,i) = vr*x1v/rad;
    w0_(m,IVY,k,j,i) = vr*x2v/rad;
    w0_(m,IVZ,k,j,i) = vr*x3v/rad;
  });

  SphericalGrid *pmy_sphere = nullptr;
  Real center[3] = {0.0};
  pmy_sphere = new SphericalGrid(pmbp, nlevel, center, true, true, 2.0);
  pmy_sphere->InterpToSphere(w0_);

  Real mass_flux = 0.0;
  for (int n=0; n<pmy_sphere->nangles; ++n) {
    Real idn = pmy_sphere->interp_vals.h_view(n,IDN);
    Real ivx = pmy_sphere->interp_vals.h_view(n,IVX);
    Real ivy = pmy_sphere->interp_vals.h_view(n,IVY);
    Real ivz = pmy_sphere->interp_vals.h_view(n,IVZ);
    Real theta = acos(pmy_sphere->cart_pos.h_view(n,2));
    Real phi = atan2(pmy_sphere->cart_pos.h_view(n,1), pmy_sphere->cart_pos.h_view(n,0));
    Real cosphi = cos(phi);
    Real sinphi = sin(phi);
    Real costheta = cos(theta);
    Real sintheta = sin(theta);
    Real ivr = ivx*cosphi*sintheta + ivy*sinphi*sintheta + ivz*costheta;
    mass_flux += pmy_sphere->area.h_view(n)*idn*ivr;
  }

  printf("|mass flux - analytic|/analytic: %24.16e\n", fabs(mass_flux-4.0*M_PI)/(4.0*M_PI));

  delete pmy_sphere;

  return;
}
