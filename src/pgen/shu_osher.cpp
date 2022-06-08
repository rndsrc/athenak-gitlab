//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shu_osher.cpp
//  \brief Problem generator for Shu-Osher shocktube test, involving interaction of a
//   Mach 3 shock with a sine wave density distribution.
//
// REFERENCE: C.W. Shu & S. Osher, "Efficient implementation of essentially
//   non-oscillatory shock-capturing schemes, II", JCP, 83, 32 (1998)

// C++ headers
#include <cmath>  // sin()
#include <iostream>
// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//  \brief Shu-Osher test problem generator

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Shu-Osher test can only be run in Hydro, but no <hydro> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // setup problem parameters
  Real dl = 3.857143;
  Real pl = 10.33333;
  Real ul = 2.629369;
  Real vl = 0.0;
  Real wl = 0.0;

  // capture variables for kernel
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  auto &u00 = pmbp->phydro->u00;

  std::cout << is << std::endl;

  par_for("pgen_shock1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is-1,ie+1,
  KOKKOS_LAMBDA(int m,int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    if (x1v < -0.8) {
      u00(m,IDN,k,j,i) = dl;
      u00(m,IM1,k,j,i) = ul*dl;
      u00(m,IM2,k,j,i) = vl*dl;
      u00(m,IM3,k,j,i) = wl*dl;
      u00(m,IEN,k,j,i) = pl/gm1 + 0.5*dl*(ul*ul + vl*vl + wl*wl);
    } else {
      u00(m,IDN,k,j,i) = 1.0 + 0.2*std::sin(5.0*M_PI*(x1v));
      u00(m,IM1,k,j,i) = 0.0;
      u00(m,IM2,k,j,i) = 0.0;
      u00(m,IM3,k,j,i) = 0.0;
      u00(m,IEN,k,j,i) = 1.0/gm1;
    }
  });

  RegionSize &size2 = pmbp->pmesh->mesh_size;
  //auto &h1 = size2.dx1;
  //auto &h2 = size.dx2;
  //auto &h3 = size.dx3;

  // fourth-order correction; change to par_for_outer and par_for_inner later to speed up.
  par_for("shu_osher fourth order", DevExeSpace(),0,(pmbp->nmb_thispack-1), 0, 4, ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i){
      Real h1 = size.h_view(m).dx1;
      Real C1 = (h1*h1)/24.0;
      u0(m,n,k,j,i) = u00(m,n,k,j,i) + C1*((u00(m,n,k,j,i-1)-2*u00(m,n,k,j,i)+u00(m,n,k,j,i+1))/(h1*h1));
  });
  return;
}
