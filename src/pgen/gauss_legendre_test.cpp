//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_one_puncture.cpp
//  \brief Problem generator for a single puncture placed at the origin of the domain

#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>
#include <cmath>  // sin()

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"
#include "geodesic-grid/gauss_legendre.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator to test Strahlkorper grid and derivatives

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Strahlkorper test is set up to run in Hydro, but no <hydro> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // setup problem parameters

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;

  par_for("pgen_test1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m,int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    int nx2 = indcs.nx2;
    int nx1 = indcs.nx1;

    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    // test the InterpToSphere using random 3d sin waves
    u0(m,IDN,k,j,i) = 1.0 + 0.2*std::sin(5.0*M_PI*(x1v))*std::sin(3.0*M_PI*(x2v))*std::sin(3.0*M_PI*(x3v));
  });


  int ntheta = 20;
  bool rotate_sphere = true;
  bool fluxes = true;
  int nfilt = 16;
  Real radius = 0.2;
  GaussLegendreGrid *S = nullptr;
  S = new GaussLegendreGrid(pmbp, ntheta, radius,nfilt);

  Real center[3] = {0.0};
  Real ctr[3] = {0,0,0};

  // check grid
  std::ofstream spherical_grid_output;
  spherical_grid_output.open ("/Users/hawking/Desktop/research/gr/athenak_versions/athenak_z4c_horizon/gauss_legendre_test/polar_pos.out", std::ios_base::app);
  for (int i=0;i<S->nangles;++i) {
    spherical_grid_output << S->polar_pos.h_view(i,0) << "\t" << S->polar_pos.h_view(i,1) << "\t" << S->int_weights.h_view(i) <<  "\n";// << ones_dphi.h_view(i) <<"\n";
  }
  spherical_grid_output.close();

  // check Cartesain coords
  spherical_grid_output.open ("/Users/hawking/Desktop/research/gr/athenak_versions/athenak_z4c_horizon/gauss_legendre_test/cart_pos.out", std::ios_base::app);
  for (int i=0;i<S->nangles;++i) {
    spherical_grid_output << S->cart_pos.h_view(i,0) << "\t" << S->cart_pos.h_view(i,1) << "\t" << S->cart_pos.h_view(i,2) <<  "\n";// << ones_dphi.h_view(i) <<"\n";
  }
  spherical_grid_output.close();
  
  // check differentiation
  DualArray1D<Real> ones;
  Kokkos::realloc(ones,S->nangles);
  DualArray1D<Real> sintheta;
  Kokkos::realloc(sintheta,S->nangles);

  DualArray1D<Real> sinphi;
  Kokkos::realloc(sinphi,S->nangles);
  for(int n=0; n<S->nangles; ++n) {
    ones.h_view(n) = 1.;
    Real &theta = S->polar_pos.h_view(n,0);
    Real &phi = S->polar_pos.h_view(n,1);
    sintheta.h_view(n) = sin(theta);
    sinphi.h_view(n) = sin(theta)*sin(phi);
  }

  auto one_dtheta = S->ThetaDerivative(ones);
  auto one_dphi = S->PhiDerivative(ones);

  auto sinth_dtheta = S->ThetaDerivative(sintheta);
  auto sinth_dphi = S->PhiDerivative(sintheta);

  auto sinph_dtheta = S->ThetaDerivative(sinphi);
  auto sinph_dphi = S->PhiDerivative(sinphi);

  spherical_grid_output.open ("/Users/hawking/Desktop/research/gr/athenak_versions/athenak_z4c_horizon/gauss_legendre_test/one_deriv.out", std::ios_base::app);
  for (int i=0;i<S->nangles;++i) {
    spherical_grid_output << one_dtheta.h_view(i) << "\t" << one_dphi.h_view(i) <<  "\n";// << ones_dphi.h_view(i) <<"\n";
  }
  spherical_grid_output.close();

  spherical_grid_output.open ("/Users/hawking/Desktop/research/gr/athenak_versions/athenak_z4c_horizon/gauss_legendre_test/sinth_deriv.out", std::ios_base::app);
  for (int i=0;i<S->nangles;++i) {
    spherical_grid_output << sinth_dtheta.h_view(i) << "\t" << sinth_dphi.h_view(i) <<  "\n";// << ones_dphi.h_view(i) <<"\n";
  }
  spherical_grid_output.close();

  spherical_grid_output.open ("/Users/hawking/Desktop/research/gr/athenak_versions/athenak_z4c_horizon/gauss_legendre_test/sinph_deriv.out", std::ios_base::app);
  for (int i=0;i<S->nangles;++i) {
    spherical_grid_output << sinph_dtheta.h_view(i) << "\t" << sinph_dphi.h_view(i) <<  "\n";// << ones_dphi.h_view(i) <<"\n";
  }
  spherical_grid_output.close();

  // Everything seems to be working so far!

  // Check reset radius
  DualArray1D<Real> rad_tmp;
  Kokkos::realloc(rad_tmp,S->nangles);
  for (int n=0; n<S->nangles; ++n) {
    Real &theta = S->polar_pos.h_view(n,0);
    Real &phi = S->polar_pos.h_view(n,1);
    rad_tmp.h_view(n) = .2 + 0.01*sin(theta);
  }
  S->SetPointwiseRadius(rad_tmp,ctr);

  spherical_grid_output.open ("/Users/hawking/Desktop/research/gr/athenak_versions/athenak_z4c_horizon/gauss_legendre_test/cart_pos_reseted.out", std::ios_base::app);
  for (int i=0;i<S->nangles;++i) {
    spherical_grid_output << S->cart_pos.h_view(i,0) << "\t" << S->cart_pos.h_view(i,1) << "\t" << S->cart_pos.h_view(i,2) <<  "\n";// << ones_dphi.h_view(i) <<"\n";
  }
  spherical_grid_output.close();


  // check interpolation (working flawlessly! Will test tensor_interp in a separate file)
  S->InterpolateToSphere(1,u0);

  DualArray1D<Real> analytic_value;
  Kokkos::realloc(analytic_value,S->nangles);

  for (int n=0; n<S->nangles; ++n) {
    Real &x = S->cart_pos.h_view(n,0);
    Real &y = S->cart_pos.h_view(n,1);
    Real &z = S->cart_pos.h_view(n,2);
    analytic_value.h_view(n) = 1.0 + 0.2*std::sin(5.0*M_PI*(x))*std::sin(3.0*M_PI*(y))*std::sin(3.0*M_PI*(z));
  }


  spherical_grid_output.open ("/Users/hawking/Desktop/research/gr/athenak_versions/athenak_z4c_horizon/gauss_legendre_test/interp_res.out", std::ios_base::app);
  for (int i=0;i<S->nangles;++i) {
    spherical_grid_output << S->interp_vals.h_view(i,0) - analytic_value.h_view(i) <<  "\n";// << ones_dphi.h_view(i) <<"\n";
  }
  spherical_grid_output.close();

  // Jacobian matrix


  return;
}
