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
#include "geodesic-grid/spherical_grid.hpp"
#include "geodesic-grid/strahlkorper.hpp"

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

  int nlev = 10;
  bool rotate_sphere = true;
  bool fluxes = true;

  Strahlkorper *S = nullptr;
  Real center[3] = {0.0};
  S = new Strahlkorper(pmbp, nlev, .2);
  Real ctr[3] = {0,0,0};
  DualArray1D<Real> rad_tmp;
  int nangles = S->nangles;
  Kokkos::realloc(rad_tmp,nangles);

  for (int n=0; n<nangles; ++n) {
    Real &theta = S->polar_pos.h_view(n,0);
    Real &phi = S->polar_pos.h_view(n,1);
    rad_tmp.h_view(n) = .2 + 0.1*sin(theta);
  }
  S->SetPointwiseRadius(rad_tmp,ctr);

  // set to constant radius
  S->InterpolateToSphere(1,u0);

  
  // evaluate and test spherical harmonics
  S->EvaluateSphericalHarm();

  // test integration on unit sphere
  DualArray1D<Real> ones;
  Kokkos::realloc(ones,nangles);

  for (int n=0; n<nangles; ++n) {
    ones.h_view(n) = 1;
  }
  // test spectral representation
  DualArray1D<Real> ones_spec;
  Kokkos::realloc(ones_spec,4*nlev*nlev);
  ones_spec = S->SpatialToSpectral(ones);

  // test derivative
  DualArray1D<Real> ones_dtheta;
  DualArray1D<Real> ones_dphi;

  Kokkos::realloc(ones_dtheta,nangles);
  Kokkos::realloc(ones_dphi,nangles);

  ones_dtheta = S->ThetaDerivative(ones);
  // ones_dphi = S->PhiDerivative(ones);

  // DualArray1D<> theta;
  // Kokkos::realloc(theta,nangles);

  Real area = S->Integrate(ones);
  std::cout << "area is " << area/M_PI << std::endl;

  std::cout << S->nangles << std::endl;
  

  int test_ind = 462;

  Real &x1min = size.d_view(0).x1min;
  Real &x1max = size.d_view(0).x1max;
  Real &x2min = size.d_view(0).x2min;
  Real &x2max = size.d_view(0).x2max;
  Real &x3min = size.d_view(0).x3min;
  Real &x3max = size.d_view(0).x3max;

  int nx3 = indcs.nx3;
  int nx2 = indcs.nx2;
  int nx1 = indcs.nx1;

  Real interp_value = S->interp_vals.h_view(test_ind,0);
  Real ana_value = 1.0 + 0.2*std::sin(5.0*M_PI*S->interp_coord.h_view(test_ind,0))*std::sin(3.0*M_PI*(S->interp_coord.h_view(test_ind,1)))*std::sin(3.0*M_PI*(S->interp_coord.h_view(test_ind,2)));
  std::cout << "interpolated value  " << interp_value << std::endl;
  std::cout << "analytical value  " << ana_value << std::endl;
  std::cout << "residual  " << fabs(interp_value-ana_value)/ana_value << std::endl;
  /*
  int ll = (int) sqrt(37);
  int mm = (int) (37-ll*ll-ll);
  std::pair<double,double> A = S->SWSphericalHarm(ll,mm,0,1.2,0.);
  std::cout << A.first << "\t" << A.second << std::endl;;
  */
  std::ofstream spherical_grid_output;
  spherical_grid_output.open ("/Users/hawking/Desktop/research/gr/athenak_versions/athenak_spherical_grid/build/spherical_grid_output.txt", std::ios_base::app);
  for (int i=0;i<S->nangles;++i) {
    spherical_grid_output << S->interp_coord.h_view(i,0) << "\t" << S->interp_coord.h_view(i,1) << "\t" << S->interp_coord.h_view(i,2) << "\t"
    << "\t" << S->polar_pos.h_view(i,0) << "\t" << S->polar_pos.h_view(i,1) << "\t" << S->basis_functions.h_view(0,352,i) << "\t" << ones_dtheta.h_view(i) << "\n";// << ones_dphi.h_view(i) <<"\n";
  }
  spherical_grid_output.close();
  
  std::ofstream spectral_rep;
  spectral_rep.open ("/Users/hawking/Desktop/research/gr/athenak_versions/athenak_spherical_grid/build/spectral_rep.txt", std::ios_base::app);
  for (int i=0;i<4*nlev*nlev;++i) {
    int l = (int) sqrt(i);
    int m = (int) (i-l*l-l);
    spectral_rep << i << "\t" << l << "\t" << m << "\t" << ones_spec.h_view(i) << "\n";// << ones_dphi.h_view(i) <<"\n";
  }
  spherical_grid_output.close();
  
  /*
  std::ofstream spectral_basis;
  spectral_basis.open ("/Users/hawking/Desktop/research/gr/athenak_versions/athenak_spherical_grid/build/spectral_rep.txt", std::ios_base::app);
  for (int i=0;i<4*nlev*nlev;++i) {

    spectral_basis << S->.h_view(i) << "\n";// << ones_dphi.h_view(i) <<"\n";
  }
  spherical_grid_output.close();
  */
  return;
}
