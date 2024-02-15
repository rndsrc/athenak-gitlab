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
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <string>     // c_str(), string
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "adm/adm.hpp"
#include "coordinates/cell_locations.hpp"

void ADMOnePunctureBoosted(MeshBlockPack *pmbp, ParameterInput *pin);
KOKKOS_INLINE_FUNCTION
void inverse(AthenaScratchTensor<Real, TensorSymm::SYM2, 4, 2> matrix,
             AthenaScratchTensor<Real, TensorSymm::SYM2, 4, 2> inverse);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single puncture
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "One Puncture test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  ADMOnePunctureBoosted(pmbp, pin);
  pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  std::cout<<"OnePuncture initialized."<<std::endl;


  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Initialize ADM vars to single boosted puncture (no spin), based on 1909.02997

void ADMOnePunctureBoosted(MeshBlockPack *pmbp, ParameterInput *pin) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  // For GLOOPS
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;
  int nmb = pmbp->nmb_thispack;
  Real ADM_mass = pin->GetOrAddReal("problem", "punc_ADM_mass", 1.);
  Real center_x1 = pin->GetOrAddReal("problem", "punc_center_x1", 0.);
  Real center_x2 = pin->GetOrAddReal("problem", "punc_center_x2", 0.);
  Real center_x3 = pin->GetOrAddReal("problem", "punc_center_x3", 0.);
  Real velocity_x1 = pin->GetOrAddReal("problem", "punc_velocity_x1", 0.);
  Real velocity_x2 = pin->GetOrAddReal("problem", "punc_velocity_x2", 0.);
  Real velocity_x3 = pin->GetOrAddReal("problem", "punc_velocity_x3", 0.);

  adm::ADM::ADM_vars &adm = pmbp->padm->adm;
  z4c::Z4c::Z4c_vars &z4c = pmbp->pz4c->z4c;

  par_for("pgen one puncture",
  DevExeSpace(),0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    x1v -= center_x1;
    x2v -= center_x2;
    x3v -= center_x3;

    Real r = std::sqrt(std::pow(x3v,2) + std::pow(x2v,2) + std::pow(x1v,2));
    // First construct the unboosted solution
    // Minkowski spacetime
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      adm.g_dd(m,a,b,k,j,i) = (a == b ? 1. : 0.);
    }

    // conformal factor and omega (whatever it's called)
    Real psi = 1.0 + 0.25*ADM_mass/r;
    Real omega = 1.0 - 0.25*ADM_mass/r;

    // unboosted variables
    // admK_dd and lapse is automatically set to 0 when is initialized as Kokkos View
    z4c.alpha(m,k,j,i) = omega/psi;
    adm.psi4(m,k,j,i) = std::pow(psi,4); // adm.psi4
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      adm.g_dd(m,a,b,k,j,i) *= adm.psi4(m,k,j,i);
    }
    // Velocity magnitude squared
    Real v_squared = SQR(velocity_x1)-SQR(velocity_x2)-SQR(velocity_x3);
    
    // Terminate if the black hole is not moving
    if (v_squared == 0) {
      return;
    }

    // Assembling the full spacetime metric
    // be careful with the indices as the 0th component is now time!
    AthenaScratchTensor<Real, TensorSymm::SYM2, 4, 2> g_full_dd;
    g_full_dd(0,0) = -std::pow(z4c.alpha(m,k,j,i),2);
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      g_full_dd(a+1,b+1) = adm.g_dd(m,a,b,k,j,i);
    }

    // Construct matrix for Lorentz boost
    AthenaScratchTensor<Real, TensorSymm::SYM2, 4, 2> lambda;


    // Calculate Lorentz factor
    Real gamma = 1/sqrt(1-v_squared);

    lambda(0,0) = gamma;
    lambda(0,1) = gamma*velocity_x1;
    lambda(0,2) = gamma*velocity_x2;
    lambda(0,3) = gamma*velocity_x3;
    lambda(1,1) = 1 + (gamma-1)*SQR(velocity_x1)/v_squared;
    lambda(2,2) = 1 + (gamma-1)*SQR(velocity_x2)/v_squared;
    lambda(3,3) = 1 + (gamma-1)*SQR(velocity_x3)/v_squared;
    lambda(1,2) = (gamma-1)*velocity_x1*velocity_x2/v_squared;
    lambda(1,3) = (gamma-1)*velocity_x1*velocity_x3/v_squared;
    lambda(2,3) = (gamma-1)*velocity_x2*velocity_x3/v_squared;
    std::cout << lambda(0,0) << "\t" << lambda(0,1) << "\t"<< lambda(0,2) << "\t"<< lambda(0,3) << "\t"<< lambda(1,1) 
              << "\t"<< lambda(1,2) << "\t"<< lambda(1,3) << "\t"<< lambda(2,2) << "\t"<< lambda(2,3) 
              << "\t"<< lambda(3,3) << "\t" << std::endl;
    // Construct the boosted metric
    AthenaScratchTensor<Real, TensorSymm::SYM2, 4, 2> g_boosted_dd;
    for(int a = 0; a < 4; ++a)
    for(int b = a; b < 4; ++b)
    for(int c = 0; c < 4; ++c)
    for(int d = 0; d < 4; ++d) {
      g_boosted_dd(a,b) += lambda(c,a)*lambda(d,b)*g_full_dd(c,d);
    }

    // Lastly calculate the ADM variable
    // First the inverted spacetime metric
    AthenaScratchTensor<Real, TensorSymm::SYM2, 4, 2> g_boosted_uu;
    inverse(g_boosted_dd,g_boosted_uu);
    z4c.alpha(m,k,j,i) = 1/sqrt(-g_boosted_uu(0,0));
    for(int a = 0; a < 3; ++a) {
      z4c.beta_u(m,a,k,j,i) = g_boosted_dd(0,a+1);
    }
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      adm.g_dd(m,a,b,k,j,i) = g_boosted_dd(a+1,b+1);
    }
  });
}

KOKKOS_INLINE_FUNCTION
void inverse(AthenaScratchTensor<Real, TensorSymm::SYM2, 4, 2> matrix,
             AthenaScratchTensor<Real, TensorSymm::SYM2, 4, 2> inverse) {
    Real det = 0;

    // Calculate the determinant using the formula for a 4x4 matrix
    det = matrix(0,0) * (matrix(1,1) * (matrix(2,2) * matrix(3,3) - matrix(2,3) * matrix(3,2)) -
                          matrix(1,2) * (matrix(2,1) * matrix(3,3) - matrix(2,3) * matrix(3,1)) +
                          matrix(1,3) * (matrix(2,1) * matrix(3,2) - matrix(2,2) * matrix(3,1))) -
          matrix(0,1) * (matrix(1,0) * (matrix(2,2) * matrix(3,3) - matrix(2,3) * matrix(3,2)) -
                          matrix(1,2) * (matrix(2,0) * matrix(3,3) - matrix(2,3) * matrix(3,0)) +
                          matrix(1,3) * (matrix(2,0) * matrix(3,2) - matrix(2,2) * matrix(3,0))) +
          matrix(0,2) * (matrix(1,0) * (matrix(2,1) * matrix(3,3) - matrix(2,3) * matrix(3,1)) -
                          matrix(1,1) * (matrix(2,0) * matrix(3,3) - matrix(2,3) * matrix(3,0)) +
                          matrix(1,3) * (matrix(2,0) * matrix(3,1) - matrix(2,1) * matrix(3,0))) -
          matrix(0,3) * (matrix(1,0) * (matrix(2,1) * matrix(3,2) - matrix(2,2) * matrix(3,1)) -
                          matrix(1,1) * (matrix(2,0) * matrix(3,2) - matrix(2,2) * matrix(3,0)) +
                          matrix(1,2) * (matrix(2,0) * matrix(3,1) - matrix(2,1) * matrix(3,0)));

    // Check if determinant is non-zero
    if (det == 0) {
        std::cerr << "Matrix is singular, cannot invert." << std::endl;
        return;
    }

    // Calculate the inverse using the formula for a 4x4 matrix
    inverse(0,0) = (matrix(1,1) * (matrix(2,2) * matrix(3,3) - matrix(2,3) * matrix(3,2)) -
                     matrix(1,2) * (matrix(2,1) * matrix(3,3) - matrix(2,3) * matrix(3,1)) +
                     matrix(1,3) * (matrix(2,1) * matrix(3,2) - matrix(2,2) * matrix(3,1))) /
                    det;
    inverse(0,1) = -(matrix(0,1) * (matrix(2,2) * matrix(3,3) - matrix(2,3) * matrix(3,2)) -
                      matrix(0,2) * (matrix(2,1) * matrix(3,3) - matrix(2,3) * matrix(3,1)) +
                      matrix(0,3) * (matrix(2,1) * matrix(3,2) - matrix(2,2) * matrix(3,1))) /
                    det;
    inverse(0,2) = (matrix(0,1) * (matrix(1,2) * matrix(3,3) - matrix(1,3) * matrix(3,2)) -
                     matrix(0,2) * (matrix(1,1) * matrix(3,3) - matrix(1,3) * matrix(3,1)) +
                     matrix(0,3) * (matrix(1,1) * matrix(3,2) - matrix(1,2) * matrix(3,1))) /
                    det;
    inverse(0,3) = -(matrix(0,1) * (matrix(1,2) * matrix(2,3) - matrix(1,3) * matrix(2,2)) -
                     matrix(0,2) * (matrix(1,1) * matrix(2,3) - matrix(1,3) * matrix(2,1)) +
                     matrix(0,3) * (matrix(1,1) * matrix(2,2) - matrix(1,2) * matrix(2,1))) /
                    det;

    inverse(1,0) = -(matrix(1,0) * (matrix(2,2) * matrix(3,3) - matrix(2,3) * matrix(3,2)) -
                      matrix(1,2) * (matrix(2,0) * matrix(3,3) - matrix(2,3) * matrix(3,0)) +
                      matrix(1,3) * (matrix(2,0) * matrix(3,2) - matrix(2,2) * matrix(3,0))) /
                    det;
    inverse(1,1) = (matrix(0,0) * (matrix(2,2) * matrix(3,3) - matrix(2,3) * matrix(3,2)) -
                     matrix(0,2) * (matrix(2,0) * matrix(3,3) - matrix(2,3) * matrix(3,0)) +
                     matrix(0,3) * (matrix(2,0) * matrix(3,2) - matrix(2,2) * matrix(3,0))) /
                    det;
    inverse(1,2) = -(matrix(0,0) * (matrix(1,2) * matrix(3,3) - matrix(1,3) * matrix(3,2)) -
                      matrix(0,2) * (matrix(1,0) * matrix(3,3) - matrix(1,3) * matrix(3,0)) +
                      matrix(0,3) * (matrix(1,0) * matrix(3,2) - matrix(1,2) * matrix(3,0))) /
                    det;
    inverse(1,3) = (matrix(0,0) * (matrix(1,2) * matrix(2,3) - matrix(1,3) * matrix(2,2)) -
                     matrix(0,2) * (matrix(1,0) * matrix(2,3) - matrix(1,3) * matrix(2,0)) +
                     matrix(0,3) * (matrix(1,0) * matrix(2,2) - matrix(1,2) * matrix(2,0))) /
                    det;

    inverse(2,0) = (matrix(1,0) * (matrix(2,1) * matrix(3,3) - matrix(2,3) * matrix(3,1)) -
                     matrix(1,1) * (matrix(2,0) * matrix(3,3) - matrix(2,3) * matrix(3,0)) +
                     matrix(1,3) * (matrix(2,0) * matrix(3,1) - matrix(2,1) * matrix(3,0))) /
                    det;
    inverse(2,1) = -(matrix(0,0) * (matrix(2,1) * matrix(3,3) - matrix(2,3) * matrix(3,1)) -
                      matrix(0,1) * (matrix(2,0) * matrix(3,3) - matrix(2,3) * matrix(3,0)) +
                      matrix(0,3) * (matrix(2,0) * matrix(3,1) - matrix(2,1) * matrix(3,0))) /
                    det;
    inverse(2,2) = (matrix(0,0) * (matrix(1,1) * matrix(3,3) - matrix(1,3) * matrix(3,1)) -
                     matrix(0,1) * (matrix(1,0) * matrix(3,3) - matrix(1,3) * matrix(3,0)) +
                     matrix(0,3) * (matrix(1,0) * matrix(3,1) - matrix(1,1) * matrix(3,0))) /
                    det;
    inverse(2,3) = -(matrix(0,0) * (matrix(1,1) * matrix(2,3) - matrix(1,3) * matrix(2,1)) -
                      matrix(0,1) * (matrix(1,0) * matrix(2,3) - matrix(1,3) * matrix(2,0)) +
                      matrix(0,3) * (matrix(1,0) * matrix(2,1) - matrix(1,1) * matrix(2,0))) /
                    det;

    inverse(3,0) = -(matrix(1,0) * (matrix(2,1) * matrix(3,2) - matrix(2,2) * matrix(3,1)) -
                      matrix(1,1) * (matrix(2,0) * matrix(3,2) - matrix(2,2) * matrix(3,0)) +
                      matrix(1,2) * (matrix(2,0) * matrix(3,1) - matrix(2,1) * matrix(3,0))) /
                    det;
    inverse(3,1) = (matrix(0,0) * (matrix(2,1) * matrix(3,2) - matrix(2,2) * matrix(3,1)) -
                     matrix(0,1) * (matrix(2,0) * matrix(3,2) - matrix(2,2) * matrix(3,0)) +
                     matrix(0,2) * (matrix(2,0) * matrix(3,1) - matrix(2,1) * matrix(3,0))) /
                    det;
    inverse(3,2) = -(matrix(0,0) * (matrix(1,1) * matrix(3,2) - matrix(1,2) * matrix(3,1)) -
                      matrix(0,1) * (matrix(1,0) * matrix(3,2) - matrix(1,2) * matrix(3,0)) +
                      matrix(0,2) * (matrix(1,0) * matrix(3,1) - matrix(1,1) * matrix(3,0))) /
                    det;
    inverse(3,3) = (matrix(0,0) * (matrix(1,1) * matrix(2,2) - matrix(1,2) * matrix(2,1)) -
                     matrix(0,1) * (matrix(1,0) * matrix(2,2) - matrix(1,2) * matrix(2,0)) +
                     matrix(0,2) * (matrix(1,0) * matrix(2,1) - matrix(1,1) * matrix(2,0))) /
                    det;
}
