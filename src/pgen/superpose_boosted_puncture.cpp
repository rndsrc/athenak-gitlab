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

void Minkowski(MeshBlockPack *pmbp, ParameterInput *pin);
void AddBoostedPuncture(MeshBlockPack *pmbp, ParameterInput *pin, int punc);

KOKKOS_INLINE_FUNCTION
AthenaScratchTensor<Real, TensorSymm::SYM2, 4, 2> 
inverse(AthenaScratchTensor<Real, TensorSymm::SYM2, 4, 2> matrix);

KOKKOS_INLINE_FUNCTION
void LorentzBoost(Real vx1, Real vx2, Real vx3, AthenaScratchTensor<Real, TensorSymm::SYM2, 4, 2> &lambda);

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

  Minkowski(pmbp,pin);
  AddBoostedPuncture(pmbp,pin,1);
  AddBoostedPuncture(pmbp,pin,2);

  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMConstraints<2>(pmbp);
            break;
    case 3: pmbp->pz4c->ADMConstraints<3>(pmbp);
            break;
    case 4: pmbp->pz4c->ADMConstraints<4>(pmbp);
            break;
  }
  std::cout<<"Superposed punctures initialized."<<std::endl;

  return;
}

void Minkowski(MeshBlockPack *pmbp, ParameterInput *pin) {
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
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;

  par_for("pgen one puncture",
  DevExeSpace(),0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Minkolsky metric in the code frame
    for(int a = 0; a < 3; ++a) {
      adm.g_dd(m,a,a,k,j,i) = 1;
    }

    // Gauge variables in the code frame
    adm.alpha(m,k,j,i) = 1;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Initialize ADM vars to single boosted puncture (no spin), based on 1909.02997

void AddBoostedPuncture(MeshBlockPack *pmbp, ParameterInput *pin, int punc_num) {
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
  Real m0 = pin->GetOrAddReal("problem", "punc" + std::to_string(punc_num) + "_ADM_mass", 1.);
  Real center_x1 = pin->GetOrAddReal("problem", "punc" + std::to_string(punc_num) + "_center_x1", 0.);
  Real center_x2 = pin->GetOrAddReal("problem", "punc" + std::to_string(punc_num) + "_center_x2", 0.);
  Real center_x3 = pin->GetOrAddReal("problem", "punc" + std::to_string(punc_num) + "_center_x3", 0.);
  Real vx1 = pin->GetOrAddReal("problem", "punc" + std::to_string(punc_num) + "_velocity_x1", 0.);
  Real vx2 = pin->GetOrAddReal("problem", "punc" + std::to_string(punc_num) + "_velocity_x2", 0.);
  Real vx3 = pin->GetOrAddReal("problem", "punc" + std::to_string(punc_num) + "_velocity_x3", 0.);

  adm::ADM::ADM_vars &adm = pmbp->padm->adm;

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

    // velocity magnitude for now assuming only vx1 is non-zero! Do a rotation later
    Real vel = std::sqrt(std::pow(vx1,2) + std::pow(vx2,2) + std::pow(vx3,2));

    // boost factor
    Real Gamma = 1/std::sqrt(1-std::pow(vel,2));

    // coordinate in the comoving frame (x0)
    Real x0[4];
    Real xinit[4] = {0,x1v,x2v,x3v};

    x0[1] = xinit[1]*Gamma;
    x0[2] = xinit[2];
    x0[3] = xinit[3];

    // radial coordinate in comoving frame
    Real r0 = std::sqrt(std::pow(x0[1],2) + std::pow(x0[2],2) + std::pow(x0[3],2));

    // conformal factor and lapse in comoving frame; equation 2 from arXiv:0810.4735
    Real psi0 = 1.0 + 0.5*m0/r0;
    Real alpha0 = (1 - 0.5*m0/r0)/psi0;

    // B0 as in equation 4 from arXiv:0810.4735
    Real B0 = std::sqrt(std::pow(Gamma,2)*(1-std::pow(vel,2)*std::pow(alpha0,2)*std::pow(psi0,-4)));

    // adm metric in the code frame
    for(int a = 1; a < 3; ++a) {
      adm.g_dd(m,a,a,k,j,i) += std::pow(psi0,4) - 1;
    }
    adm.g_dd(m,1,1,k,j,i) += std::pow(B0,2)*std::pow(psi0,4) - 1;

    // extrinsic curvature
    Real alpha0p = 4*m0/std::pow(m0+2*r0,2);
    Real second_term =
    ((4 * std::pow(vel, 2) * std::pow((m0 - 2 * r0), 2)) / std::pow((m0 + 2 * r0), 3) +
    (4 * std::pow(vel, 2) * (m0 - 2 * r0)) / std::pow((m0 + 2 * r0), 2) -
    (m0 * std::pow((m0 + 2 * r0), 3)) / (4 * std::pow(r0, 5))) /
    ((1 + m0 / (2 * r0)) * (1 + m0 / (2 * r0)) * (1 + m0 / (2 * r0)) * (1 + m0 / (2 * r0)) -
    (std::pow(vel, 2) * std::pow((m0 - 2 * r0), 2)) / std::pow((m0 + 2 * r0), 2));

    adm.vK_dd(m,0,0,k,j,i) += Gamma * Gamma * B0 * x1v * vel / r0 * (2 * alpha0p - alpha0 / 2 * second_term);
    adm.vK_dd(m,1,1,k,j,i) += 2 * Gamma * Gamma * x1v * vel * alpha0 * (- m0 / (2 * r0 * r0)) / (psi0 * B0 * r0);
    adm.vK_dd(m,2,2,k,j,i) += 2 * Gamma * Gamma * x1v * vel * alpha0 * (- m0 / (2 * r0 * r0)) / (psi0 * B0 * r0);
    adm.vK_dd(m,0,1,k,j,i) += B0 * x2v * vel / r0 * (alpha0p - alpha0 / 2 * second_term);
    adm.vK_dd(m,0,2,k,j,i) += B0 * x3v * vel / r0 * (alpha0p - alpha0 / 2 * second_term);
  });
}
