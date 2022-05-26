//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_linear_wave.cpp
//  \brief GR radiation linear wave test

// C++ headers
#include <algorithm>  // min, max
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <string>     // c_str()
#include <limits>

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "radiation/radiation.hpp"
#include "srcterms/srcterms.hpp"
#include "pgen.hpp"

#include "radiation/radiation_tetrad.hpp"


//----------------------------------------------------------------------------------------
//! \struct RadLinWaveVariables
//  \brief container for variables shared with vector potential and error functions

struct RadLinWaveVariables {
  Real k_par;
  Real cos_a2, cos_a3, sin_a2, sin_a3;
};

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for GR radiation linear wave test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  bool along_x1 = pin->GetOrAddBoolean("problem", "along_x1", false);
  bool along_x2 = pin->GetOrAddBoolean("problem", "along_x2", false);
  bool along_x3 = pin->GetOrAddBoolean("problem", "along_x3", false);
  // error check input flags
  if ((along_x1 && (along_x2 || along_x3)) || (along_x2 && along_x3)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Can only specify one of along_x1/2/3 to be true" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((along_x2 || along_x3) && pmy_mesh_->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Cannot specify waves along x2 or x3 axis in 1D" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (along_x3 && pmy_mesh_->two_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Cannot specify waves along x3 axis in 2D" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Code below will automatically calculate wavevector along grid diagonal, imposing the
  // conditions of periodicity and exactly one wavelength along each grid direction
  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real x2size = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  Real x3size = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;

  // start with wavevector along x1 axis
  RadLinWaveVariables rlw;
  rlw.cos_a3 = 1.0;
  rlw.sin_a3 = 0.0;
  rlw.cos_a2 = 1.0;
  rlw.sin_a2 = 0.0;
  if (pmy_mesh_->multi_d && !(along_x1)) {
    Real ang_3 = std::atan(x1size/x2size);
    rlw.sin_a3 = std::sin(ang_3);
    rlw.cos_a3 = std::cos(ang_3);
  }
  if (pmy_mesh_->three_d && !(along_x1)) {
    Real ang_2 = std::atan(0.5*(x1size*rlw.cos_a3 + x2size*rlw.sin_a3)/x3size);
    rlw.sin_a2 = std::sin(ang_2);
    rlw.cos_a2 = std::cos(ang_2);
  }

  // hardcode wavevector along x2 axis, override ang_2, ang_3
  if (along_x2) {
    rlw.cos_a3 = 0.0;
    rlw.sin_a3 = 1.0;
    rlw.cos_a2 = 1.0;
    rlw.sin_a2 = 0.0;
  }

  // hardcode wavevector along x3 axis, override ang_2, ang_3
  if (along_x3) {
    rlw.cos_a3 = 0.0;
    rlw.sin_a3 = 1.0;
    rlw.cos_a2 = 0.0;
    rlw.sin_a2 = 1.0;
  }

  // choose the smallest projection of the wavelength in each direction that is > 0
  Real lambda = std::numeric_limits<float>::max();
  if (rlw.cos_a2*rlw.cos_a3 > 0.0) {
    lambda = std::min(lambda, x1size*rlw.cos_a2*rlw.cos_a3);
  }
  if (rlw.cos_a2*rlw.sin_a3 > 0.0) {
    lambda = std::min(lambda, x2size*rlw.cos_a2*rlw.sin_a3);
  }
  if (rlw.sin_a2 > 0.0) lambda = std::min(lambda, x3size*rlw.sin_a2);

  // Initialize k_parallel
  rlw.k_par = 2.0*(M_PI)/lambda;

  // set EOS data
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;

  // Set eigensystem
  Real omega_real = pin->GetReal("problem", "omega_real");
  Real omega_imag = pin->GetReal("problem", "omega_imag");

  Real rho = pin->GetReal("problem", "rho");
  Real pgas = pin->GetReal("problem", "pgas");
  Real ux = pin->GetReal("problem", "ux");
  Real uy = pin->GetReal("problem", "uy");
  Real uz = pin->GetReal("problem", "uz");
  Real erad = pin->GetReal("problem", "erad");
  Real fxrad = pin->GetReal("problem", "fxrad");
  Real fyrad = pin->GetReal("problem", "fyrad");
  Real fzrad = pin->GetReal("problem", "fzrad");

  Real delta = pin->GetReal("problem", "delta");
  Real drho_real = pin->GetReal("problem", "drho_real");
  Real drho_imag = pin->GetReal("problem", "drho_imag");
  Real dpgas_real = pin->GetReal("problem", "dpgas_real");
  Real dpgas_imag = pin->GetReal("problem", "dpgas_imag");
  Real dux_real = pin->GetReal("problem", "dux_real");
  Real dux_imag = pin->GetReal("problem", "dux_imag");
  Real duy_real = pin->GetReal("problem", "duy_real");
  Real duy_imag = pin->GetReal("problem", "duy_imag");
  Real duz_real = pin->GetReal("problem", "duz_real");
  Real duz_imag = pin->GetReal("problem", "duz_imag");
  Real derad_real = pin->GetReal("problem", "derad_real");
  Real derad_imag = pin->GetReal("problem", "derad_imag");
  Real dfxrad_real = pin->GetReal("problem", "dfxrad_real");
  Real dfxrad_imag = pin->GetReal("problem", "dfxrad_imag");
  Real dfyrad_real = pin->GetReal("problem", "dfyrad_real");
  Real dfyrad_imag = pin->GetReal("problem", "dfyrad_imag");
  Real dfzrad_real = pin->GetReal("problem", "dfzrad_real");
  Real dfzrad_imag = pin->GetReal("problem", "dfzrad_imag");

  Real tlim = pin->GetReal("time", "tlim");
  pin->SetReal("time", "tlim", tlim*log(2.0)/fabs(omega_imag));

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  auto &size = pmbp->pmb->mb_size;
  int nangles_ = pmbp->prad->nangles;
  auto &coord = pmbp->pcoord->coord_data;

  auto &w0 = pmbp->phydro->w0;
  par_for("rad_wave",DevExeSpace(),0,(pmbp->nmb_thispack-1),0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
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

    Real x = rlw.cos_a2*(x1v*rlw.cos_a3 + x2v*rlw.sin_a3) + x3v*rlw.sin_a2;
    Real sn = sin(rlw.k_par*x);
    Real cn = cos(rlw.k_par*x);
    Real rhon  = rho  + delta*(drho_real*cn - drho_imag*sn);
    Real pgasn = pgas + delta*(dpgas_real*cn - dpgas_imag*sn);
    Real uxn   = ux   + delta*(dux_real*cn - dux_imag*sn);
    Real uyn   = uy   + delta*(duy_real*cn - duy_imag*sn);
    Real uzn   = uz   + delta*(duz_real*cn - duz_imag*sn);

    w0(m,IDN,k,j,i) = rhon;
    w0(m,IVX,k,j,i) = uxn*rlw.cos_a2*rlw.cos_a3-uyn*rlw.sin_a3-uzn*rlw.sin_a2*rlw.cos_a3;
    w0(m,IVY,k,j,i) = uxn*rlw.cos_a2*rlw.sin_a3+uyn*rlw.cos_a3-uzn*rlw.sin_a2*rlw.sin_a3;
    w0(m,IVZ,k,j,i) = uxn*rlw.sin_a2                          +uzn*rlw.cos_a2;
    w0(m,IEN,k,j,i) = pgasn/gm1;
  });

  // Convert primitives to conserved
  auto &u0 = pmbp->phydro->u0;
  pmbp->phydro->peos->PrimToCons(w0, u0, 0, (n1-1), 0, (n2-1), 0, (n3-1));

  auto nh_c_ = pmbp->prad->nh_c;
  auto norm_to_tet_ = pmbp->prad->norm_to_tet;
  auto tetcov_c_ = pmbp->prad->tetcov_c;

  auto &i0 = pmbp->prad->i0;
  par_for("rad_wave2",DevExeSpace(),0,(pmbp->nmb_thispack-1),0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
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

    Real g_[NMETRIC], gi_[NMETRIC];
    ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin, g_, gi_);

    Real x = rlw.cos_a2*(x1v*rlw.cos_a3 + x2v*rlw.sin_a3) + x3v*rlw.sin_a2;
    Real sn = sin(rlw.k_par*x);
    Real cn = cos(rlw.k_par*x);

    // Calculate wave-aligned coordinate-frame fluid velocity
    Real u_wave[4];
    u_wave[1] = ux + delta*(dux_real*cn - dux_imag*sn);
    u_wave[2] = uy + delta*(duy_real*cn - duy_imag*sn);
    u_wave[3] = uz + delta*(duz_real*cn - duz_imag*sn);
    u_wave[0] = hypot(1.0, hypot(u_wave[1], hypot(u_wave[2], u_wave[3])));

    // Calculate coordinate-frame fluid velocity
    Real u[4];
    u[0] = u_wave[0];
    u[1] = (u_wave[1]*rlw.cos_a2*rlw.cos_a3 -u_wave[2]*rlw.sin_a3 -
            u_wave[3]*rlw.sin_a2*rlw.cos_a3);
    u[2] = (u_wave[1]*rlw.cos_a2*rlw.sin_a3 +u_wave[2]*rlw.cos_a3 -
            u_wave[3]*rlw.sin_a2*rlw.sin_a3);
    u[3] = u_wave[1]*rlw.sin_a2 + u_wave[3]*rlw.cos_a2;

    // Calculate wave-aligned fluid-frame radiation moments
    Real rf_wave[4][4];
    rf_wave[0][0] =  erad + delta*( derad_real*cn -  derad_imag*sn);
    rf_wave[0][1] = fxrad + delta*(dfxrad_real*cn - dfxrad_imag*sn);
    rf_wave[0][2] = fyrad + delta*(dfyrad_real*cn - dfyrad_imag*sn);
    rf_wave[0][3] = fzrad + delta*(dfzrad_real*cn - dfzrad_imag*sn);
    rf_wave[1][1] = 1.0/3.0*rf_wave[0][0];
    rf_wave[2][2] = 1.0/3.0*rf_wave[0][0];
    rf_wave[3][3] = 1.0/3.0*rf_wave[0][0];
    rf_wave[1][2] = 0.0;
    rf_wave[1][3] = 0.0;
    rf_wave[2][3] = 0.0;
    rf_wave[1][0] = rf_wave[0][1];
    rf_wave[2][0] = rf_wave[0][2];
    rf_wave[3][0] = rf_wave[0][3];
    rf_wave[2][1] = rf_wave[1][2];
    rf_wave[3][1] = rf_wave[1][3];
    rf_wave[3][2] = rf_wave[2][3];

    // Calculate wave-aligned coordinate-frame radiation moments
    Real lambda_c_f_wave[4][4];
    lambda_c_f_wave[0][0] = u_wave[0];
    lambda_c_f_wave[0][1] = u_wave[1];
    lambda_c_f_wave[0][2] = u_wave[2];
    lambda_c_f_wave[0][3] = u_wave[3];
    lambda_c_f_wave[1][1] = 1.0 + 1.0/(1.0 + u_wave[0])*SQR(u_wave[1]);
    lambda_c_f_wave[2][2] = 1.0 + 1.0/(1.0 + u_wave[0])*SQR(u_wave[2]);
    lambda_c_f_wave[3][3] = 1.0 + 1.0/(1.0 + u_wave[0])*SQR(u_wave[3]);
    lambda_c_f_wave[1][2] = 1./(1.+u_wave[0])*u_wave[1]*u_wave[2];
    lambda_c_f_wave[1][3] = 1./(1.+u_wave[0])*u_wave[1]*u_wave[3];
    lambda_c_f_wave[2][3] = 1./(1.+u_wave[0])*u_wave[2]*u_wave[3];
    lambda_c_f_wave[1][0] = lambda_c_f_wave[0][1];
    lambda_c_f_wave[2][0] = lambda_c_f_wave[0][2];
    lambda_c_f_wave[3][0] = lambda_c_f_wave[0][3];
    lambda_c_f_wave[2][1] = lambda_c_f_wave[1][2];
    lambda_c_f_wave[3][1] = lambda_c_f_wave[1][3];
    lambda_c_f_wave[3][2] = lambda_c_f_wave[2][3];

    Real r_wave[4][4];
    for (int alpha=0; alpha<4; ++alpha) {
      for (int beta=0; beta<4; ++beta) {
        r_wave[alpha][beta] = 0.0;
        for (int mu=0; mu<4; ++mu) {
          for (int nu=0; nu<4; ++nu) {
            r_wave[alpha][beta] += lambda_c_f_wave[alpha][mu]
               *lambda_c_f_wave[beta][nu]*rf_wave[mu][nu];
          }
        }
      }
    }

    // Calculate coordinate-frame radiation moments
    Real r[4][4];
    r[0][0] = r_wave[0][0];
    r[0][1] = (rlw.cos_a2*rlw.cos_a3*r_wave[0][1] - rlw.sin_a3*r_wave[0][2] -
               rlw.cos_a3*rlw.sin_a2*r_wave[0][3]);
    r[0][2] = (rlw.cos_a2*rlw.sin_a3*r_wave[0][1] + rlw.cos_a3*r_wave[0][2] -
               rlw.sin_a2*rlw.sin_a3*r_wave[0][3]);
    r[0][3] = rlw.cos_a2*r_wave[0][3] + rlw.sin_a2*r_wave[0][1];
    r[1][1] = (r_wave[1][1]*SQR(rlw.cos_a2)*SQR(rlw.cos_a3) +
               r_wave[2][2]*SQR(rlw.sin_a3) +
               r_wave[3][3]*SQR(rlw.cos_a3)*SQR(rlw.sin_a2) -
               2.0*r_wave[1][3]*rlw.cos_a2*SQR(rlw.cos_a3)*rlw.sin_a2 -
               2.0*r_wave[1][2]*rlw.cos_a2*rlw.cos_a3*rlw.sin_a3 +
               2.0*r_wave[2][3]*rlw.cos_a3*rlw.sin_a2*rlw.sin_a3);
    r[2][2] = (r_wave[1][1]*SQR(rlw.cos_a2)*SQR(rlw.sin_a3) +
               r_wave[2][2]*SQR(rlw.cos_a3) +
               r_wave[3][3]*SQR(rlw.sin_a2)*SQR(rlw.sin_a3) +
               2.0*r_wave[1][2]*rlw.cos_a2*rlw.cos_a3*rlw.sin_a3 -
               2.0*r_wave[2][3]*rlw.cos_a3*rlw.sin_a2*rlw.sin_a3 -
               2.0*r_wave[1][3]*rlw.cos_a2*rlw.sin_a2*SQR(rlw.sin_a3));
    r[3][3] = (r_wave[3][3]*SQR(rlw.cos_a2) + 2.0*r_wave[1][3]*rlw.cos_a2*rlw.sin_a2 +
               r_wave[1][1]*SQR(rlw.sin_a2));
    r[1][2] = (r_wave[1][2]*rlw.cos_a2*SQR(rlw.cos_a3) -
               r_wave[2][3]*SQR(rlw.cos_a3)*rlw.sin_a2 -
               r_wave[2][2]*rlw.cos_a3*rlw.sin_a3 +
               r_wave[1][1]*SQR(rlw.cos_a2)*rlw.cos_a3*rlw.sin_a3  +
               r_wave[3][3]*rlw.cos_a3*SQR(rlw.sin_a2)*rlw.sin_a3 -
               r_wave[1][2]*rlw.cos_a2*SQR(rlw.sin_a3) +
               r_wave[2][3]*rlw.sin_a2*SQR(rlw.sin_a3) -
               2.0*r_wave[1][3]*rlw.cos_a2*rlw.cos_a3*rlw.sin_a2*rlw.sin_a3);
    r[1][3] = (r_wave[1][3]*SQR(rlw.cos_a2)*rlw.cos_a3 +
               r_wave[1][1]*rlw.cos_a2*rlw.cos_a3*rlw.sin_a2 -
               r_wave[3][3]*rlw.cos_a2*rlw.cos_a3*rlw.sin_a2 -
               r_wave[1][3]*rlw.cos_a3*SQR(rlw.sin_a2) -
               r_wave[2][3]*rlw.cos_a2*rlw.sin_a3 -
               r_wave[1][2]*rlw.sin_a2*rlw.sin_a3);
    r[2][3] = (r_wave[2][3]*rlw.cos_a2*rlw.cos_a3 +
               r_wave[1][2]*rlw.cos_a3*rlw.sin_a2 +
               r_wave[1][3]*SQR(rlw.cos_a2)*rlw.sin_a3 +
               r_wave[1][1]*rlw.cos_a2*rlw.sin_a2*rlw.sin_a3 -
               r_wave[3][3]*rlw.cos_a2*rlw.sin_a2*rlw.sin_a3 -
               r_wave[1][3]*SQR(rlw.sin_a2)*rlw.sin_a3);
    r[1][0] = r[0][1];
    r[2][0] = r[0][2];
    r[3][0] = r[0][3];
    r[2][1] = r[1][2];
    r[3][1] = r[1][3];
    r[3][2] = r[2][3];

    // Calculate fluid-frame radiation moments
    Real lambda_f_c[4][4];
    lambda_f_c[0][0] =  u[0];
    lambda_f_c[0][1] = -u[1];
    lambda_f_c[0][2] = -u[2];
    lambda_f_c[0][3] = -u[3];
    lambda_f_c[1][1] = 1.0 + 1.0/(1.0 + u[0])*u[1]*u[1];
    lambda_f_c[2][2] = 1.0 + 1.0/(1.0 + u[0])*u[2]*u[2];
    lambda_f_c[3][3] = 1.0 + 1.0/(1.0 + u[0])*u[3]*u[3];
    lambda_f_c[1][2] = 1.0/(1.0 + u[0])*u[1]*u[2];
    lambda_f_c[1][3] = 1.0/(1.0 + u[0])*u[1]*u[3];
    lambda_f_c[2][3] = 1.0/(1.0 + u[0])*u[2]*u[3];
    lambda_f_c[1][0] = lambda_f_c[0][1];
    lambda_f_c[2][0] = lambda_f_c[0][2];
    lambda_f_c[3][0] = lambda_f_c[0][3];
    lambda_f_c[2][1] = lambda_f_c[1][2];
    lambda_f_c[3][1] = lambda_f_c[1][3];
    lambda_f_c[3][2] = lambda_f_c[2][3];

    Real rf[4][4];
    for (int alpha=0; alpha<4; ++alpha) {
      for (int beta=0; beta<4; ++beta) {
        rf[alpha][beta] = 0.0;
        for (int mu=0; mu<4; ++mu) {
          for (int nu=0; nu<4; ++nu) {
            rf[alpha][beta] += lambda_f_c[alpha][mu]*lambda_f_c[beta][nu]*r[mu][nu];
          }
        }
      }
    }

    // Calculate normalized flux in fluid frame
    Real ee_f  = rf[0][0];
    Real ff1_f = rf[0][1];
    Real ff2_f = rf[0][2];
    Real ff3_f = rf[0][3];
    Real ff_f = sqrt(SQR(ff1_f) + SQR(ff2_f) + SQR(ff3_f));
    Real f_f  = ff_f/ee_f;
    Real f1_f = ff1_f/ff_f;
    Real f2_f = ff2_f/ff_f;
    Real f3_f = ff3_f/ff_f;

    // Compute fluid velocity in tetrad frame
    Real uu1 = u[1];
    Real uu2 = u[2];
    Real uu3 = u[3];
    Real tmp_var = g_[I11]*uu1*uu1 + 2.0*g_[I12]*uu1*uu2 + 2.0*g_[I13]*uu1*uu3
                                   +     g_[I22]*uu2*uu2 + 2.0*g_[I23]*uu2*uu3
                                                         +     g_[I33]*uu3*uu3;
    Real uu0 = sqrt(1.0 + tmp_var);

    Real u_tet_[4];
    u_tet_[0] = (norm_to_tet_(m,0,0,k,j,i)*uu0 + norm_to_tet_(m,0,1,k,j,i)*uu1 +
                 norm_to_tet_(m,0,2,k,j,i)*uu2 + norm_to_tet_(m,0,3,k,j,i)*uu3);
    u_tet_[1] = (norm_to_tet_(m,1,0,k,j,i)*uu0 + norm_to_tet_(m,1,1,k,j,i)*uu1 +
                 norm_to_tet_(m,1,2,k,j,i)*uu2 + norm_to_tet_(m,1,3,k,j,i)*uu3);
    u_tet_[2] = (norm_to_tet_(m,2,0,k,j,i)*uu0 + norm_to_tet_(m,2,1,k,j,i)*uu1 +
                 norm_to_tet_(m,2,2,k,j,i)*uu2 + norm_to_tet_(m,2,3,k,j,i)*uu3);
    u_tet_[3] = (norm_to_tet_(m,3,0,k,j,i)*uu0 + norm_to_tet_(m,3,1,k,j,i)*uu1 +
                 norm_to_tet_(m,3,2,k,j,i)*uu2 + norm_to_tet_(m,3,3,k,j,i)*uu3);

    // Go through each angle
    for (int n=0; n<nangles_; ++n) {
      // Calculate direction in fluid frame
      Real un_t =  (u_tet_[1]*nh_c_.d_view(n,1) + u_tet_[2]*nh_c_.d_view(n,2) +
                    u_tet_[3]*nh_c_.d_view(n,3));

      Real n0_f =  u_tet_[0]*nh_c_.d_view(n,0) - un_t;
      Real n1_f = (-u_tet_[1]*nh_c_.d_view(n,0) + u_tet_[1]/(u_tet_[0] + 1.0)*un_t +
                   nh_c_.d_view(n,1));
      Real n2_f = (-u_tet_[2]*nh_c_.d_view(n,0) + u_tet_[2]/(u_tet_[0] + 1.0)*un_t +
                   nh_c_.d_view(n,2));
      Real n3_f = (-u_tet_[3]*nh_c_.d_view(n,0) + u_tet_[3]/(u_tet_[0] + 1.0)*un_t +
                   nh_c_.d_view(n,3));

      // Calculate intensity in fluid frame
      Real fn_f = f1_f*n1_f + f2_f*n2_f + f3_f*n3_f;
      Real ii_f = 0.0;
      if (f_f <= 1.0/3.0) {
        ii_f = ee_f/(4.0*M_PI)*(1.0 + 3.0*f_f*fn_f);
      } else {
        ii_f = ee_f/(9.0*M_PI)*(fn_f - 3.0*f_f + 2.0)/SQR(1.0 - f_f);
      }

      // Calculate intensity in tetrad frame
      Real n_0 = 0.0;
      for (int d=0; d<4; ++d) {  n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);  }
      i0(m,n,k,j,i) = n_0*ii_f/SQR(SQR(n0_f));
    }
  });

  return;
}
