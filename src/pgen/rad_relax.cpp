//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_relax.cpp
//  \brief thermal relaxation test

// C++ headers
#include <algorithm>  // min, max
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

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

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for GR radiation beam test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // capture variables for kernel
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  auto &size = pmbp->pmb->mb_size;
  auto &coord = pmbp->pcoord->coord_data;
  int nmb1 = (pmbp->nmb_thispack-1);
  int nang1 = (pmbp->prad->nangles-1);

  Real erad = pin->GetReal("problem", "erad");
  Real temp = pin->GetReal("problem", "temp");
  Real v1 = pin->GetReal("problem", "v1");
  Real lf = 1.0/sqrt(1.0-(SQR(v1)));

  auto &w0 = pmbp->phydro->w0;
  par_for("pgen_shadow1",DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    w0(m,IDN,k,j,i) = 1.0;
    w0(m,IVX,k,j,i) = lf*v1;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    w0(m,IEN,k,j,i) = temp;  // assumes that gm1=1
  });

  // Convert primitives to conserved
  auto &u0 = pmbp->phydro->u0;
  pmbp->phydro->peos->PrimToCons(w0, u0, 0, (n1-1), 0, (n2-1), 0, (n3-1));

  auto norm_to_tet_ = pmbp->prad->norm_to_tet;
  auto nh_c_ = pmbp->prad->nh_c;
  auto tetcov_c_ = pmbp->prad->tetcov_c;

  auto &i0 = pmbp->prad->i0;
  par_for("rad_relax",DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
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

    // Calculate normalized flux in fluid frame
    Real ee_f  = erad;

    // Compute fluid velocity in tetrad frame
    Real uu1 = w0(m,IVX,k,j,i);
    Real uu2 = w0(m,IVY,k,j,i);
    Real uu3 = w0(m,IVZ,k,j,i);
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
    for (int n=0; n<=nang1; ++n) {
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
      Real ii_f =  ee_f/(4.0*M_PI);

      // Calculate intensity in tetrad frame
      Real n_0 = 0.0;
      for (int d=0; d<4; ++d) {  n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);  }

      printf("%24.16e %24.16e\n", ii_f, n_0);

      i0(m,n,k,j,i) = n_0*ii_f/SQR(SQR(n0_f));
    }
  });

  return;
}
