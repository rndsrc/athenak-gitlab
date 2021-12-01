//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_rad_shadow.cpp
//  \brief Shadow test (flat space)

// C++ headers
#include <algorithm>  // min, max
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "radiation/radiation.hpp"
#include "srcterms/srcterms.hpp"
#include "pgen.hpp"

#include "radiation/radiation_tetrad.hpp"

int nangles_;
radiation::Radiation *my_prad;

void ShadowInnerX1(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &cc,
                   bool hydro_flag, bool rad_flag);

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for GR radiation beam test

void ProblemGenerator::UserProblem(MeshBlockPack *pmbp, ParameterInput *pin)
{
  // capture variables for kernel
  auto &indcs = pmbp->coord.coord_data.mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  nangles_ = pmbp->prad->.nangles;
  my_prad = pmbp->prad;

  auto &coord = pmbp->coord.coord_data;
  int nmb1 = (pmbp->nmb_thispack-1);

  if (pmbp->phydro != nullptr) {
    auto &w0 = pmbp->phydro->w0;
    par_for("pgen_shadow1",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i)
      {
        Real &x1min = coord.mb_size.d_view(m).x1min;
        Real &x1max = coord.mb_size.d_view(m).x1max;
        int nx1 = coord.mb_indcs.nx1;
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

        Real &x2min = coord.mb_size.d_view(m).x2min;
        Real &x2max = coord.mb_size.d_view(m).x2max;
        int nx2 = coord.mb_indcs.nx2;
        Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

        // radiation field
        w0(m,IDN,k,j,i) = 1.0 + 9.0/(1.0+exp(10.0*(SQR(x1v/0.1)+SQR(x2v/0.06)-1.0)));
        w0(m,IVX,k,j,i) = 0.0;
        w0(m,IVY,k,j,i) = 0.0;
        w0(m,IVZ,k,j,i) = 0.0;
        w0(m,IPR,k,j,i) = 1.0;
      }
    );
    // Convert primitives to conserved
    auto &u0 = pmbp->phydro->u0;
    pmbp->phydro->peos->PrimToCons(w0, u0);
  }

  if (pmbp->prad != nullptr) {
    auto &i0 = pmbp->prad->i0;
    par_for("pgen_shadow2",DevExeSpace(),0,nmb1,0,nangles_-1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int lm, int k, int j, int i)
      {
        i0(m,lm,k,j,i) = 0.0;
      }
    );
  }

  // Enroll boundary function
  if (pin->GetString("mesh", "ix1_bc")=="user") {
    pmbp->pmesh->EnrollBoundaryFunction(BoundaryFace::inner_x1, ShadowInnerX1);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn ShadowInnerX1
//  \brief Sets boundary condition on inner X1 boundary
// Note quantities at this boundary are held Shadow to initial condition values

void ShadowInnerX1(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &cc,
                   bool hydro_flag, bool rad_flag)
{
  auto &indcs = coord.mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;

  if (hydro_flag) {
    int nvar = cc.extent_int(1);
    par_for("outflow_ix1",DevExeSpace(),0,(nvar-1),0,(n3-1),0,(n2-1),0,(ng-1),
      KOKKOS_LAMBDA(int n, int k, int j, int i)
      {
        cc(m,n,k,j,is-i-1) = cc(m,n,k,j,is);
      }
    );
  }

  if (rad_flag) {
    Real my_spread = 30.0;
    Real dir_1_ = 1.0;
    Real dir_2_ = 0.0;
    Real dir_3_ = 0.0;
    auto nh_c_ = my_prad->nh_c;

    par_for("shadow_ix1",DevExeSpace(),0,(n3-1),0,(n2-1),0,(ng-1),
      KOKKOS_LAMBDA(int k, int j, int i)
      {
        Real &x1min = coord.mb_size.d_view(m).x1min;
        Real &x1max = coord.mb_size.d_view(m).x1max;
        int nx1 = coord.mb_indcs.nx1;
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

        Real &x2min = coord.mb_size.d_view(m).x2min;
        Real &x2max = coord.mb_size.d_view(m).x2max;
        int nx2 = coord.mb_indcs.nx2;
        Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

        Real &x3min = coord.mb_size.d_view(m).x3min;
        Real &x3max = coord.mb_size.d_view(m).x3max;
        int nx3 = coord.mb_indcs.nx3;
        Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

        // compute metric and inverse and tetrad
        Real g_[NMETRIC], gi_[NMETRIC];
        ComputeMetricAndInverse(x1v, x2v, x3v, true, coord.snake,
                                coord.bh_mass, coord.bh_spin, g_, gi_);
        Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
        ComputeTetrad(x1v, x2v, x3v, coord.snake, coord.bh_mass, coord.bh_spin,
                      e, e_cov, omega);

        Real mu_min = cos(my_spread/2.0 * M_PI/180.0);
        // Calculate contravariant time component of direction
        Real temp_a = g_[I00];
        Real temp_b = 2.0*(g_[I01]*dir_1_ + g_[I02]*dir_2_ + g_[I03]*dir_3_);
        Real temp_c = g_[I11]*SQR(dir_1_) + 2.0*g_[I12]*dir_1_*dir_2_
                      + 2.0*g_[I13]*dir_1_*dir_3_ + g_[I22]*SQR(dir_2_)
                      + 2.0*g_[I23]*dir_2_*dir_3_ + g_[I33]*SQR(dir_3_);
        Real dir_0 = ((-temp_b - sqrt(SQR(temp_b) - 4.0*temp_a*temp_c))
                      / (2.0*temp_a));

        // lower indices
        Real dc0 = g_[I00]*dir_0 + g_[I01]*dir_1_ + g_[I02]*dir_2_ + g_[I03]*dir_3_;
        Real dc1 = g_[I01]*dir_0 + g_[I11]*dir_1_ + g_[I12]*dir_2_ + g_[I13]*dir_3_;
        Real dc2 = g_[I02]*dir_0 + g_[I12]*dir_1_ + g_[I22]*dir_2_ + g_[I23]*dir_3_;
        Real dc3 = g_[I03]*dir_0 + g_[I13]*dir_1_ + g_[I23]*dir_2_ + g_[I33]*dir_3_;

        // Calculate covariant direction in tetrad frame
        Real dtc0 = (e[0][0]*dc0 + e[0][1]*dc1 + e[0][2]*dc2 + e[0][3]*dc3);
        Real dtc1 = (e[1][0]*dc0 + e[1][1]*dc1 + e[1][2]*dc2 + e[1][3]*dc3)/(-dtc0);
        Real dtc2 = (e[2][0]*dc0 + e[2][1]*dc1 + e[2][2]*dc2 + e[2][3]*dc3)/(-dtc0);
        Real dtc3 = (e[3][0]*dc0 + e[3][1]*dc1 + e[3][2]*dc2 + e[3][3]*dc3)/(-dtc0);

        for (int lm=0; lm<nangles_; ++lm) {
          Real mu = (nh_c_.d_view(lm,1)*dtc1
                   + nh_c_.d_view(lm,2)*dtc2
                   + nh_c_.d_view(lm,3)*dtc3);
          Real theta = acos(mu)*180.0/M_PI;
          if (fabs(theta) < 20.0) {
            cc(m,lm,k,j,(is-i-1)) = 100.0;
          } else {
            cc(m,lm,k,j,(is-i-1)) = 0.0;
          }
        }
      }
    );
  }
  return;
}
