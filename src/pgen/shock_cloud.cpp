//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shock_cloud.cpp
//! \brief Problem generator for shock-cloud problem: a planar shock impacting a single
//! spherical cloud. Input parameters are:
//!    - problem/Mach   = Mach number of incident shock
//!    - problem/drat   = density ratio of cloud to ambient
//!    - problem/beta   = ratio of Pgas/Pmag
//! The cloud radius is fixed at 1.0.  The center of the coordinate system defines the
//! center of the cloud, and should be in the middle of the cloud. The shock is initially
//! at x1=-2.0.  A typical grid domain should span x1 in [-3.0,7.0] , y and z in
//! [-2.5,2.5] (see input file).
//========================================================================================

#include <iostream>
#include <sstream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "coordinates/cell_locations.hpp"

// postshock flow variables are shared with IIB function
namespace {
Real shock_d, shock_m, shock_e;
} // namespace

// fixes BCs on L-x1 (left edge) of grid to postshock flow.
void ShockCloudInnerX1(int m, CoordData &coord, DvceArray5D<Real> &u);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::_()
//! \brief Problem Generator for the shock-cloud interaction problem

void ProblemGenerator::UserProblem(MeshBlockPack *pmbp, ParameterInput *pin)
{
  Real gm = pmbp->phydro->peos->eos_data.gamma;
  Real gm1 = gm - 1.0;

  // Read input parameters
  Real xshock = -2.0;
  Real rad    = 1.0;
  Real mach = pin->GetReal("problem","Mach");
  Real drat = pin->GetReal("problem","drat");

  // Set paramters in ambient medium ("R-state" for shock)
  Real dr = 1.0;
  Real pr = 1.0/gm;
  Real ur = 0.0;

  // Uses Rankine Hugoniot relations for adiabatic gas to initialize problem
  Real jump1 = (gm + 1.0)/(gm1 + 2.0/(mach*mach));
  Real jump2 = (2.0*gm*mach*mach - gm1)/(gm + 1.0);
  Real jump3 = 2.0*(1.0 - 1.0/(mach*mach))/(gm + 1.0);

  Real dl = dr*jump1;
  Real pl = pr*jump2;
  Real ul = ur + jump3*mach*std::sqrt(gm*pr/dr);

  shock_d = dl;
  shock_m = dl*ul;
  shock_e = pl/gm1 + 0.5*dl*(ul*ul);

  // capture variables for the kernel
  auto &indcs = pmbp->coord.coord_data.mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto coord = pmbp->coord.coord_data;

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {

    auto &u0 = pmbp->phydro->u0;
    par_for("pgen_cloud1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m,int k, int j, int i)
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

        // postshock flow
        if (x1v < xshock) {
          u0(m,IDN,k,j,i) = dl;
          u0(m,IM1,k,j,i) = ul*dl;
          u0(m,IM2,k,j,i) = 0.0;
          u0(m,IM3,k,j,i) = 0.0;
          u0(m,IEN,k,j,i) = pl/gm1 + 0.5*dl*(ul*ul);

          // preshock ambient gas
        } else {
          u0(m,IDN,k,j,i) = dr;
          u0(m,IM1,k,j,i) = ur*dr;
          u0(m,IM2,k,j,i) = 0.0;
          u0(m,IM3,k,j,i) = 0.0;
          u0(m,IEN,k,j,i) = pr/gm1 + 0.5*dr*(ur*ur);
        }

        // cloud interior
        Real diag = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
        if (diag < rad) {
          u0(m,IDN,k,j,i) = dr*drat;
          u0(m,IM1,k,j,i) = ur*dr*drat;
          u0(m,IM2,k,j,i) = 0.0;
          u0(m,IM3,k,j,i) = 0.0;
          u0(m,IEN,k,j,i) = pr/gm1 + 0.5*dr*drat*(ur*ur);
        }
      }
    );
  }  // End initialization of Hydro variables

  // Enroll boundary function
  pmbp->pmesh->EnrollBoundaryFunction(BoundaryFace::inner_x1, ShockCloudInnerX1);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ShockCloudInnerX1()
//  \brief Sets boundary condition on left X boundary (iib)
// Note quantities at this boundary are held fixed at the downstream state

void ShockCloudInnerX1(int m, CoordData &coord, DvceArray5D<Real> &u)
{
  auto &indcs = coord.mb_indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;

  Real &shock_d_ = shock_d;
  Real &shock_m_ = shock_m;
  Real &shock_e_ = shock_e;

  par_for("outflow_ix1", DevExeSpace(),0,(n3-1),0,(n2-1),0,(ng-1),
    KOKKOS_LAMBDA(int k, int j, int i)
    {
      u(m,IDN,k,j,is-i-1) = shock_d_;
      u(m,IM1,k,j,is-i-1) = shock_m_;
      u(m,IM2,k,j,is-i-1) = 0.0;
      u(m,IM3,k,j,is-i-1) = 0.0;
      u(m,IEN,k,j,is-i-1) = shock_e_;
    }
  );
  return;
}