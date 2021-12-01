//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_rad_hohlraum.cpp
//  \brief Hohlraum test (flat space)

// C++ headers
#include <algorithm>  // min, max
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "radiation/radiation.hpp"
#include "srcterms/srcterms.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for GR radiation beam test

void HohlraumInnerX1(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &i,
                     bool hydro_flag, bool rad_flag);
void HohlraumOuterX1(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &i,
                     bool hydro_flag, bool rad_flag);
void HohlraumInnerX2(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &i,
                     bool hydro_flag, bool rad_flag);
void HohlraumOuterX2(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &i,
                     bool hydro_flag, bool rad_flag);
void HohlraumInnerX3(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &i,
                     bool hydro_flag, bool rad_flag);
void HohlraumOuterX3(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &i,
                     bool hydro_flag, bool rad_flag);

int nangles_;
Real ii_ix1, ii_ox1;  // x1-boundary radiation intensities
Real ii_ix2, ii_ox2;  // x2-boundary radiation intensities
Real ii_ix3, ii_ox3;  // x3-boundary radiation intensities

void ProblemGenerator::UserProblem(MeshBlockPack *pmbp, ParameterInput *pin)
{
  // capture variables for kernel
  auto &indcs = pmbp->coord.coord_data.mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  nangles_ = pmbp->prad->nangles;

  auto &i0 = pmbp->prad->i0;
  int nmb1 = (pmbp->nmb_thispack-1);

  ii_ix1 = pin->GetReal("problem", "ii_ix1") / (4.*M_PI);
  ii_ox1 = pin->GetReal("problem", "ii_ox1") / (4.*M_PI);
  ii_ix2 = pin->GetReal("problem", "ii_ix2") / (4.*M_PI);
  ii_ox2 = pin->GetReal("problem", "ii_ox2") / (4.*M_PI);
  ii_ix3 = pin->GetReal("problem", "ii_ix3") / (4.*M_PI);
  ii_ox3 = pin->GetReal("problem", "ii_ox3") / (4.*M_PI);

  par_for("rad_beam",DevExeSpace(),0,nmb1,0,nangles_-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int lm, int k, int j, int i)
    {
      // radiation field
      i0(m,lm,k,j,i) = 0.0;
    }
  );

  // Enroll boundary function
  if (pin->GetString("mesh", "ix1_bc")=="user") {
    pmbp->pmesh->EnrollBoundaryFunction(BoundaryFace::inner_x1, HohlraumInnerX1);
  }
  if (pin->GetString("mesh", "ox1_bc")=="user") {
    pmbp->pmesh->EnrollBoundaryFunction(BoundaryFace::outer_x1, HohlraumOuterX1);
  }
  if (pin->GetString("mesh", "ix2_bc")=="user") {
    pmbp->pmesh->EnrollBoundaryFunction(BoundaryFace::inner_x2, HohlraumInnerX2);
  }
  if (pin->GetString("mesh", "ox2_bc")=="user") {
    pmbp->pmesh->EnrollBoundaryFunction(BoundaryFace::outer_x2, HohlraumOuterX2);
  }
  if (pin->GetString("mesh", "ix3_bc")=="user") {
    pmbp->pmesh->EnrollBoundaryFunction(BoundaryFace::inner_x3, HohlraumInnerX3);
  }
  if (pin->GetString("mesh", "ox3_bc")=="user") {
    pmbp->pmesh->EnrollBoundaryFunction(BoundaryFace::outer_x3, HohlraumOuterX3);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn HohlraumInnerX1
//  \brief Sets boundary condition on inner X1 boundary
// Note quantities at this boundary are held Hohlraum to initial condition values

void HohlraumInnerX1(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &ii,
                     bool hydro_flag, bool rad_flag);
{
  auto &indcs = coord.mb_indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;

  if (rad_flag) {
    par_for("hohlraum_ix1", DevExeSpace(),0,(n3-1),0,(n2-1),0,(ng-1),
      KOKKOS_LAMBDA(int k, int j, int i)
      {
        for (int n=0; n<nangles_; ++n) {
          ii(m,n,k,j,(is-i-1)) = ii_ix1;
        }
      }
    );
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn HohlraumOuterrX1
//  \brief Sets boundary condition on outer X1 boundary
// Note quantities at this boundary are held Hohlraum to initial condition values

void HohlraumOuterX1(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &ii,
                     bool hydro_flag, bool rad_flag);
{
  auto &indcs = coord.mb_indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &ie = indcs.ie;

  if (rad_flag) {
    par_for("hohlraum_ox1", DevExeSpace(),0,(n3-1),0,(n2-1),0,(ng-1),
      KOKKOS_LAMBDA(int k, int j, int i)
      {
        for (int n=0; n<nangles_; ++n) {
          ii(m,n,k,j,(ie+i+1)) = ii_ox1;
        }
      }
    );
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn HohlraumInnerX2
//  \brief Sets boundary condition on inner X2 boundary
// Note quantities at this boundary are held Hohlraum to initial condition values

void HohlraumInnerX2(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &ii,
                     bool hydro_flag, bool rad_flag);
{
  auto &indcs = coord.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &js = indcs.js;

  if (rad_flag) {
    par_for("hohlraum_ix2", DevExeSpace(),0,(n3-1),0,(ng-1),0,(n1-1),
      KOKKOS_LAMBDA(int k, int j, int i)
      {
        for (int n=0; n<nangles_; ++n) {
          ii(m,n,k,(js-j-1),i) = ii_ix2;
        }
      }
    );
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn HohlraumOuterX2
//  \brief Sets boundary condition on outer X2 boundary
// Note quantities at this boundary are held Hohlraum to initial condition values

void HohlraumOuterX2(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &ii,
                     bool hydro_flag, bool rad_flag);
{
  auto &indcs = coord.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &je = indcs.je;

  if (rad_flag) {
    par_for("hohlraum_ox2", DevExeSpace(),0,(n3-1),0,(ng-1),0,(n1-1),
      KOKKOS_LAMBDA(int k, int j, int i)
      {
        for (int n=0; n<nangles_; ++n) {
          ii(m,n,k,(je+j+1),i) = ii_ox2;
        }
      }
    );
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn HohlraumInnerX3
//  \brief Sets boundary condition on inner X3 boundary
// Note quantities at this boundary are held Hohlraum to initial condition values

void HohlraumInnerX3(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &ii,
                     bool hydro_flag, bool rad_flag);
{
  auto &indcs = coord.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = indcs.nx2 + 2*ng;
  int &ks = indcs.ks;

  if (rad_flag) {
    par_for("hohlraum_ix3", DevExeSpace(),0,(ng-1),0,(n2-1),0,(n1-1),
      KOKKOS_LAMBDA(int k, int j, int i)
      {
        for (int n=0; n<nangles_; ++n) {
          ii(m,n,(ks-k-1),j,i) = ii_ix3;
        }
      }
    );
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn HohlraumOuterX3
//  \brief Sets boundary condition on outer X3 boundary
// Note quantities at this boundary are held Hohlraum to initial condition values

void HohlraumOuterX3(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &ii,
                     bool hydro_flag, bool rad_flag);
{
  auto &indcs = coord.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = indcs.nx2 + 2*ng;
  int &ke = indcs.ke;

  if (rad_flag) {
    par_for("hohlraum_ox3", DevExeSpace(),0,(ng-1),0,(n2-1),0,(n1-1),
      KOKKOS_LAMBDA(int k, int j, int i)
      {
        for (int n=0; n<nangles_; ++n) {
          ii(m,n,(ke+k+1),j,i) = ii_ox3;
        }
      }
    );
  }

  return;
}

