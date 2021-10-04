//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file outflow_radiation.cpp
//  \brief implementation of outflow BCs for Radiation conserved vars in each dimension
//   BCs applied to a single MeshBlock specified by input integer index to each function

// Athena++ headers
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "radiation/radiation.hpp"

namespace radiation {

//----------------------------------------------------------------------------------------
//! \fn void Radiation::OutflowInnerX1(
//  \brief OUTFLOW boundary conditions, inner x1 boundary

void Radiation::OutflowInnerX1(int m)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int nvar = nangles;
  auto &ci0_ = ci0;

  // project radiation variables in first active cell into ghost zones
  par_for("outflow_ix1", DevExeSpace(),0,(nvar-1),0,(n3-1),0,(n2-1),0,(ng-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {
      ci0_(m,n,k,j,is-i-1) = ci0_(m,n,k,j,is);
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Radiation::OutflowOuterX1(
//  \brief OUTFLOW boundary conditions, outer x1 boundary

void Radiation::OutflowOuterX1(int m)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &ie = indcs.ie;
  int nvar = nangles;
  auto &ci0_ = ci0;

  // project radiation variables in first active cell into ghost zones
  par_for("outflow_ox1", DevExeSpace(),0,(nvar-1),0,(n3-1),0,(n2-1),0,(ng-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {
      ci0_(m,n,k,j,ie+i+1) = ci0_(m,n,k,j,ie);
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Radiation::OutflowInnerX2(
//  \brief OUTFLOW boundary conditions, inner x2 boundary

void Radiation::OutflowInnerX2(int m)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &js = indcs.js;
  int nvar = nangles;
  auto &ci0_ = ci0;

  // project radiation variables in first active cell into ghost zones
  par_for("outflow_ix2", DevExeSpace(),0,(nvar-1),0,(n3-1),0,(ng-1),0,(n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      ci0_(m,n,k,js-j-1,i) =  ci0_(m,n,k,js,i);
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Radiation::OutflowOuterX2(
//  \brief OUTFLOW boundary conditions, outer x2 boundary

void Radiation::OutflowOuterX2(int m)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &je = indcs.je;
  int nvar = nangles;
  auto &ci0_ = ci0;

  // project radiation variables in first active cell into ghost zones
  par_for("outflow_ox2", DevExeSpace(),0,(nvar-1),0,(n3-1),0,(ng-1),0,(n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      ci0_(m,n,k,je+j+1,i) =  ci0_(m,n,k,je,i);
    }   
  );

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Radiation::OutflowInnerX3(
//  \brief OUTFLOW boundary conditions, inner x3 boundary

void Radiation::OutflowInnerX3(int m)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = indcs.nx2 + 2*ng;
  int &ks = indcs.ks;
  int nvar = nangles;
  auto &ci0_ = ci0;

  // project radiation variables in first active cell into ghost zones
  par_for("outflow_ix3", DevExeSpace(),0,(nvar-1),0,(ng-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      ci0_(m,n,ks-k-1,j,i) =  ci0_(m,n,ks,j,i);
    }   
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Radiation::OutflowOuterX3(
//  \brief OUTFLOW boundary conditions, outer x3 boundary

void Radiation::OutflowOuterX3(int m)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = indcs.nx2 + 2*ng;
  int &ke = indcs.ke;
  int nvar = nangles;
  auto &ci0_ = ci0;

  // project radiation variables in first active cell into ghost zones
  par_for("outflow_ox3", DevExeSpace(),0,(nvar-1),0,(ng-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {   
      ci0_(m,n,ke+k+1,j,i) =  ci0_(m,n,ke,j,i);
    }
  );

  return;
}
} // namespace radiation
