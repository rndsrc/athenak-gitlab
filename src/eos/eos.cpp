//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos.cpp
//! \brief implements constructor and some fns for EquationOfState abstract base class

#include <float.h>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"

//----------------------------------------------------------------------------------------
// EquationOfState constructor

EquationOfState::EquationOfState(std::string bk, MeshBlockPack* pp, ParameterInput *pin) :
    pmy_pack(pp) {
  eos_data.dfloor = pin->GetOrAddReal(bk,"dfloor",(FLT_MIN));
  eos_data.pfloor = pin->GetOrAddReal(bk,"pfloor",(FLT_MIN));
  eos_data.tfloor = pin->GetOrAddReal(bk,"tfloor",(FLT_MIN));
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief No-Op versions of hydro and MHD conservative to primitive functions.
//! Required because each derived class overrides only one.

void EquationOfState::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                                 const int il, const int iu, const int jl, const int ju,
                                 const int kl, const int ku) {
}

void EquationOfState::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                                 DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                                 const int il, const int iu, const int jl, const int ju,
                                 const int kl, const int ku) {
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCon()
//! \brief No-Op versions of hydro and MHD primitive to conservative functions.
//! Required because each derived class overrides only one.

void EquationOfState::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                                 const int il, const int iu, const int jl, const int ju,
                                 const int kl, const int ku) {
}
void EquationOfState::PrimToCons(const DvceArray5D<Real> &prim,
                                 const DvceArray5D<Real> &bcc, DvceArray5D<Real> &cons,
                                 const int il, const int iu, const int jl, const int ju,
                                 const int kl, const int ku) {
}


void EquationOfState::ConsToPrim4thOrder(
  DvceArray5D<Real> &cons, DvceArray5D<Real> &cons_ctr, 
   DvceArray5D<Real> &prim, //DvceArray5D<Real> &prim_ave2, // prim == prim averaged 
  const int il, const int iu, const int jl, const int ju,
  const int kl, const int ku) {
  
  // RegionSize &size = pmy_pack->pmb->mb_size; 
  
  // will need the resolution in each direction later on, right now everything factors out for uniform grid, even with mesh-refinement
  auto C = 1.0/24.0;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;

  int nvar = nhyd+nscal;
  
  // Constructing cell-centered conserved variable from the cell-average through a 5d loop
  
  if (pmy_pack->pmesh->one_d) {  // 1-d only along i-direction
    par_for("cons_to_prim_4th_order", DevExeSpace(),0, (nmb-1), 0, (nvar-1), kl, ku,jl, ju, il+1, iu-1,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i){
      cons_ctr(m,n,k,j,i) = cons(m,n,k,j,i) - C*((cons(m,n,k,j,i-1)-2*cons(m,n,k,j,i)+cons(m,n,k,j,i+1))); //- C1*((cons(m,n,k-1,j,i)-2*cons(m,n,k,j,i)+cons(m,n,k+1,j,i))/(h3*h3)
    });
  } else if (pmy_pack->pmesh->two_d) {  // 2-d along ij direction
    par_for("cons_to_prim_4th_order", DevExeSpace(),0, (nmb-1), 0, (nvar-1), kl, ku, jl+1, ju-1, il+1, iu-1,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i){
        cons_ctr(m,n,k,j,i) = cons(m,n,k,j,i) - C*((cons(m,n,k,j,i-1)-2*cons(m,n,k,j,i)+cons(m,n,k,j,i+1))+
          (cons(m,n,k,j-1,i)-2*cons(m,n,k,j,i)+cons(m,n,k,j+1,i))); 
    });
  } else if (pmy_pack->pmesh->three_d) { // 3-d
    par_for("cons_to_prim_4th_order", DevExeSpace(),0, (nmb-1), 0, (nvar-1), kl+1, ku-1, jl+1, ju-1, il+1, iu-1,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i){
        cons_ctr(m,n,k,j,i) = cons(m,n,k,j,i) - C*((cons(m,n,k,j,i-1)-2*cons(m,n,k,j,i)+cons(m,n,k,j,i+1))+
          (cons(m,n,k,j-1,i)-2*cons(m,n,k,j,i)+cons(m,n,k,j+1,i))+
          (cons(m,n,k-1,j,i)-2*cons(m,n,k,j,i)+cons(m,n,k+1,j,i))); 
    });
  }
  // 4th Order cell-centered primitive variable, saved in prim to save memory
  ConsToPrim(cons_ctr,prim, il, iu, jl, ju, kl, ku);

  //cell-averaged, note that we are rewriting prim averaged into cons_centered
  
  //write 2nd order cell-averaged primitive variables into cons_centered to save memory
  ConsToPrim(cons,cons_ctr,il, iu, jl, ju, kl, ku);
  
  
  if (pmy_pack->pmesh->one_d) { // 1-d
    par_for("cons_to_prim_4th_order_2", DevExeSpace(),0, (nmb-1), 0, (nvar-1), kl, ku,jl, ju, il+1, iu-1,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i){
      prim(m,n,k,j,i) += C*((cons_ctr(m,n,k,j,i-1)-2*cons_ctr(m,n,k,j,i)+cons_ctr(m,n,k,j,i+1)));
    });
  } else if (pmy_pack->pmesh->two_d) { // 2-d
    par_for("cons_to_prim_4th_order_2", DevExeSpace(),0, (nmb-1), 0, (nvar-1), kl, ku, jl+1, ju-1, il+1, iu-1,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i){
        prim(m,n,k,j,i) += C*((cons_ctr(m,n,k,j,i-1)-2*cons_ctr(m,n,k,j,i)+cons_ctr(m,n,k,j,i+1))+
          (cons_ctr(m,n,k,j-1,i)-2*cons_ctr(m,n,k,j,i)+cons_ctr(m,n,k,j+1,i)));
    });
  } else if (pmy_pack->pmesh->three_d) { // 3-d
    par_for("cons_to_prim_4th_order_2", DevExeSpace(),0, (nmb-1), 0, (nvar-1), kl+1, ku-1, jl+1, ju-1, il+1, iu-1,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i){
        prim(m,n,k,j,i) += C*((cons_ctr(m,n,k,j,i-1)-2*cons_ctr(m,n,k,j,i)+cons_ctr(m,n,k,j,i+1))+
          (cons_ctr(m,n,k,j-1,i)-2*cons_ctr(m,n,k,j,i)+cons_ctr(m,n,k,j+1,i))+
          (cons_ctr(m,n,k-1,j,i)-2*cons_ctr(m,n,k,j,i)+cons_ctr(m,n,k+1,j,i))); 
    });
  }
  return;
}