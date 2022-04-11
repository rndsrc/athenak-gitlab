//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_rad_beam.cpp
//  \brief Beam test for radiation

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

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // capture variables for kernel
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb1 = (pmbp->nmb_thispack-1);
  int nang1 = (pmbp->prad->nangles-1);

  auto &i0 = pmbp->prad->i0;
  par_for("rad_beam",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    i0(m,n,k,j,i) = 0.0;
  });

  return;
}
