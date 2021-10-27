//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_rad_beam.cpp
//  \brief Beam test for radiation (in flat space)

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

void ProblemGenerator::UserProblem(MeshBlockPack *pmbp, ParameterInput *pin)
{
  // capture variables for kernel
  auto &indcs = pmbp->coord.coord_data.mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  auto &aindcs = pmbp->prad->amesh_indcs;
  int &zs = aindcs.zs; int &ze = aindcs.ze;
  int &ps = aindcs.ps; int &pe = aindcs.pe;

  auto &i0 = pmbp->prad->i0;
  int nmb1 = (pmbp->nmb_thispack-1);

  par_for("rad_beam",DevExeSpace(),0,nmb1,zs,ze,ps,pe,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int z, int p, int k, int j, int i)
    {
      // radiation field
      int zp = AngleInd(z,p,false,false,aindcs);
      i0(m,zp,k,j,i) = 0.0;
    }
  );

  return;
}

