//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_rad_rekax.cpp
//  \brief Relaxation test (flat space)

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

void OpacityRelax(Real rho, Real temp, Real& kappa_a, Real& kappa_s, Real& kappa_p);

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

  int nangles_ = pmbp->prad->nangles;

  auto &coord = pmbp->coord.coord_data;
  int nmb1 = (pmbp->nmb_thispack-1);

  Real erad = pin->GetReal("problem", "erad");
  Real temp = pin->GetReal("problem", "temp");

  auto nmu_ = pmbp->prad->nmu;
  auto solid_angle_ = pmbp->prad->solid_angle;

  if (pmbp->phydro != nullptr) {
    auto &w0 = pmbp->phydro->w0;
    par_for("pgen_rekax1",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
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
        w0(m,IDN,k,j,i) = 1.0;
        w0(m,IVX,k,j,i) = 0.0;
        w0(m,IVY,k,j,i) = 0.0;
        w0(m,IVZ,k,j,i) = 0.0;
        w0(m,IPR,k,j,i) = temp;
      }
    );
    // Convert primitives to conserved
    auto &u0 = pmbp->phydro->u0;
    pmbp->phydro->peos->PrimToCons(w0, u0);
  }

  if (pmbp->prad != nullptr) {
    auto &i0 = pmbp->prad->i0;
    par_for("pgen_rekax2",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i)
      {
        Real wght_erad = 0.0;
        for (int lm=0; lm<nangles_; ++lm) {
          i0(m,lm,k,j,i) = 1.0;
          wght_erad += (SQR(nmu_(m,lm,k,j,i,0))*i0(m,lm,k,j,i)*solid_angle_.d_view(lm));
        }
        for (int lm=0; lm<nangles_; ++lm) {
          i0(m,lm,k,j,i) = erad*i0(m,lm,k,j,i)/wght_erad;
        }
      }
    );
  }

  // Enroll opacity function
  pmbp->prad->EnrollOpacityFunction(OpacityRelax);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void OpacityRelax(const Real rho, const Real temp,
//                        Real& kappa_a, Real& kappa_s, Real& kappa_p)
//  \brief Sets opacities for relaxation problem

void OpacityRelax(const Real rho, const Real temp,
                  Real& kappa_a, Real& kappa_s, Real& kappa_p)
{
  kappa_a = 100.0/rho;
  kappa_s = 0.0;
  kappa_p = 0.0;
}
