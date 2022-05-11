//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.cpp
//  Implements various (physics) source terms to be added to the Hydro or MHD equations.
//  Currently [constant_acceleration, shearing_box] are implemented.
//  Source terms objects are stored in the respective fluid class, so that Hydro/MHD can
//  have different source terms

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "turb_driver.hpp"
#include "srcterms.hpp"
#include "ismcooling.hpp"
#include "units/units.hpp"

//----------------------------------------------------------------------------------------
// constructor, parses input file and initializes data structures and parameters
// Only source terms specified in input file are initialized.  If none requested,
// 'source_terms_enabled' flag is false.

SourceTerms::SourceTerms(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp),
  source_terms_enabled(false) {
  // (1) (constant) gravitational acceleration
  const_accel = pin->GetOrAddBoolean(block,"const_accel",false);
  if (const_accel) {
    source_terms_enabled = true;
    const_accel_val = pin->GetReal(block,"const_accel_val");
    const_accel_dir = pin->GetInteger(block,"const_accel_dir");
    if (const_accel_dir < 1 || const_accel_dir > 3) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "const_accle_dir must be 1,2, or 3" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // (2) shearing box (hydro and MHD)
  shearing_box = pin->GetOrAddBoolean(block,"shearing_box",false);
  if (shearing_box) {
    source_terms_enabled = true;
    qshear = pin->GetReal(block,"qshear");
    omega0 = pin->GetReal(block,"omega0");
  }

  // (3) Optically thin (ISM) cooling
  ism_cooling = pin->GetOrAddBoolean(block,"ism_cooling",false);
  if (ism_cooling) {
    source_terms_enabled = true;
    hrate = pin->GetReal(block,"hrate");
  }

  // (4) beam source (radiation)
  beam_source = pin->GetOrAddBoolean(block,"beam_source",false);
  if (beam_source) {
    source_terms_enabled = true;
    pos_1 = pin->GetReal(block, "pos_1");
    pos_2 = pin->GetReal(block, "pos_2");
    pos_3 = pin->GetReal(block, "pos_3");
    dir_1 = pin->GetReal(block, "dir_1");
    dir_2 = pin->GetReal(block, "dir_2");
    dir_3 = pin->GetReal(block, "dir_3");
    width = pin->GetReal(block, "width");
    spread = pin->GetReal(block, "spread");
    dii_dt = pin->GetReal(block, "dii_dt");
  }
}

//----------------------------------------------------------------------------------------
// destructor

SourceTerms::~SourceTerms() {
}

//----------------------------------------------------------------------------------------
//! \fn
// Add constant acceleration
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::AddConstantAccel(DvceArray5D<Real> &u0, const DvceArray5D<Real> &w0,
                                   const Real bdt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  Real &g = const_accel_val;
  int &dir = const_accel_dir;

  par_for("const_acc", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real src = bdt*g*w0(m,IDN,k,j,i);
    u0(m,dir,k,j,i) += src;
    if ((u0.extent_int(1) - 1) == IEN) { u0(m,IEN,k,j,i) += src*w0(m,dir,k,j,i); }
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn
// Add Shearing box source terms in the momentum and energy equations for Hydro.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::AddShearingBox(DvceArray5D<Real> &u0, const DvceArray5D<Real> &w0,
                                   const Real bdt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  //  Terms are implemented with orbital advection, so that v3 represents the perturbation
  //  from the Keplerian flow v_{K} = - q \Omega x
  Real &omega0_ = omega0;
  Real &qshear_ = qshear;
  Real qo  = qshear*omega0;

  par_for("sbox", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &den = w0(m,IDN,k,j,i);
    Real mom1 = den*w0(m,IVX,k,j,i);
    Real mom3 = den*w0(m,IVZ,k,j,i);
    u0(m,IM1,k,j,i) += 2.0*bdt*(omega0_*mom3);
    u0(m,IM3,k,j,i) += (qshear_ - 2.0)*bdt*omega0_*mom1;
    if ((u0.extent_int(1) - 1) == IEN) { u0(m,IEN,k,j,i) += qo*bdt*(mom1*mom3/den); }
  });
}

//----------------------------------------------------------------------------------------
//! \fn
// Add Shearing box source terms in the momentum and energy equations for Hydro.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::AddShearingBox(DvceArray5D<Real> &u0, const DvceArray5D<Real> &w0,
                                 const DvceArray5D<Real> &bcc0, const Real bdt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  //  Terms are implemented with orbital advection, so that v3 represents the perturbation
  //  from the Keplerian flow v_{K} = - q \Omega x
  Real &omega0_ = omega0;
  Real &qshear_ = qshear;
  Real qo  = qshear*omega0;

  par_for("sbox", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &den = w0(m,IDN,k,j,i);
    Real mom1 = den*w0(m,IVX,k,j,i);
    Real mom3 = den*w0(m,IVZ,k,j,i);
    u0(m,IM1,k,j,i) += 2.0*bdt*(omega0_*mom3);
    u0(m,IM3,k,j,i) += (qshear_ - 2.0)*bdt*omega0_*mom1;
    if ((u0.extent_int(1) - 1) == IEN) {
      u0(m,IEN,k,j,i) -= qo*bdt*(bcc0(m,IBX,k,j,i)*bcc0(m,IBZ,k,j,i) - mom1*mom3/den);
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::AddSBoxEField
//  \brief Add electric field in rotating frame E = - (v_{K} x B) where v_{K} is
//  background orbital velocity v_{K} = - q \Omega x in the toriodal (\phi or y) direction
//  See SG eqs. [49-52] (eqs for orbital advection), and [60]

void SourceTerms::AddSBoxEField(const DvceFaceFld4D<Real> &b0,
                                DvceEdgeFld4D<Real> &efld) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  Real qomega  = qshear*omega0;

  int nmb1 = pmy_pack->nmb_thispack - 1;
  size_t scr_size = 0;
  int scr_level = 0;

  //---- 2-D problem:
  // electric field E = - (v_{K} x B), where v_{K} is in the z-direction.  Thus
  // E_{x} = -(v x B)_{x} = -(vy*bz - vz*by) = +v_{K}by --> E1 = -(q\Omega x)b2
  // E_{y} = -(v x B)_{y} =  (vx*bz - vz*bx) = -v_{K}bx --> E2 = +(q\Omega x)b1
  if (pmy_pack->pmesh->two_d) {
    auto e1 = efld.x1e;
    auto e2 = efld.x2e;
    auto b1 = b0.x1f;
    auto b2 = b0.x2f;
    par_for_outer("acc0", DevExeSpace(), scr_size, scr_level, 0, nmb1, js, je+1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j) {
      par_for_inner(member, is, ie+1, [&](const int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

        e1(m,ks,  j,i) -= qomega*x1v*b2(m,ks,j,i);
        e1(m,ke+1,j,i) -= qomega*x1v*b2(m,ks,j,i);

        Real x1f = LeftEdgeX(i-is, nx1, x1min, x1max);
        e2(m,ks  ,j,i) += qomega*x1f*b1(m,ks,j,i);
        e2(m,ke+1,j,i) += qomega*x1f*b1(m,ks,j,i);
      });
    });
  }
  // TODO(@user): add 3D shearing box

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::AddISMCooling()
//! \brief Add explict ISM cooling and heating source terms in the energy equations.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::AddISMCooling(DvceArray5D<Real> &u0, const DvceArray5D<Real> &w0,
                                const EOS_Data &eos_data, const Real bdt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real use_e = eos_data.use_e;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real heating_rate = hrate;
  Real temp_unit = pmy_pack->punit->temperature_cgs();
  Real n_unit = pmy_pack->punit->density_cgs()/pmy_pack->punit->mu()
                /pmy_pack->punit->atomic_mass_unit_cgs;
  Real cooling_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                      /n_unit/n_unit;
  Real heating_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()/n_unit;

  par_for("cooling", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // temperature in cgs unit
    Real temp = 1.0;
    if (use_e) {
      temp = temp_unit*w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    } else {
      temp = temp_unit*w0(m,ITM,k,j,i);
    }

    Real lambda_cooling = ISMCoolFn(temp)/cooling_unit;
    Real gamma_heating = heating_rate/heating_unit;

    u0(m,IEN,k,j,i) -= bdt * w0(m,IDN,k,j,i) *
                        (w0(m,IDN,k,j,i) * lambda_cooling - gamma_heating);
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn
// Add beam of radiation
// NOTE Radiation beam source terms calculation does not depend on values stored in i0.
// Rather, it directly updates i0.

void SourceTerms::AddBeamSource(DvceArray5D<Real> &i0, const Real bdt) {
  Real &p1 = pos_1;
  Real &p2 = pos_2;
  Real &p3 = pos_3;
  Real &d1 = dir_1;
  Real &d2 = dir_2;
  Real &d3 = dir_3;
  Real &width_ = width;
  Real &spread_ = spread;
  Real &dii_dt_ = dii_dt;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  int nmb1 = (pmy_pack->nmb_thispack-1);
  int nang1 = (pmy_pack->prad->nangles-1);

  auto &coord = pmy_pack->pcoord->coord_data;
  auto nh_c_ = pmy_pack->prad->nh_c;
  auto tet_c_ = pmy_pack->prad->tet_c;
  auto tetcov_c_ = pmy_pack->prad->tetcov_c;

  par_for("beam_source",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
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
    Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
    ComputeTetrad(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin, e, e_cov, omega);

    // Calculate proper distance to beam origin and minimum angle between directions
    Real dx1 = x1v - p1;
    Real dx2 = x2v - p2;
    Real dx3 = x3v - p3;
    Real dx_sq = (g_[I11]*dx1*dx1 + 2.0*g_[I12]*dx1*dx2 + 2.0*g_[I13]*dx1*dx3
                                  +     g_[I22]*dx2*dx2 + 2.0*g_[I23]*dx2*dx3
                                                        +     g_[I33]*dx3*dx3);
    Real mu_min = cos(spread_/2.0*M_PI/180.0);

    // Calculate contravariant time component of direction
    Real temp_a = g_[I00];
    Real temp_b = 2.0*(g_[I01]*d1 + g_[I02]*d2 + g_[I03]*d3);
    Real temp_c = (g_[I11]*d1*d1 + 2.0*g_[I12]*d1*d2 + 2.0*g_[I13]*d1*d3
                                 +     g_[I22]*d2*d2 + 2.0*g_[I23]*d2*d3
                                                     +     g_[I33]*d3*d3);
    Real d0 = ((-temp_b - sqrt(SQR(temp_b) - 4.0*temp_a*temp_c))/(2.0*temp_a));

    // lower indices
    Real dc0 = g_[I00]*d0 + g_[I01]*d1 + g_[I02]*d2 + g_[I03]*d3;
    Real dc1 = g_[I01]*d0 + g_[I11]*d1 + g_[I12]*d2 + g_[I13]*d3;
    Real dc2 = g_[I02]*d0 + g_[I12]*d1 + g_[I22]*d2 + g_[I23]*d3;
    Real dc3 = g_[I03]*d0 + g_[I13]*d1 + g_[I23]*d2 + g_[I33]*d3;

    // Calculate covariant direction in tetrad frame
    Real dtc0 = (tet_c_(m,0,0,k,j,i)*dc0 + tet_c_(m,0,1,k,j,i)*dc1 +
                 tet_c_(m,0,2,k,j,i)*dc2 + tet_c_(m,0,3,k,j,i)*dc3);
    Real dtc1 = (tet_c_(m,1,0,k,j,i)*dc0 + tet_c_(m,1,1,k,j,i)*dc1 +
                 tet_c_(m,1,2,k,j,i)*dc2 + tet_c_(m,1,3,k,j,i)*dc3)/(-dtc0);
    Real dtc2 = (tet_c_(m,2,0,k,j,i)*dc0 + tet_c_(m,2,1,k,j,i)*dc1 +
                 tet_c_(m,2,2,k,j,i)*dc2 + tet_c_(m,2,3,k,j,i)*dc3)/(-dtc0);
    Real dtc3 = (tet_c_(m,3,0,k,j,i)*dc0 + tet_c_(m,3,1,k,j,i)*dc1 +
                 tet_c_(m,3,2,k,j,i)*dc2 + tet_c_(m,3,3,k,j,i)*dc3)/(-dtc0);

    // Go through angles
    for (int n=0; n<=nang1; ++n) {
      Real mu = (nh_c_.d_view(n,1) * dtc1
               + nh_c_.d_view(n,2) * dtc2
               + nh_c_.d_view(n,3) * dtc3);
      if ((dx_sq < SQR(width_/2.0)) && (mu > mu_min)) {
        Real n_0 = 0.0;
        for (int d=0; d<4; ++d) {
          n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);
        }
        i0(m,n,k,j,i) += n_0*dii_dt_*bdt;
      }
    }
  });
  return;
}
