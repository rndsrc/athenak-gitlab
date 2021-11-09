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

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "turb_driver.hpp"
#include "srcterms.hpp"

#include "radiation/radiation_tetrad.hpp"


//----------------------------------------------------------------------------------------
// constructor, parses input file and initializes data structures and parameters
// Only source terms specified in input file are initialized.  If none requested,
// 'source_terms_enabled' flag is false.

SourceTerms::SourceTerms(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp),
  source_terms_enabled(false)
{
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

  // (3) beam source (radiation)
  beam_source = pin->GetOrAddBoolean(block,"beam_source",false);
  if (beam_source) {
    source_terms_enabled = true;
    pos_1 = pin->GetReal(block, "pos_1");
    pos_2 = pin->GetReal(block, "pos_2");
    pos_3 = pin->GetReal(block, "pos_3");
    width = pin->GetReal(block, "width");
    dir_1 = pin->GetReal(block, "dir_1");
    dir_2 = pin->GetReal(block, "dir_2");
    dir_3 = pin->GetReal(block, "dir_3");
    spread = pin->GetReal(block, "spread");
    dii_dt = pin->GetReal(block, "dii_dt");
  }

  // TODO: finish implementing cooling
  // (4) Optically thin (ISM) cooling
  ism_cooling = pin->GetOrAddBoolean(block,"ism_cooling",false);
  if (ism_cooling) {
    source_terms_enabled = true;
    mbar   = pin->GetReal(block,"mbar");
    kboltz = pin->GetReal(block,"kboltz");
    hrate  = pin->GetReal(block,"heating_rate");
  }
}

//----------------------------------------------------------------------------------------
// destructor
  
SourceTerms::~SourceTerms()
{
}

//----------------------------------------------------------------------------------------
//! \fn 
// Add constant acceleration
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::AddConstantAccel(DvceArray5D<Real> &u0, const DvceArray5D<Real> &w0,
                                   const Real bdt)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  Real &g = const_accel_val;
  int &dir = const_accel_dir;

  par_for("const_acc", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i)
    {
      Real src = bdt*g*w0(m,IDN,k,j,i);
      u0(m,dir,k,j,i) += src;
      if ((u0.extent_int(1) - 1) == IEN) { u0(m,IEN,k,j,i) += src*w0(m,dir,k,j,i); }
    }
  );
  return;
}

//----------------------------------------------------------------------------------------
//! \fn
// Add beam of radiation
// NOTE Radiation beam source terms calculation does not depend on values stored in i0.
// Rather, it directly updates i0.

void SourceTerms::AddBeamSource(DvceArray5D<Real> &i0, const Real bdt)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  auto nangles_ = pmy_pack->prad->nangles;

  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto coord = pmy_pack->coord.coord_data;

  auto nh_c_ = pmy_pack->prad->nh_c;
  auto n0_n_mu_ = pmy_pack->prad->n0_n_mu;

  Real &pos_1_ = pos_1;
  Real &pos_2_ = pos_2;
  Real &pos_3_ = pos_3;
  Real &width_ = width;
  Real &dir_1_ = dir_1;
  Real &dir_2_ = dir_2;
  Real &dir_3_ = dir_3;
  Real &spread_ = spread;
  Real &dii_dt_ = dii_dt;

  par_for("beam_source", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i)
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

      Real g_[NMETRIC], gi_[NMETRIC];
      ComputeMetricAndInverse(x1v, x2v, x3v, true,
                              coord.bh_mass, coord.bh_spin, g_, gi_);
      Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
      ComputeTetrad(x1v, x2v, x3v, coord.bh_mass, coord.bh_spin, e, e_cov, omega);

      // Calculate proper distance to beam origin and minimum angle between directions
      Real dx1 = x1v - pos_1_;
      Real dx2 = x2v - pos_2_;
      Real dx3 = x3v - pos_3_;
      Real dx_sq = g_[I11] * SQR(dx1) + 2.0 * g_[I12] * dx1 * dx2
                   + 2.0 * g_[I13] * dx1 * dx3 + g_[I22] * SQR(dx2)
                   + 2.0 * g_[I23] * dx2 * dx3 + g_[I33] * SQR(dx3);
      Real mu_min = cos(spread_/2.0 * M_PI/180.0);

      // Calculate contravariant time component of direction
      Real temp_a = g_[I00];
      Real temp_b = 2.0 * (g_[I01] * dir_1_ + g_[I02] * dir_2_ + g_[I03] * dir_3_);
      Real temp_c = g_[I11] * SQR(dir_1_) + 2.0 * g_[I12] * dir_1_ * dir_2_
                    + 2.0 * g_[I13] * dir_1_ * dir_3_ + g_[I22] * SQR(dir_2_)
                    + 2.0 * g_[I23] * dir_2_ * dir_3_ + g_[I33] * SQR(dir_3_);
      Real dir_0 = ((-temp_b - sqrt(SQR(temp_b) - 4.0 * temp_a * temp_c))
                    / (2.0 * temp_a));

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

      // Go through angles
      for (int lm=0; lm<nangles_; ++lm) {
        Real mu = (nh_c_.d_view(lm,1) * dtc1
                 + nh_c_.d_view(lm,2) * dtc2
                 + nh_c_.d_view(lm,3) * dtc3);
        Real dcons_dt = 0.;
        if (dx_sq < SQR(width_/2.0) && mu > mu_min) {
          dcons_dt = dii_dt_;
        }
        i0(m,lm,k,j,i) += dcons_dt*bdt;
      }
    }
  );
  return;
}

//----------------------------------------------------------------------------------------
//! \fn 
// Add Shearing box source terms in the momentum and energy equations for Hydro.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::AddShearingBox(DvceArray5D<Real> &u0, const DvceArray5D<Real> &w0,
                                   const Real bdt)
{ 
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
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
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i)
    {
      Real &den = w0(m,IDN,k,j,i);
      Real mom1 = den*w0(m,IVX,k,j,i);
      Real mom3 = den*w0(m,IVZ,k,j,i);
      u0(m,IM1,k,j,i) += 2.0*bdt*(omega0_*mom3);
      u0(m,IM3,k,j,i) += (qshear_ - 2.0)*bdt*omega0_*mom1;
      if ((u0.extent_int(1) - 1) == IEN) { u0(m,IEN,k,j,i) += qo*bdt*(mom1*mom3/den); }
    }
  );

}

//----------------------------------------------------------------------------------------
//! \fn 
// Add Shearing box source terms in the momentum and energy equations for Hydro.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars
  
void SourceTerms::AddShearingBox(DvceArray5D<Real> &u0, const DvceArray5D<Real> &w0,
                                 const DvceArray5D<Real> &bcc0, const Real bdt)
{ 
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
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
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i)
    { 
      Real &den = w0(m,IDN,k,j,i);
      Real mom1 = den*w0(m,IVX,k,j,i);
      Real mom3 = den*w0(m,IVZ,k,j,i);
      u0(m,IM1,k,j,i) += 2.0*bdt*(omega0_*mom3);
      u0(m,IM3,k,j,i) += (qshear_ - 2.0)*bdt*omega0_*mom1; 
      if ((u0.extent_int(1) - 1) == IEN) {
        u0(m,IEN,k,j,i) -= qo*bdt*(bcc0(m,IBX,k,j,i)*bcc0(m,IBZ,k,j,i) - mom1*mom3/den);
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::AddSBoxEField
//  \brief Add electric field in rotating frame E = - (v_{K} x B) where v_{K} is
//  background orbital velocity v_{K} = - q \Omega x in the toriodal (\phi or y) direction
//  See SG eqs. [49-52] (eqs for orbital advection), and [60]

void SourceTerms::AddSBoxEField(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
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
    auto &coord = pmy_pack->coord.coord_data;
    auto e1 = efld.x1e;
    auto e2 = efld.x2e;
    auto b1 = b0.x1f;
    auto b2 = b0.x2f;
    par_for_outer("acc0", DevExeSpace(), scr_size, scr_level, 0, nmb1, js, je+1,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j)
      {
        par_for_inner(member, is, ie+1, [&](const int i)
        {
          Real &x1min = coord.mb_size.d_view(m).x1min;
          Real &x1max = coord.mb_size.d_view(m).x1max;
          int nx1 = coord.mb_indcs.nx1;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          e1(m,ks,  j,i) -= qomega*x1v*b2(m,ks,j,i);
          e1(m,ke+1,j,i) -= qomega*x1v*b2(m,ks,j,i);

          Real x1f = LeftEdgeX(i-is, nx1, x1min, x1max);
          e2(m,ks  ,j,i) += qomega*x1f*b1(m,ks,j,i);
          e2(m,ke+1,j,i) += qomega*x1f*b1(m,ks,j,i);
        });
      }
    );
  }
  // TODO: add 3D shearing box

  return;
}
