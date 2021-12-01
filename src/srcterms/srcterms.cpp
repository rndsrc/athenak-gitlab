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
#include "eos/eos.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "turb_driver.hpp"
#include "srcterms.hpp"

#include "radiation/radiation_tetrad.hpp"

bool FourthPolyRoot(const Real coef4, const Real tconst, Real &root);

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

  // (4) radiation source term (radiation + (M)HD)
  rad_source = pin->GetOrAddBoolean(block,"rad_source",false);
  if (rad_source) {
    source_terms_enabled = true;
    coupling = pin->GetBoolean(block, "coupling");
    arad = pin->GetReal(block, "arad");
    sigma_a = pin->GetReal(block, "sigma_a");
    sigma_p = pin->GetReal(block, "sigma_p");
    sigma_s = pin->GetReal(block, "sigma_s");
  }

  // TODO: finish implementing cooling
  // (5) Optically thin (ISM) cooling
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
      ComputeMetricAndInverse(x1v, x2v, x3v, true, coord.snake,
                              coord.bh_mass, coord.bh_spin, g_, gi_);
      Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
      ComputeTetrad(x1v, x2v, x3v, coord.snake, coord.bh_mass, coord.bh_spin,
                    e, e_cov, omega);

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
// Add implicit radiation source term
// NOTE Only executed on final stage of integration.

void SourceTerms::AddRadiationSourceTerm(DvceArray5D<Real> &i0, DvceArray5D<Real> &i1,
                                         const Real dt)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  auto nangles = pmy_pack->prad->nangles;

  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto coord = pmy_pack->coord.coord_data;

  // Load hydro quantities
  // TODO(@pdmullen) This is mostly a hack... we are not really coupled yet.
  // In future, we need w0 at the start of the time-step (not stage), however,
  // when coupled, w0 will be updated to the primitives after the first stage of
  // integration, will we need a separate register w1 to hold W at beginning of stage.
  auto u0_ = pmy_pack->phydro->u0;
  auto w0_ = pmy_pack->phydro->w0;
  Real gamma_ = pmy_pack->phydro->peos->eos_data.gamma;

  // extract unit system and opacities
  auto arad_ = arad;
  auto sigma_a_ = kappa_a;
  auto sigma_p_ = kappa_p;
  auto sigma_s_ = kappa_s;

  // extract geometric data
  auto nmu_ = pmy_pack->prad->nmu;
  auto nh_c_ = pmy_pack->prad->nh_c;
  auto n_mu_ = pmy_pack->prad->n_mu;
  auto solid_angle_ = pmy_pack->prad->solid_angle;
  auto norm_to_tet_ = pmy_pack->prad->norm_to_tet;

  // extract coupling flag
  bool coupling_ = coupling;

  // compute implicit source term
  par_for("rad_source",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
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

      Real &x3min = coord.mb_size.d_view(m).x3min;
      Real &x3max = coord.mb_size.d_view(m).x3max;
      int nx3 = coord.mb_indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      // compute metric and inverse
      Real g_[NMETRIC], gi_[NMETRIC];
      ComputeMetricAndInverse(x1v, x2v, x3v, true, coord.snake,
                              coord.bh_mass, coord.bh_spin, g_, gi_);

      // fluid state
      Real rho = w0_(m,IDN,k,j,i);
      Real pgas = w0_(m,IPR,k,j,i);
      Real uu1 = w0_(m,IVX,k,j,i);
      Real uu2 = w0_(m,IVY,k,j,i);
      Real uu3 = w0_(m,IVZ,k,j,i);
      Real tgas = pgas/rho;
      Real tgasnew = tgas;
      Real uu0 = sqrt(1.0 + (g_[I11]*SQR(uu1) + 2.0*g_[I12]*uu1*uu2 +
                             2.0*g_[I13]*uu1*uu3 + g_[I22]*SQR(uu2) +
                             2.0*g_[I23]*uu2*uu3 + g_[I33]*SQR(uu3)));

      // compute fluid velocity in tetrad frame
      Real u_tet_[4];
      u_tet_[0] = (norm_to_tet_(m,0,0,k,j,i)*uu0 + norm_to_tet_(m,0,1,k,j,i)*uu1 +
                   norm_to_tet_(m,0,2,k,j,i)*uu2 + norm_to_tet_(m,0,3,k,j,i)*uu3);
      u_tet_[1] = (norm_to_tet_(m,1,0,k,j,i)*uu0 + norm_to_tet_(m,1,1,k,j,i)*uu1 +
                   norm_to_tet_(m,1,2,k,j,i)*uu2 + norm_to_tet_(m,1,3,k,j,i)*uu3);
      u_tet_[2] = (norm_to_tet_(m,2,0,k,j,i)*uu0 + norm_to_tet_(m,2,1,k,j,i)*uu1 +
                   norm_to_tet_(m,2,2,k,j,i)*uu2 + norm_to_tet_(m,2,3,k,j,i)*uu3);
      u_tet_[3] = (norm_to_tet_(m,3,0,k,j,i)*uu0 + norm_to_tet_(m,3,1,k,j,i)*uu1 +
                   norm_to_tet_(m,3,2,k,j,i)*uu2 + norm_to_tet_(m,3,3,k,j,i)*uu3);

      // compute intensities and solid angles in comoving frame
      Real wght_sum_ = 0.0;
      Real omega_cm_[nangles];
      Real intensity_cm_[nangles];
      Real n0_cm_[nangles];
      for (int lm=0; lm<nangles; ++lm) {
        Real un_tet = (u_tet_[1]*nh_c_.d_view(lm,1) + u_tet_[2]*nh_c_.d_view(lm,2) +
                       u_tet_[3]*nh_c_.d_view(lm,3));
        n0_cm_[lm] = (u_tet_[0]*nh_c_.d_view(lm,0) - un_tet);
        omega_cm_[lm] = solid_angle_.d_view(lm)/SQR(n0_cm_[lm]);
        intensity_cm_[lm] = i1(m,lm,k,j,i)*SQR(SQR(n0_cm_[lm]));
        wght_sum_ += omega_cm_[lm];
      }

      // normalization
      for (int lm=0; lm<nangles; ++lm) {
        omega_cm_[lm] /= wght_sum_;
        intensity_cm_[lm] *= 4.0*M_PI;
      }

      // opacities
      Real dtaucsigmaa = dt*sigma_a_/(uu0*sqrt(-gi_[I00]));
      Real dtaucsigmas = dt*sigma_s_/(uu0*sqrt(-gi_[I00]));
      Real dtaucsigmap = dt*sigma_p_/(uu0*sqrt(-gi_[I00]));
      Real dtcsigmaa = dt*sigma_a_;
      Real dtcsigmas = dt*sigma_s_;
      Real dtcsigmap = dt*sigma_p_;

      // Calculate polynomial coefficients
      Real suma1 = 0.0;
      Real suma2 = 0.0;
      Real suma3 = 0.0;
      Real jr_cm = 0.0;
      Real vncsigma2_[nangles];
      for (int lm=0; lm<nangles; ++lm) {
        Real n0_local = nmu_(m,lm,k,j,i,0);
        Real vncsigma = 1.0/(n0_local + (dtcsigmaa + dtcsigmas)*n0_cm_[lm]);
        vncsigma2_[lm] = n0_cm_[lm]*vncsigma;
        Real ir_weight = intensity_cm_[lm]*omega_cm_[lm];
        jr_cm += ir_weight;
        suma1 += omega_cm_[lm]*vncsigma2_[lm];
        suma2 += ir_weight*n0_local*vncsigma;
      }
      suma3 += suma1*(dtcsigmas - dtcsigmap);
      suma1 *= (dtcsigmaa + dtcsigmap);

      // compute coefficients
      Real coef[2] = {0.0};
      coef[1] = ((dtaucsigmaa+dtaucsigmap-(dtaucsigmaa+dtaucsigmap)*suma1
                  / (1.0-suma3))*arad*(gamma_-1.0)/rho);
      coef[0] = (-tgas-(dtaucsigmaa+dtaucsigmap)*suma2*(gamma_-1.0)/(rho*(1.0-suma3)));

      // Calculate new gas temperature
      bool badcell = false;
      if (fabs(coef[1]) > 1.0e-20) {
        bool flag = FourthPolyRoot(coef[1], coef[0], tgasnew);
        if (not flag or isnan(tgasnew)) {
          badcell = true;
          tgasnew = tgas;
        }
      } else {
        tgasnew = -coef[0];
      }

      // Update the comoving frame specific intensity
      Real di_cm[nangles];
      if (not badcell) {
        // Calculate emission coefficient and updated jr_cm
        Real emission = arad*SQR(SQR(tgasnew));
        jr_cm = (suma1*emission + suma2)/(1.0 - suma3);
        for (int lm=0; lm<nangles; ++lm) {
          di_cm[lm] = ( (  (dtcsigmas-dtcsigmap)*jr_cm
                         + (dtcsigmaa+dtcsigmap)*emission
                         - (dtcsigmas+dtcsigmaa)*intensity_cm_[lm])*vncsigma2_[lm]);
        }
      }

      // Apply radiation-fluid coupling to radiation in coordinate frame
      for (int lm=0; lm<nangles; ++lm) {
        i0(m,lm,k,j,i) = (i0(m,lm,k,j,i) + (di_cm[lm]/(4.0*M_PI*SQR(SQR(n0_cm_[lm])))));
      }

      // apply coupling to hydro
      if (coupling_) {
        // compute moments before and after coupling
        Real m_old[4] = {0.0}, m_new[4] = {0.0};
        for (int lm=0; lm<nangles; ++lm) {
          Real sa = solid_angle_.d_view(lm);
          m_old[0] += (nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,0)*i1(m,lm,k,j,i)*sa);
          m_old[1] += (nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,1)*i1(m,lm,k,j,i)*sa);
          m_old[2] += (nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,2)*i1(m,lm,k,j,i)*sa);
          m_old[3] += (nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,3)*i1(m,lm,k,j,i)*sa);
          m_new[0] += (nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,0)*i0(m,lm,k,j,i)*sa);
          m_new[1] += (nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,1)*i0(m,lm,k,j,i)*sa);
          m_new[2] += (nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,2)*i0(m,lm,k,j,i)*sa);
          m_new[3] += (nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,3)*i0(m,lm,k,j,i)*sa);
        }

        // update conserved variables
        u0_(m,IEN,k,j,i) += (m_old[0] - m_new[0]);
        // u0_(m,IM1,k,j,i) += (m_old[1] - m_new[1]);
        // u0_(m,IM2,k,j,i) += (m_old[2] - m_new[2]);
        // u0_(m,IM3,k,j,i) += (m_old[3] - m_new[3]);
      }
    }
  );

  if (coupling_) {
    // conserved to primitive
    // TODO(@pdmullen) Need to be sure this is synced correctly with hydro
    // evolution.  As written, the RadiationTaskList will not communicate u0 ghost
    // zone values nor set boundary conditions.  This will be problematic when hydro
    // evolution is enabled.
    pmy_pack->phydro->peos->ConsToPrim(u0_, w0_);
  }

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

//----------------------------------------------------------------------------------------
//! \fn  bool FourthPolyRoot
//  \brief Exact solution for fourth order polynomial of
//  the form coef4 * x^4 + x + tconst = 0.

bool FourthPolyRoot(const Real coef4, const Real tconst, Real &root) {
  // Calculate real root of z^3 - 4*tconst/coef4 * z - 1/coef4^2 = 0
  Real asquar = coef4 * coef4;
  Real acubic = coef4 * asquar;
  Real ccubic = tconst * tconst * tconst;
  Real delta1 = 0.25 - 64.0 * ccubic * coef4 / 27.0;
  if (delta1 < 0.0) {
    return false;
  }
  delta1 = sqrt(delta1);
  if (delta1 < 0.5) {
    return false;
  }
  Real zroot;
  if (delta1 > 1.0e11) {  // to avoid small number cancellation
    zroot = pow(delta1, -2.0/3.0) / 3.0;
  } else {
    zroot = pow(0.5 + delta1, 1.0/3.0) - pow(-0.5 + delta1, 1.0/3.0);
  }
  if (zroot < 0.0) {
    return false;
  }
  zroot *= pow(coef4, -2.0/3.0);

  // Calculate quartic root using cubic root
  Real rcoef = sqrt(zroot);
  Real delta2 = -zroot + 2.0 / (coef4 * rcoef);
  if (delta2 < 0.0) {
    return false;
  }
  delta2 = sqrt(delta2);
  root = 0.5 * (delta2 - rcoef);
  if (root < 0.0) {
    return false;
  }
  return true;
}
