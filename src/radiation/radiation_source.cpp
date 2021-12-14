//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_update.cpp
//  \brief Performs update of Radiation conserved variables (i0) for each stage of
//   explicit SSP RK integrators (e.g. RK1, RK2, RK3). Update uses weighted average and
//   partial time step appropriate to stage.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"
#include "hydro/hydro.hpp"
#include "radiation.hpp"

#include "radiation/radiation_tetrad.hpp"

namespace radiation {

KOKKOS_INLINE_FUNCTION
bool FourthPolyRoot(const Real coef4, const Real tconst, Real &root);

//----------------------------------------------------------------------------------------
//! \fn
// Add implicit radiation source term
// NOTE Only executed on final stage of integration.

TaskStatus Radiation::AddRadiationSourceTerm(Driver *pdriver, int stage)
{
  if (not rad_source || stage != (pdriver->nexp_stages)) {
    return TaskStatus::complete; // only execute last stage
  }

  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nangles_ = nangles;

  Real dt_ = pmy_pack->pmesh->dt;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  // Load radiation quantities
  auto i0_ = i0;
  auto i1_ = i1;

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

  // extract geometric data
  auto nmu_ = pmy_pack->prad->nmu;
  auto nh_c_ = pmy_pack->prad->nh_c;
  auto n_mu_ = pmy_pack->prad->n_mu;
  auto solid_angle_ = pmy_pack->prad->solid_angle;
  auto norm_to_tet_ = pmy_pack->prad->norm_to_tet;
  auto coord = pmy_pack->coord.coord_data;

  // opacity function
  auto OpacityFunc_ = pmy_pack->prad->OpacityFunc;

  // extract coupling flag
  bool coupling_ = coupling;

  // compute implicit source term
  par_for("rad_source",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      // scratch arrays
      Real *omega_cm_ = new Real[nangles_];
      Real *intensity_cm_ = new Real[nangles_];
      Real *n0_cm_= new Real[nangles_];
      Real *vncsigma2_ = new Real[nangles_];
      Real *di_cm_ = new Real[nangles_];

      // coordinates
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
      for (int lm=0; lm<nangles_; ++lm) {
        Real un_tet = (u_tet_[1]*nh_c_.d_view(lm,1) +
                       u_tet_[2]*nh_c_.d_view(lm,2) +
                       u_tet_[3]*nh_c_.d_view(lm,3));
        n0_cm_[lm] = (u_tet_[0]*nh_c_.d_view(lm,0) - un_tet);
        omega_cm_[lm] = solid_angle_.d_view(lm)/SQR(n0_cm_[lm]);
        intensity_cm_[lm] = i1_(m,lm,k,j,i)*SQR(SQR(n0_cm_[lm]));
        wght_sum_ += omega_cm_[lm];
      }

      // normalizations
      for (int lm=0; lm<nangles_; ++lm) {
        omega_cm_[lm] /= wght_sum_;
        intensity_cm_[lm] *= 4.0*M_PI;
      }

      // set opacities
      Real kappa_a = 0.0, kappa_s = 0.0, kappa_p = 0.0;
      OpacityFunc_(rho, tgas, kappa_a, kappa_s, kappa_p);
      Real sigma_a = rho*kappa_a;
      Real sigma_s = rho*kappa_s;
      Real sigma_p = rho*kappa_p;
      Real dtaucsigmaa = dt_*sigma_a/(uu0*sqrt(-gi_[I00]));
      Real dtaucsigmas = dt_*sigma_s/(uu0*sqrt(-gi_[I00]));
      Real dtaucsigmap = dt_*sigma_p/(uu0*sqrt(-gi_[I00]));
      Real dtcsigmaa = dt_*sigma_a;
      Real dtcsigmas = dt_*sigma_s;
      Real dtcsigmap = dt_*sigma_p;

      // Calculate polynomial coefficients
      Real suma1 = 0.0;
      Real suma2 = 0.0;
      Real suma3 = 0.0;
      Real jr_cm = 0.0;
      for (int lm=0; lm<nangles_; ++lm) {
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
                  / (1.0-suma3))*arad_*(gamma_-1.0)/rho);
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
      if (not badcell) {
        // Calculate emission coefficient and updated jr_cm
        Real emission = arad_*SQR(SQR(tgasnew));
        jr_cm = (suma1*emission + suma2)/(1.0 - suma3);
        for (int lm=0; lm<nangles_; ++lm) {
          di_cm_[lm] = ( (  (dtcsigmas-dtcsigmap)*jr_cm
                         + (dtcsigmaa+dtcsigmap)*emission
                         - (dtcsigmas+dtcsigmaa)*intensity_cm_[lm])*vncsigma2_[lm]);
        }
      }

      // Apply radiation-fluid coupling to radiation in coordinate frame
      for (int lm=0; lm<nangles_; ++lm) {
        i0_(m,lm,k,j,i) = (i0_(m,lm,k,j,i)+(di_cm_[lm]/(4.0*M_PI*SQR(SQR(n0_cm_[lm])))));
      }

      // apply coupling to hydro
      if (coupling_) {
        // compute moments before and after coupling
        Real m_old[4] = {0.0}, m_new[4] = {0.0};
        for (int lm=0; lm<nangles_; ++lm) {
          Real sa = solid_angle_.d_view(lm);
          m_old[0] += (nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,0)*i1_(m,lm,k,j,i)*sa);
          m_old[1] += (nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,1)*i1_(m,lm,k,j,i)*sa);
          m_old[2] += (nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,2)*i1_(m,lm,k,j,i)*sa);
          m_old[3] += (nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,3)*i1_(m,lm,k,j,i)*sa);
          m_new[0] += (nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,0)*i0_(m,lm,k,j,i)*sa);
          m_new[1] += (nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,1)*i0_(m,lm,k,j,i)*sa);
          m_new[2] += (nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,2)*i0_(m,lm,k,j,i)*sa);
          m_new[3] += (nmu_(m,lm,k,j,i,0)*n_mu_(m,lm,k,j,i,3)*i0_(m,lm,k,j,i)*sa);
        }

        // update conserved variables
        u0_(m,IEN,k,j,i) += (m_old[0] - m_new[0]);
        // u0_(m,IM1,k,j,i) += (m_old[1] - m_new[1]);
        // u0_(m,IM2,k,j,i) += (m_old[2] - m_new[2]);
        // u0_(m,IM3,k,j,i) += (m_old[3] - m_new[3]);
      }

      // delete scratch arrays
      delete [] omega_cm_;
      delete [] intensity_cm_;
      delete [] n0_cm_;
      delete [] vncsigma2_;
      delete [] di_cm_;
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

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  bool FourthPolyRoot
//  \brief Exact solution for fourth order polynomial of
//  the form coef4 * x^4 + x + tconst = 0.

KOKKOS_INLINE_FUNCTION
bool FourthPolyRoot(const Real coef4, const Real tconst, Real &root)
{
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

} // namespace radiation
