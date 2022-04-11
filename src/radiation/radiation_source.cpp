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
#include "radiation/radiation_opacities.hpp"

namespace radiation {

KOKKOS_INLINE_FUNCTION
bool FourthPolyRoot(const Real coef4, const Real tconst, Real &root);

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::AddRadiationSourceTerm(Driver *pdriver, int stage)
// Add implicit radiation source term

TaskStatus Radiation::AddRadiationSourceTerm(Driver *pdriver, int stage) {
  if (!(rad_source)) {
    return TaskStatus::complete;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nangles_ = nangles;

  Real dt_ = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang1 = nangles - 1;
  auto &size = pmy_pack->pmb->mb_size;

  auto &coord = pmy_pack->pcoord->coord_data;

  // Load radiation quantities
  auto i0_ = i0;

  // Load hydro quantities
  auto u0_ = pmy_pack->phydro->u0;
  auto w0_ = pmy_pack->phydro->w0;
  // TODO(@pdmullen): is this okay? this is before physical boundary conditions are
  // applied and before MeshBlock boundaries have been set.  But I think this might be
  // safe because this source term doesn't need boundary values.
  pmy_pack->phydro->peos->ConsToPrim(u0_, w0_, is, ie, js, je, ks, ke);

  // extract gas and radiation constants
  Real gamma_ = pmy_pack->phydro->peos->eos_data.gamma;
  Real gm1_ = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
  auto arad_ = arad;
  auto use_e = pmy_pack->phydro->peos->eos_data.use_e;

  // extract frame data
  auto nmu_ = nmu;
  auto nh_c_ = nh_c;
  auto n_mu_ = n_mu;
  auto solid_angle_ = solid_angle;
  auto norm_to_tet_ = norm_to_tet;

  // opacities
  bool constant_opacity_ = constant_opacity;
  bool power_opacity_ = power_opacity;
  auto kappa_a_ = kappa_a;
  auto kappa_s_ = kappa_s;
  auto kappa_p_ = kappa_p;

  // extract coupling flag
  bool affect_fluid_ = affect_fluid;

  // extract excision flags
  auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &cc_mask_ = pmy_pack->pcoord->cc_mask;

  // compute implicit source term
  par_for_outer("beam_source",DevExeSpace(),0,0,0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j, const int i) {
    // coordinates
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    // Extract components of metric
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));

    // compute metric and inverse
    Real g_[NMETRIC], gi_[NMETRIC];
    ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin, g_, gi_);

    // fluid state
    Real rho  = w0_(m,IDN,k,j,i);
    Real uu1  = w0_(m,IVX,k,j,i);
    Real uu2  = w0_(m,IVY,k,j,i);
    Real uu3  = w0_(m,IVZ,k,j,i);

    // derived quantities
    Real tgas = gm1_*w0_(m,IEN,k,j,i)/rho;
    Real uu0 = sqrt(1. + (g_[I11]*uu1*uu1 + 2.*g_[I12]*uu1*uu2 + 2.*g_[I13]*uu1*uu3
                                          +    g_[I22]*uu2*uu2 + 2.*g_[I23]*uu2*uu3
                                                               +    g_[I33]*uu3*uu3));

    // compute fluid velocity in tetrad frame
    Real u_tet[4];
    u_tet[0] = (norm_to_tet_(m,0,0,k,j,i)*uu0 + norm_to_tet_(m,0,1,k,j,i)*uu1 +
                norm_to_tet_(m,0,2,k,j,i)*uu2 + norm_to_tet_(m,0,3,k,j,i)*uu3);
    u_tet[1] = (norm_to_tet_(m,1,0,k,j,i)*uu0 + norm_to_tet_(m,1,1,k,j,i)*uu1 +
                norm_to_tet_(m,1,2,k,j,i)*uu2 + norm_to_tet_(m,1,3,k,j,i)*uu3);
    u_tet[2] = (norm_to_tet_(m,2,0,k,j,i)*uu0 + norm_to_tet_(m,2,1,k,j,i)*uu1 +
                norm_to_tet_(m,2,2,k,j,i)*uu2 + norm_to_tet_(m,2,3,k,j,i)*uu3);
    u_tet[3] = (norm_to_tet_(m,3,0,k,j,i)*uu0 + norm_to_tet_(m,3,1,k,j,i)*uu1 +
                norm_to_tet_(m,3,2,k,j,i)*uu2 + norm_to_tet_(m,3,3,k,j,i)*uu3);

    // compute intensities and solid angles in comoving frame
    Real wght_sum = 0.0;
    for (int n=0; n<nangles_; ++n) {
      Real un_tet = (u_tet[1]*nh_c_.d_view(n,1) +
                     u_tet[2]*nh_c_.d_view(n,2) +
                     u_tet[3]*nh_c_.d_view(n,3));
      Real n0_cm  = (u_tet[0]*nh_c_.d_view(n,0) - un_tet);
      wght_sum   += solid_angle_.d_view(n)/SQR(n0_cm);
    }

    // set opacities
    Real sigma_a = 0.0, sigma_s = 0.0, sigma_p = 0.0;
    OpacityFunction(rho, tgas, kappa_a_, kappa_s_, kappa_p_,
                    constant_opacity_, power_opacity_,
                    sigma_a, sigma_s, sigma_p);
    Real dtcsigmaa = dt_*sigma_a;
    Real dtcsigmas = dt_*sigma_s;
    Real dtcsigmap = dt_*sigma_p;
    Real dtaucsigmaa = dtcsigmaa/(uu0*sqrt(-gi_[I00]));
    Real dtaucsigmas = dtcsigmas/(uu0*sqrt(-gi_[I00]));
    Real dtaucsigmap = dtcsigmap/(uu0*sqrt(-gi_[I00]));

    // Calculate polynomial coefficients
    Real suma1 = 0.0;
    Real suma2 = 0.0;
    Real jr_cm = 0.0;
    for (int n=0; n<nangles_; ++n) {
      Real n0_local = nmu_(m,n,k,j,i,0);
      Real un_tet   = (u_tet[1]*nh_c_.d_view(n,1) +
                       u_tet[2]*nh_c_.d_view(n,2) +
                       u_tet[3]*nh_c_.d_view(n,3));
      Real n0_cm    = (u_tet[0]*nh_c_.d_view(n,0) - un_tet);
      Real omega_cm = solid_angle_.d_view(n)/SQR(n0_cm)/wght_sum;
      Real intensity_cm = 4.0*M_PI*i0_(m,n,k,j,i)*SQR(SQR(n0_cm));
      Real vncsigma = 1.0/(n0_local + (dtcsigmaa + dtcsigmas)*n0_cm);
      Real vncsigma2 = n0_cm*vncsigma;
      Real ir_weight = intensity_cm*omega_cm;
      jr_cm += ir_weight;
      suma1 += omega_cm*vncsigma2;
      suma2 += ir_weight*n0_local*vncsigma;
    }
    Real suma3 = suma1*(dtcsigmas - dtcsigmap);
    suma1 *= (dtcsigmaa + dtcsigmap);

    // compute coefficients
    Real coef[2] = {0.0};
    coef[1] = ((dtaucsigmaa+dtaucsigmap-(dtaucsigmaa+dtaucsigmap)*suma1
                / (1.0-suma3))*arad_*(gamma_-1.0)/rho);
    coef[0] = (-tgas-(dtaucsigmaa+dtaucsigmap)*suma2*(gamma_-1.0)/(rho*(1.0-suma3)));

    // Calculate new gas temperature
    Real tgasnew = tgas;
    bool badcell = false;
    if (fabs(coef[1]) > 1.0e-20) {
      bool flag = FourthPolyRoot(coef[1], coef[0], tgasnew);
      if (!(flag) || isnan(tgasnew)) {
        badcell = true;
        tgasnew = tgas;
      }
    } else {
      tgasnew = -coef[0];
    }

    // compute moments before coupling
    Real m_old[4] = {0.0};
    if (affect_fluid_) {
      for (int n=0; n<nangles_; ++n) {
        Real sa = solid_angle_.d_view(n);
        m_old[0] += (nmu_(m,n,k,j,i,0)*n_mu_(m,n,k,j,i,0)*i0_(m,n,k,j,i)*sa);
        m_old[1] += (nmu_(m,n,k,j,i,0)*n_mu_(m,n,k,j,i,1)*i0_(m,n,k,j,i)*sa);
        m_old[2] += (nmu_(m,n,k,j,i,0)*n_mu_(m,n,k,j,i,2)*i0_(m,n,k,j,i)*sa);
        m_old[3] += (nmu_(m,n,k,j,i,0)*n_mu_(m,n,k,j,i,3)*i0_(m,n,k,j,i)*sa);
      }
    }

    // Update the specific intensity
    if (!(badcell)) {
      // Calculate emission coefficient and updated jr_cm
      Real emission = arad_*SQR(SQR(tgasnew));
      jr_cm = (suma1*emission + suma2)/(1.0 - suma3);
      par_for_inner(member, 0, nang1, [&](const int n) {
        Real n0_local = nmu_(m,n,k,j,i,0);
        Real un_tet   = (u_tet[1]*nh_c_.d_view(n,1) +
                         u_tet[2]*nh_c_.d_view(n,2) +
                         u_tet[3]*nh_c_.d_view(n,3));
        Real n0_cm    = (u_tet[0]*nh_c_.d_view(n,0) - un_tet);
        Real omega_cm = solid_angle_.d_view(n)/SQR(n0_cm)/wght_sum;
        Real intensity_cm = 4.0*M_PI*i0_(m,n,k,j,i)*SQR(SQR(n0_cm));
        Real vncsigma = 1.0/(n0_local + (dtcsigmaa + dtcsigmas)*n0_cm);
        Real vncsigma2 = n0_cm*vncsigma;
        Real ir_weight = intensity_cm*omega_cm;
        Real di_cm = ( ((dtcsigmas-dtcsigmap)*jr_cm
                      + (dtcsigmaa+dtcsigmap)*emission
                      - (dtcsigmas+dtcsigmaa)*intensity_cm)*vncsigma2);
        i0_(m,n,k,j,i) = fmax((i0_(m,n,k,j,i)+(di_cm/(4.0*M_PI*SQR(SQR(n0_cm))))), 0.0);
        // if excising, handle r_ks < 0.5*(r_inner + r_outer)
        if (excise) {
          if (cc_mask_(m,k,j,i)) {
            i0_(m,n,k,j,i) = 0.0;
          }
        }
      });
    }

    // apply coupling to hydro
    Real m_new[4] = {0.0};
    if (affect_fluid_) {
      for (int n=0; n<nangles_; ++n) {
        Real sa = solid_angle_.d_view(n);
        m_new[0] += (nmu_(m,n,k,j,i,0)*n_mu_(m,n,k,j,i,0)*i0_(m,n,k,j,i)*sa);
        m_new[1] += (nmu_(m,n,k,j,i,0)*n_mu_(m,n,k,j,i,1)*i0_(m,n,k,j,i)*sa);
        m_new[2] += (nmu_(m,n,k,j,i,0)*n_mu_(m,n,k,j,i,2)*i0_(m,n,k,j,i)*sa);
        m_new[3] += (nmu_(m,n,k,j,i,0)*n_mu_(m,n,k,j,i,3)*i0_(m,n,k,j,i)*sa);
      }

      // update conserved variables
      u0_(m,IEN,k,j,i) += (m_old[0] - m_new[0]);
      u0_(m,IM1,k,j,i) += (m_old[1] - m_new[1]);
      u0_(m,IM2,k,j,i) += (m_old[2] - m_new[2]);
      u0_(m,IM3,k,j,i) += (m_old[3] - m_new[3]);
    }
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  bool FourthPolyRoot
//  \brief Exact solution for fourth order polynomial of
//  the form coef4 * x^4 + x + tconst = 0.

KOKKOS_INLINE_FUNCTION
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

} // namespace radiation
