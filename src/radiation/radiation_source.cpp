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
#include "mhd/mhd.hpp"
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

  // extract indices and size/coord data
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &aindcs = amesh_indcs;
  int &zs = aindcs.zs, &ze = aindcs.ze;
  int &ps = aindcs.ps, &pe = aindcs.pe;

  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &size = pmy_pack->pmb->mb_size;
  auto &coord = pmy_pack->pcoord->coord_data;

  // extract gas and radiation constants
  Real gamma_ = pmy_pack->phydro->peos->eos_data.gamma;
  Real gm1_ = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
  Real gammap_ = gamma_/gm1_;
  auto arad_ = arad;

  // extract frame data
  auto nmu_ = nmu;
  auto nh_c_ = nh_c;
  auto n_mu_ = n_mu;
  auto solid_angle_ = solid_angle;
  auto norm_to_tet_ = norm_to_tet;

  // extract coupling flags
  bool is_hydro_enabled_ = is_hydro_enabled;
  bool is_mhd_enabled_ = is_mhd_enabled;
  bool fixed_fluid_ = fixed_fluid;
  bool affect_fluid_ = affect_fluid;
  bool zero_radiation_force_ = zero_radiation_force;

  // Load radiation quantities
  auto i0_ = i0;

  // Load hydro/mhd quantities
  DvceArray5D<Real> u0_, w0_;
  if (is_hydro_enabled_) {
    u0_ = pmy_pack->phydro->u0;
    w0_ = pmy_pack->phydro->w0;
  } else if (is_mhd_enabled_) {
    u0_ = pmy_pack->pmhd->u0;
    w0_ = pmy_pack->pmhd->w0;
  }

  // Load magnetic field quantities
  DvceArray5D<Real> bcc0_;
  if (is_mhd_enabled_) {
    bcc0_ = pmy_pack->pmhd->bcc0;
  }

  // NOTE(@pdmullen): ConsToPrim over active zones
  if (!(fixed_fluid_)) {
    if (is_hydro_enabled_) {
      pmy_pack->phydro->peos->ConsToPrim(u0_, w0_, is, ie, js, je, ks, ke);
    } else if (is_mhd_enabled_) {
      auto &b0_ = pmy_pack->pmhd->b0;
      pmy_pack->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,is, ie, js, je, ks, ke);
    }
  }

  // opacities
  bool constant_opacity_ = constant_opacity;
  bool power_opacity_ = power_opacity;
  auto kappa_a_ = kappa_a;
  auto kappa_s_ = kappa_s;
  auto kappa_p_ = kappa_p;

  // extract excision flags
  auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &cc_rad_mask_ = pmy_pack->pcoord->cc_rad_mask;

  // timestep
  Real dt_ = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

  // compute implicit source term
  par_for_outer("radiation_source",DevExeSpace(),0,0,0,nmb1,ks,ke,js,je,is,ie,
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

    // compute metric and inverse
    Real g_[NMETRIC], gi_[NMETRIC];
    ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin, g_, gi_);

    // fluid state
    Real &wdn = w0_(m,IDN,k,j,i);
    Real &wvx = w0_(m,IVX,k,j,i);
    Real &wvy = w0_(m,IVY,k,j,i);
    Real &wvz = w0_(m,IVZ,k,j,i);
    Real &wen = w0_(m,IEN,k,j,i);

    // derived quantities
    Real tgas = (gm1_*wen)/wdn;
    Real gg = sqrt(1. + (g_[I11]*wvx*wvx + 2.*g_[I12]*wvx*wvy + 2.*g_[I13]*wvx*wvz
                                         +    g_[I22]*wvy*wvy + 2.*g_[I23]*wvy*wvz
                                                              +    g_[I33]*wvz*wvz));

    // set opacities
    Real sigma_a = 0.0, sigma_s = 0.0, sigma_p = 0.0;
    OpacityFunction(wdn, tgas, kappa_a_, kappa_s_, kappa_p_,
                    constant_opacity_, power_opacity_,
                    sigma_a, sigma_s, sigma_p);

    // compute fluid velocity in tetrad frame
    Real u_tet[4];
    u_tet[0] = (norm_to_tet_(m,0,0,k,j,i)*gg  + norm_to_tet_(m,0,1,k,j,i)*wvx +
                norm_to_tet_(m,0,2,k,j,i)*wvy + norm_to_tet_(m,0,3,k,j,i)*wvz);
    u_tet[1] = (norm_to_tet_(m,1,0,k,j,i)*gg  + norm_to_tet_(m,1,1,k,j,i)*wvx +
                norm_to_tet_(m,1,2,k,j,i)*wvy + norm_to_tet_(m,1,3,k,j,i)*wvz);
    u_tet[2] = (norm_to_tet_(m,2,0,k,j,i)*gg  + norm_to_tet_(m,2,1,k,j,i)*wvx +
                norm_to_tet_(m,2,2,k,j,i)*wvy + norm_to_tet_(m,2,3,k,j,i)*wvz);
    u_tet[3] = (norm_to_tet_(m,3,0,k,j,i)*gg  + norm_to_tet_(m,3,1,k,j,i)*wvx +
                norm_to_tet_(m,3,2,k,j,i)*wvy + norm_to_tet_(m,3,3,k,j,i)*wvz);

    // compute intensities and solid angles in comoving frame
    Real wght_sum = 0.0;
    for (int z=zs; z<=ze; ++z) {
      for (int p=ps; p<=pe; ++p) {
        Real un_tet = (u_tet[1]*nh_c_.d_view(z,p,1) +
                       u_tet[2]*nh_c_.d_view(z,p,2) +
                       u_tet[3]*nh_c_.d_view(z,p,3));
        Real n0_cm  = (u_tet[0]*nh_c_.d_view(z,p,0) - un_tet);
        wght_sum += solid_angle_.d_view(z,p)/SQR(n0_cm);
      }
    }

    Real dtcsigmaa = dt_*sigma_a;
    Real dtcsigmas = dt_*sigma_s;
    Real dtcsigmap = dt_*sigma_p;
    Real dtaucsigmaa = dtcsigmaa/(gg*sqrt(-gi_[I00]));
    Real dtaucsigmas = dtcsigmas/(gg*sqrt(-gi_[I00]));
    Real dtaucsigmap = dtcsigmap/(gg*sqrt(-gi_[I00]));

    // Calculate polynomial coefficients
    Real suma1 = 0.0;
    Real suma2 = 0.0;
    Real jr_cm = 0.0;
    for (int z=zs; z<=ze; ++z) {
      for (int p=ps; p<=pe; ++p) {
        int n = AngleInd(z,p,false,false,aindcs);
        Real n0_local = nmu_(m,z,p,k,j,i,0);
        Real un_tet   = (u_tet[1]*nh_c_.d_view(z,p,1) +
                         u_tet[2]*nh_c_.d_view(z,p,2) +
                         u_tet[3]*nh_c_.d_view(z,p,3));
        Real n0_cm    = (u_tet[0]*nh_c_.d_view(z,p,0) - un_tet);
        Real omega_cm = solid_angle_.d_view(z,p)/SQR(n0_cm)/wght_sum;
        Real intensity_cm = 4.0*M_PI*i0_(m,n,k,j,i)*SQR(SQR(n0_cm));
        Real vncsigma = 1.0/(n0_local + (dtcsigmaa + dtcsigmas)*n0_cm);
        Real vncsigma2 = n0_cm*vncsigma;
        Real ir_weight = intensity_cm*omega_cm;
        jr_cm += ir_weight;
        suma1 += omega_cm*vncsigma2;
        suma2 += ir_weight*n0_local*vncsigma;
      }
    }
    Real suma3 = suma1*(dtcsigmas - dtcsigmap);
    suma1 *= (dtcsigmaa + dtcsigmap);

    // compute coefficients
    Real coef[2] = {0.0};
    coef[1] = ((dtaucsigmaa+dtaucsigmap-(dtaucsigmaa+dtaucsigmap)*suma1
                / (1.0-suma3))*arad_*(gamma_-1.0)/wdn);
    coef[0] = (-tgas-(dtaucsigmaa+dtaucsigmap)*suma2*(gamma_-1.0)/(wdn*(1.0-suma3)));

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
    if (affect_fluid_ && !(zero_radiation_force_)) {
      for (int z=zs; z<=ze; ++z) {
        for (int p=ps; p<=pe; ++p) {
          int n = AngleInd(z,p,false,false,aindcs);
          Real sa = solid_angle_.d_view(z,p);
          m_old[0] += (nmu_(m,z,p,k,j,i,0)*n_mu_(m,z,p,k,j,i,0)*i0_(m,n,k,j,i)*sa);
          m_old[1] += (nmu_(m,z,p,k,j,i,0)*n_mu_(m,z,p,k,j,i,1)*i0_(m,n,k,j,i)*sa);
          m_old[2] += (nmu_(m,z,p,k,j,i,0)*n_mu_(m,z,p,k,j,i,2)*i0_(m,n,k,j,i)*sa);
          m_old[3] += (nmu_(m,z,p,k,j,i,0)*n_mu_(m,z,p,k,j,i,3)*i0_(m,n,k,j,i)*sa);
        }
      }
    }

    // Update the specific intensity
    if (!(badcell)) {
      // Calculate emission coefficient and updated jr_cm
      Real emission = arad_*SQR(SQR(tgasnew));
      jr_cm = (suma1*emission + suma2)/(1.0 - suma3);
      for (int z=zs; z<=ze; ++z) {
        for (int p=ps; p<=pe; ++p) {
          int n = AngleInd(z,p,false,false,aindcs);
          Real n0_local = nmu_(m,z,p,k,j,i,0);
          Real un_tet   = (u_tet[1]*nh_c_.d_view(z,p,1) +
                           u_tet[2]*nh_c_.d_view(z,p,2) +
                           u_tet[3]*nh_c_.d_view(z,p,3));
          Real n0_cm    = (u_tet[0]*nh_c_.d_view(z,p,0) - un_tet);
          Real intensity_cm = 4.0*M_PI*i0_(m,n,k,j,i)*SQR(SQR(n0_cm));
          Real vncsigma = 1.0/(n0_local + (dtcsigmaa + dtcsigmas)*n0_cm);
          Real vncsigma2 = n0_cm*vncsigma;
          Real di_cm = ( ((dtcsigmas-dtcsigmap)*jr_cm
                        + (dtcsigmaa+dtcsigmap)*emission
                        - (dtcsigmas+dtcsigmaa)*intensity_cm)*vncsigma2);
          i0_(m,n,k,j,i) = fmax((i0_(m,n,k,j,i)+(di_cm/(4.0*M_PI*SQR(SQR(n0_cm))))), 0.0);
          if (excise) {
            if (cc_rad_mask_(m,k,j,i)) {
              i0_(m,n,k,j,i) = 0.0;
            }
          }
        }
      }
    }

    // apply coupling to hydro
    if (affect_fluid_) {
      // update conserved variables
      if (zero_radiation_force_) {
        Real alpha = sqrt(-1.0/gi_[I00]);
        Real u0_tmp = gg/alpha;
        Real u1_tmp = wvx - alpha * gg * gi_[I01];
        Real u2_tmp = wvy - alpha * gg * gi_[I02];
        Real u3_tmp = wvz - alpha * gg * gi_[I03];
        Real u_0_tmp = g_[I00]*u0_tmp + g_[I01]*u1_tmp + g_[I02]*u2_tmp + g_[I03]*u3_tmp;
        Real u_1_tmp = g_[I01]*u0_tmp + g_[I11]*u1_tmp + g_[I12]*u2_tmp + g_[I13]*u3_tmp;
        Real u_2_tmp = g_[I02]*u0_tmp + g_[I12]*u1_tmp + g_[I22]*u2_tmp + g_[I23]*u3_tmp;
        Real u_3_tmp = g_[I03]*u0_tmp + g_[I13]*u1_tmp + g_[I23]*u2_tmp + g_[I33]*u3_tmp;
        Real pgas_new = wdn*tgasnew;
        if (is_hydro_enabled_) {
          Real wgas_u0 = (wdn + gammap_ * pgas_new) * u0_tmp;
          u0_(m,IEN,k,j,i) = wgas_u0 * u_0_tmp + pgas_new + wdn * u0_tmp;
          u0_(m,IM1,k,j,i) = wgas_u0 * u_1_tmp;
          u0_(m,IM2,k,j,i) = wgas_u0 * u_2_tmp;
          u0_(m,IM3,k,j,i) = wgas_u0 * u_3_tmp;
        } else if (is_mhd_enabled_) {
          Real &bcc1 = bcc0_(m,IBX,k,j,i);
          Real &bcc2 = bcc0_(m,IBY,k,j,i);
          Real &bcc3 = bcc0_(m,IBZ,k,j,i);
          Real b0 = g_[I01]*u0_tmp*bcc1 + g_[I02]*u0_tmp*bcc2 + g_[I03]*u0_tmp*bcc3
                  + g_[I11]*u1_tmp*bcc1 + g_[I12]*u1_tmp*bcc2 + g_[I13]*u1_tmp*bcc3
                  + g_[I12]*u2_tmp*bcc1 + g_[I22]*u2_tmp*bcc2 + g_[I23]*u2_tmp*bcc3
                  + g_[I13]*u3_tmp*bcc1 + g_[I23]*u3_tmp*bcc2 + g_[I33]*u3_tmp*bcc3;
          Real b1 = (bcc1 + b0 * u1_tmp) / u0_tmp;
          Real b2 = (bcc2 + b0 * u2_tmp) / u0_tmp;
          Real b3 = (bcc3 + b0 * u3_tmp) / u0_tmp;
          Real b_0 = g_[I00]*b0 + g_[I01]*b1 + g_[I02]*b2 + g_[I03]*b3;
          Real b_1 = g_[I01]*b0 + g_[I11]*b1 + g_[I12]*b2 + g_[I13]*b3;
          Real b_2 = g_[I02]*b0 + g_[I12]*b1 + g_[I22]*b2 + g_[I23]*b3;
          Real b_3 = g_[I03]*b0 + g_[I13]*b1 + g_[I23]*b2 + g_[I33]*b3;
          Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;
          Real wtot = wdn + gammap_ * pgas_new + b_sq;
          Real ptot = pgas_new + 0.5 * b_sq;
          u0_(m,IEN,k,j,i) = wtot * u0_tmp * u_0_tmp - b0 * b_0 + ptot + wdn * u0_tmp;
          u0_(m,IM1,k,j,i) = wtot * u0_tmp * u_1_tmp - b0 * b_1;
          u0_(m,IM2,k,j,i) = wtot * u0_tmp * u_2_tmp - b0 * b_2;
          u0_(m,IM3,k,j,i) = wtot * u0_tmp * u_3_tmp - b0 * b_3;
        }
      } else {
        Real m_new[4] = {0.0};
        for (int z=zs; z<=ze; ++z) {
          for (int p=ps; p<=pe; ++p) {
            int n = AngleInd(z,p,false,false,aindcs);
            Real sa = solid_angle_.d_view(z,p);
            m_new[0] += (nmu_(m,z,p,k,j,i,0)*n_mu_(m,z,p,k,j,i,0)*i0_(m,n,k,j,i)*sa);
            m_new[1] += (nmu_(m,z,p,k,j,i,0)*n_mu_(m,z,p,k,j,i,1)*i0_(m,n,k,j,i)*sa);
            m_new[2] += (nmu_(m,z,p,k,j,i,0)*n_mu_(m,z,p,k,j,i,2)*i0_(m,n,k,j,i)*sa);
            m_new[3] += (nmu_(m,z,p,k,j,i,0)*n_mu_(m,z,p,k,j,i,3)*i0_(m,n,k,j,i)*sa);
          }
        }
        u0_(m,IEN,k,j,i) += (m_old[0] - m_new[0]);
        u0_(m,IM1,k,j,i) += (m_old[1] - m_new[1]);
        u0_(m,IM2,k,j,i) += (m_old[2] - m_new[2]);
        u0_(m,IM3,k,j,i) += (m_old[3] - m_new[3]);
      }
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
