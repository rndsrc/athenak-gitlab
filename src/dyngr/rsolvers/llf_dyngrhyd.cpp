//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_grmhd.cpp
//! \brief LLF Riemann solver for general relativistic hydrodynamics.

#include <math.h>

#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "adm/adm.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

namespace dyngr {

//----------------------------------------------------------------------------------------
//! \fn void FLUX_PT_DYNGR
//! \brief Inline function for calculating the GRHD flux for a single given state.
KOKKOS_INLINE_FUNCTION
void FLUX_PT_DYNGR(const Real prim_pt[NPRIM], Real cons_pt[NCONS],
                   Real flux_pt[NCONS], const Real g3d[NSPMETRIC], const Real beta_u,
                   const Real alpha, const Real sdetg, int pvx, int csx) {
  Real W = sqrt(1.0 + Primitive::SquareVector(&prim_pt[PVX], g3d));
  Real vi = prim_pt[pvx]/W;

  Real v_c = vi - beta_u/alpha;

  flux_pt[CDN] = alpha*cons_pt[CDN]*v_c;
  flux_pt[CSX] = alpha*cons_pt[CSX]*v_c;
  flux_pt[CSY] = alpha*cons_pt[CSY]*v_c;
  flux_pt[CSZ] = alpha*cons_pt[CSZ]*v_c;
  flux_pt[CTA] = alpha*(cons_pt[CTA]*v_c + sdetg*prim_pt[PPR]*vi);
  flux_pt[csx] += alpha*sdetg*prim_pt[PPR];
}

//----------------------------------------------------------------------------------------
//! \fn void SingleStateLLF_DYNGR
//! \brief
template<class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void SingleStateLLF_DYNGR(const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
                          Real wl[NPRIM], Real wr[NPRIM], const int ivx,
                          Real g3d[NSPMETRIC], Real beta_u[3],
                          Real& alpha, Real flx[NCONS]) {
  // Cyclic permutation of array indices
  int pvx, idx, csx;
  if (ivx == IVX) {
    pvx = PVX;
    csx = CSX;
    idx = S11;
  }
  else if (ivx == IVY) {
    pvx = PVY;
    csx = CSY;
    idx = S22;
  }
  else if (ivx == IVZ) {
    pvx = PVZ;
    csx = CSZ;
    idx = S33;
  }

  const Real mb = eos.ps.GetEOS().GetBaryonMass();
  Real b[3] = {0.0};

  // Calculate the inverse metric
  Real detg = Primitive::GetDeterminant(g3d);
  Real sdetg = sqrt(detg);
  Real g3u[NSPMETRIC];
  Primitive::InvertMatrix(g3u, g3d, detg);

  // TODO: Based on where this is being called, we shouldn't need to floor. But this should
  // be validated.
  Real cons_l[NCONS], cons_r[NCONS], fl[NCONS], fr[NCONS];
  eos.ps.PrimToCon(wl, cons_l, b, g3d);
  eos.ps.PrimToCon(wr, cons_r, b, g3d);

  // Because we directly called the PrimToCon routine in PrimitiveSolver, we need to densitize
  // the conserved variables.
  for (int i = 0; i < NCONS; i++) {
    cons_l[i] *= sdetg;
    cons_r[i] *= sdetg;
  }

  // Calculate the fluxes
  FLUX_PT_DYNGR(wl, cons_l, fl, g3d, beta_u[pvx - PVX], alpha, sdetg, pvx, csx);
  FLUX_PT_DYNGR(wr, cons_r, fr, g3d, beta_u[pvx - PVX], alpha, sdetg, pvx, csx);

  // Get the sound speeds
  Real lambda_lp, lambda_lm, lambda_rp, lambda_rm;
  eos.GetGRSoundSpeeds(lambda_lp, lambda_lm, wl, g3d, beta_u, alpha, g3u[idx], pvx);
  eos.GetGRSoundSpeeds(lambda_rp, lambda_rm, wr, g3d, beta_u, alpha, g3u[idx], pvx);

  // Get the extremal wavespeeds
  Real lambda_l = fmin(lambda_lm, lambda_rm);
  Real lambda_r = fmax(lambda_lp, lambda_rp);
  Real lambda = fmax(lambda_r, -lambda_l);
  
  flx[CDN] = 0.5 * (fl[CDN] + fr[CDN] - lambda * (cons_r[CDN] - cons_l[CDN]));
  flx[CSX] = 0.5 * (fl[CSX] + fr[CSX] - lambda * (cons_r[CSX] - cons_l[CSX]));
  flx[CSY] = 0.5 * (fl[CSY] + fr[CSY] - lambda * (cons_r[CSY] - cons_l[CSY]));
  flx[CSZ] = 0.5 * (fl[CSZ] + fr[CSZ] - lambda * (cons_r[CSZ] - cons_l[CSZ]));
  flx[CTA] = 0.5 * (fl[CTA] + fr[CTA] - lambda * (cons_r[CTA] - cons_l[CTA]));
}

//----------------------------------------------------------------------------------------
//! \fn void LLF_DYNGR
//! \brief

template<class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void LLF_DYNGR(TeamMember_t const &member, 
     const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
     const RegionIndcs &indcs, const DualArray1D<RegionSize> &size, const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const int& nhyd, const int& nscal,
     const AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> &gamma_dd,
     const AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> &b_u,
     const AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> &alp, DvceArray5D<Real> flx) {

  const Real mb = eos.ps.GetEOS().GetBaryonMass();

  //auto &nhyd = eos.pmy_pack->phydro->nhydro;
  //auto &nscal = eos.pmy_pack->phydro->nscalars;
  int pvx, idx, csx;
  if (ivx == IVX) {
    pvx = PVX;
    csx = CSX;
    idx = S11;
  }
  else if (ivx == IVY) {
    pvx = PVY;
    csx = CSY;
    idx = S22;
  }
  else if (ivx == IVZ) {
    pvx = PVZ;
    csx = CSZ;
    idx = S33;
  }

  par_for_inner(member, il, iu, [&](const int i) {
    // Extract metric components
    Real g3d[NSPMETRIC];
    g3d[S11] = gamma_dd(0, 0, i);
    g3d[S12] = gamma_dd(0, 1, i);
    g3d[S13] = gamma_dd(0, 2, i);
    g3d[S22] = gamma_dd(1, 1, i);
    g3d[S23] = gamma_dd(1, 2, i);
    g3d[S33] = gamma_dd(2, 2, i);
    Real detg = Primitive::GetDeterminant(g3d);
    Real sdetg = sqrt(detg);
    Real g3u[NSPMETRIC];
    Primitive::InvertMatrix(g3u, g3d, detg);

    // Shift vector
    Real beta_u[3] = {0.0};
    beta_u[0] = b_u(0, i);
    beta_u[1] = b_u(1, i);
    beta_u[2] = b_u(2, i);

    // Lapse
    Real alpha = alp(i);

    // Extract left primitives and calculate left conserved variables
    Real prim_l[NPRIM], cons_l[NCONS];
    eos.PrimToConsPt(wl, prim_l, cons_l,
                      g3d, sdetg, i, nhyd, nscal);

    // Extract right primitives and calculate right conserved variables
    Real prim_r[NPRIM], cons_r[NCONS];
    eos.PrimToConsPt(wr, prim_r, cons_r,
                      g3d, sdetg, i, nhyd, nscal);

    // Calculate the fluxes.
    Real fl[NCONS], fr[NCONS];
    FLUX_PT_DYNGR(prim_l, cons_l, fl, g3d, beta_u[pvx - PVX], alpha, sdetg, pvx, csx);
    FLUX_PT_DYNGR(prim_r, cons_r, fr, g3d, beta_u[pvx - PVX], alpha, sdetg, pvx, csx);

    // Get the sound speeds
    Real lambda_lp, lambda_lm, lambda_rp, lambda_rm;
    eos.GetGRSoundSpeeds(lambda_lp, lambda_lm, prim_l, g3d, beta_u, alpha, g3u[idx], pvx);
    eos.GetGRSoundSpeeds(lambda_rp, lambda_rm, prim_r, g3d, beta_u, alpha, g3u[idx], pvx);

    // Get the extremal wavespeeds
    Real lambda_l = fmin(lambda_lm, lambda_rm);
    Real lambda_r = fmax(lambda_lp, lambda_rp);
    Real lambda = fmax(lambda_r, -lambda_l);
    //Real lambda = 1.0;

    // Store the complete fluxes.
    // Note that we don't need to worry about scalars -- Athena will do that automatically.
    flx(m, IDN, k, j, i) = 0.5 * (fl[CDN] + fr[CDN] - lambda * (cons_r[CDN] - cons_l[CDN]));
    flx(m, IEN, k, j, i) = 0.5 * (fl[CTA] + fr[CTA] - lambda * (cons_r[CTA] - cons_l[CTA]));
    flx(m, IVX, k, j, i) = 0.5 * (fl[CSX] + fr[CSX] - lambda * (cons_r[CSX] - cons_l[CSX]));
    flx(m, IVY, k, j, i) = 0.5 * (fl[CSY] + fr[CSY] - lambda * (cons_r[CSY] - cons_l[CSY]));
    flx(m, IVZ, k, j, i) = 0.5 * (fl[CSZ] + fr[CSZ] - lambda * (cons_r[CSZ] - cons_l[CSZ]));
  });
}

}