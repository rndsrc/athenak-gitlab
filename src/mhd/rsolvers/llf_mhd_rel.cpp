//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_mhd.cpp
//  \brief Local Lax Friedrichs (LLF) Riemann solver for MHD
//
//  Computes 1D fluxes using the LLF Riemann solver, also known as Rusanov's method.
//  This flux is very diffusive, even more diffusive than HLLE, and so it is not
//  recommended for use in applications.  However, it is useful for testing, or for
//  problems where other Riemann solvers fail.
//
// REFERENCES:
// - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//   Springer-Verlag, Berlin, (1999) chpt. 10.

// C/C++ headers
#include <algorithm>  // max(), min()

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn void LLF
//  \brief The LLF Riemann solver for MHD (both adiabatic and isothermal)

KOKKOS_INLINE_FUNCTION
void LLF_rel(TeamMember_t const &member, const EOS_Data &eos,
     const int m, const int k, const int j,  const int il, const int iu,
     const int ivx, const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     const DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez)
{
  int ivy = IVX + ((ivx-IVX) + 1)%3;
  int ivz = IVX + ((ivx-IVX) + 2)%3;
  int iby = ((ivx-IVX) + 1)%3;
  int ibz = ((ivx-IVX) + 2)%3;
  Real du[7],fl[7],fr[7];
  Real gm1 = eos.gamma - 1.0;
  Real iso_cs = eos.iso_cs;

  par_for_inner(member, il, iu, [&](const int i)
  {
    //--- Step 1.  Create local references for L/R states
    Real &wl_idn=wl(IDN,i);
    Real &wl_ivx=wl(ivx,i);
    Real &wl_ivy=wl(ivy,i);
    Real &wl_ivz=wl(ivz,i);
    Real &wl_iby=bl(iby,i);
    Real &wl_ibz=bl(ibz,i);

    Real &wr_idn=wr(IDN,i);
    Real &wr_ivx=wr(ivx,i);
    Real &wr_ivy=wr(ivy,i);
    Real &wr_ivz=wr(ivz,i);
    Real &wr_iby=br(iby,i);
    Real &wr_ibz=br(ibz,i);

    Real &bxi = bx(m,k,j,i);

    Real &wl_ipr=wl(IPR,i);
    Real &wr_ipr=wr(IPR,i);

    //--- Step 2.  Compute wave speeds in L,R states (see Toro eq. 10.43)


    Real b0l = bxi * wl_ivx + wl_iby * wl_ivy + wl_ibz * wl_ivz;
    Real b0r = bxi * wr_ivx + wr_iby * wr_ivy + wr_ibz * wr_ivz;

    Real u2l = SQR(wl_ivz) + SQR(wl_ivy) + SQR(wl_ivx);
    Real u2r = SQR(wr_ivz) + SQR(wr_ivy) + SQR(wr_ivx);

    Real u0l  = sqrt(1. + u2l);
    Real u0r  = sqrt(1. + u2r);

    Real b2l = ((bxi*bxi + wl_iby*wl_iby + wl_ibz*wl_ibz) + b0l*b0l) / (1.+u2l);
    Real b2r = ((bxi*bxi + wr_iby*wr_iby + wr_ibz*wr_ibz) + b0r*b0r) / (1.+u2r);

    // FIXME ERM: Ideal fluid for now
    Real wgas_l = wl_idn + (eos.gamma/gm1) * wl_ipr + b2l;
    Real wgas_r = wr_idn + (eos.gamma/gm1) * wr_ipr + b2r;

    Real pl = wl_ipr + 0.5*b2l;
    Real pr = wr_ipr + 0.5*b2r;

    Real qa,qb, lm,lp;
//    if (eos.is_adiabatic) {
    eos.FastMagnetosonicSpeedSR(wgas_l,b2l, wl_ipr, wl_ivx/u0l, 1.+u2l, lp, lm);
    eos.FastMagnetosonicSpeedSR(wgas_r,b2r, wr_ipr, wr_ivx/u0r, 1.+u2r, qb,qa);
//    } else {
//      qa = eos.FastMagnetosonicSpeed(wl_idn,bxi,wl_iby,wl_ibz);
//      qb = eos.FastMagnetosonicSpeed(wr_idn,bxi,wr_iby,wr_ibz);
//    }
    qa = fmax(-fmin(lm,qa), 0.);
    Real a = fmax(fmax(lp,qb), qa);
    

    //--- Step 3.  Compute L/R fluxes

    fl[IDN] = wl_idn * wl_ivx;
    qa = wgas_l * wl_ivx;
    qb = (bxi + b0l)*wl_ivx/u0l;
    fl[IVX] = qa*wl_ivx - (qb * qb) + pl;
    fl[IVY] = qa*wl_ivy - (qb * (wl_iby + b0l*wl_ivy)/u0l);
    fl[IVZ] = qa*wl_ivz - (qb * (wl_ibz + b0l*wl_ivz)/u0l);

    Real el = wgas_l*u0l*u0l - pl - wl_idn*u0l;
    fl[IEN] = (el + pl)*wl_ivx/u0l - b0l*qb;
    el -= b0l*b0l;

    fl[5  ] = (wl_iby*wl_ivx - bxi*wl_ivy)/u0l;
    fl[6  ] = (wl_ibz*wl_ivx - bxi*wl_ivz)/u0l;


    fr[IDN] = wr_idn * wr_ivx;
    qa = wgas_r * wr_ivx;
    qb = (bxi + b0r)*wr_ivx/u0r;
    fr[IVX] = qa*wr_ivx - (qb * bxi) + pr;
    fr[IVY] = qa*wr_ivy - (qb * (wr_iby + b0r*wr_ivy)/u0r);
    fr[IVZ] = qa*wr_ivz - (qb * (wr_ibz + b0r*wr_ivz)/u0r);

    Real er = wgas_r*u0r*u0r - pr - wr_idn*u0r;
    fr[IEN] = (er + pr)*wr_ivx/u0r - b0r * qb;

    er -= b0r*b0r;


    fr[5  ] = (wr_iby*wr_ivx - bxi*wr_ivy)/u0r;
    fr[6  ] = (wr_ibz*wr_ivx - bxi*wr_ivz)/u0r;


//    fl[IEN] = (((wl_ipr / gm1) + wl_ipr) * u0l + (wl_idn/(1.+ u0l)*u2l))*wl_ivx;
//    fr[IEN] = (((wr_ipr / gm1) + wr_ipr) * u0r + (wr_idn/(1.+ u0r)*u2r))*wr_ivx;
//    fl[IEN] = (((wl_ipr / gm1) + wl_ipr) * u0l + (wl_idn/(1.+ u0l)*u2l))*wl_ivx;
//    fr[IEN] = (((wr_ipr / gm1) + wr_ipr) * u0r + (wr_idn/(1.+ u0r)*u2r))*wr_ivx;

    du[IDN] = wr_idn*u0r          - wl_idn * u0l;
    du[IVX] = wgas_r*u0r*wr_ivx - wgas_l*u0l*wl_ivx;
    du[IVY] = wgas_r*u0r*wr_ivy - wgas_l*u0l*wl_ivy;
    du[IVZ] = wgas_r*u0r*wr_ivz - wgas_l*u0l*wl_ivz;
//    du[IEN] = (wr_ipr / gm1) * u0r*u0r + ( wr_ipr + wr_idn*u0r / (1.+ u0r))*u2r;
//    du[IEN]-= (wl_ipr / gm1) * u0l*u0l + ( wl_ipr + wl_idn*u0l / (1.+ u0l))*u2l;
    du[IEN] = er - el;

    du[5  ] = wr_iby - wl_iby;
    du[6  ] = wr_ibz - wl_ibz;

    //--- Step 5. Store results into 3D array of fluxes

    flx(m,IDN,k,j,i) = 0.5*(fl[IDN] + fr[IDN]) - a*du[IDN];
    flx(m,ivx,k,j,i) = 0.5*(fl[IVX] + fr[IVX]) - a*du[IVX];
    flx(m,ivy,k,j,i) = 0.5*(fl[IVY] + fr[IVY]) - a*du[IVY];
    flx(m,ivz,k,j,i) = 0.5*(fl[IVZ] + fr[IVZ]) - a*du[IVZ];
    flx(m,IEN,k,j,i) = 0.5*(fl[IEN] + fr[IEN]) - a*du[IEN];


    //--- Step 5.  Compute the LLF flux at interface (see Toro eq. 10.42).

    ey(m,k,j,i) = -0.5*(fl[5  ] + fr[5  ]) + a*du[5  ];
    ez(m,k,j,i) = 0.5*(fl[6  ] + fr[6  ]) - a*du[6  ];
  });

  return;
}

} // namespace mhd
