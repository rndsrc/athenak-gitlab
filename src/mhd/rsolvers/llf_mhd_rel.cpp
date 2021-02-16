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


    Real b0 = bxi * wl_ivx + wl_iby * wl_ivy + wl_ibz * wl_ivz;

    Real u2 = SQR(wl_ivz) + SQR(wl_ivy) + SQR(wl_ivx);

    Real u0  = sqrt(1. + u2);

    Real b2 = ((bxi*bxi + wl_iby*wl_iby + wl_ibz*wl_ibz) + b0*b0) / (1.+u2);

    // FIXME ERM: Ideal fluid for now
    Real wgas = wl_idn + (eos.gamma/gm1) * wl_ipr;

    Real p = wl_ipr + 0.5*b2;

    Real llm,llp, lm,lp;
//    if (eos.is_adiabatic) {
    eos.FastMagnetosonicSpeedSR(wgas,b2, wl_ipr, wl_ivx/u0, 1.+u2, lp, lm);

    wgas += b2;

    Real &qa = llp;
    Real &qb = llm;

    fl[IDN] = wl_idn * wl_ivx;
    qa = wgas * wl_ivx;
    qb = (bxi + b0*wl_ivx)/u0;
    fl[IVX] = qa*wl_ivx - (qb * qb) + p;
    fl[IVY] = qa*wl_ivy - (qb * (wl_iby + b0*wl_ivy)/u0);
    fl[IVZ] = qa*wl_ivz - (qb * (wl_ibz + b0*wl_ivz)/u0);

    Real e = wgas*u0*u0 - wl_idn*u0;
    fl[IEN] = e *wl_ivx/u0 - b0*qb;
    e -= b0*b0 + p;

    du[IEN] = -e;
    du[IDN] = - wl_idn * u0;

    fl[5  ] = (wl_iby*wl_ivx - bxi*wl_ivy)/u0;
    fl[6  ] = (wl_ibz*wl_ivx - bxi*wl_ivz)/u0;


    du[IVX] = -qa*u0 + b0*qb;
    du[IVY] = -wgas*u0*wl_ivy + b0*(wl_iby + b0*wl_ivy)/u0;
    du[IVZ] = -wgas*u0*wl_ivz + b0*(wl_ibz + b0*wl_ivz)/u0;


    b0 = bxi * wr_ivx + wr_iby * wr_ivy + wr_ibz * wr_ivz;
    u2 = SQR(wr_ivz) + SQR(wr_ivy) + SQR(wr_ivx);
    u0  = sqrt(1. + u2);
    b2 = ((bxi*bxi + wr_iby*wr_iby + wr_ibz*wr_ibz) + b0*b0) / (1.+u2);
    wgas = wr_idn + (eos.gamma/gm1) * wr_ipr;
    p = wr_ipr + 0.5*b2;

    eos.FastMagnetosonicSpeedSR(wgas,b2, wr_ipr, wr_ivx/u0, 1.+u2, llp,llm);
//    } else {
//      qa = eos.FastMagnetosonicSpeed(wl_idn,bxi,wl_iby,wl_ibz);
//      qb = eos.FastMagnetosonicSpeed(wr_idn,bxi,wr_iby,wr_ibz);
//    }
    qa = fmax(-fmin(lm,llm), 0.);
    Real a = fmax(fmax(lp,llp), llm);

    wgas += b2;

    

    //--- Step 3.  Compute L/R fluxes
    //
    fr[IDN] = wr_idn * wr_ivx;
    qa = wgas * wr_ivx;
    qb = (bxi + b0*wr_ivx)/u0;
    fr[IVX] = qa*wr_ivx - (qb * qb) + p;
    fr[IVY] = qa*wr_ivy - (qb * (wr_iby + b0*wr_ivy)/u0);
    fr[IVZ] = qa*wr_ivz - (qb * (wr_ibz + b0*wr_ivz)/u0);

    e = wgas*u0*u0 - wr_idn*u0;
    fr[IEN] = e*wr_ivx/u0 - b0 * qb;

    e -= b0*b0 + p;

    du[IVX] += qa*u0            - b0*qb;
    du[IVY] += wgas*u0*wr_ivy - b0*(wr_iby + b0*wr_ivy)/u0;
    du[IVZ] += wgas*u0*wr_ivz - b0*(wr_ibz + b0*wr_ivz)/u0;

    du[IDN] +=  wr_idn * u0;
    du[IEN] += e;


    fr[5  ] = (wr_iby*wr_ivx - bxi*wr_ivy)/u0;
    fr[6  ] = (wr_ibz*wr_ivx - bxi*wr_ivz)/u0;


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
