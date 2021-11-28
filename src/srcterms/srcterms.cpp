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
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "turb_driver.hpp"
#include "srcterms.hpp"
#include "eos/eos.hpp"
#include "utils/units.hpp" 

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

  // TODO: finish implementing cooling
  // (3) Optically thin (ISM) cooling
  ism_cooling = pin->GetOrAddBoolean(block,"ism_cooling",false);
  if (ism_cooling) {
    source_terms_enabled = true;
    mbar   = pin->GetReal(block,"mbar");
    kboltz = pin->GetReal(block,"kboltz");
    hrate  = pin->GetReal(block,"heating_rate");
  }

  // (4) Optically thin (ISM) cooling & heating
  cooling = pin->GetOrAddBoolean(block,"cooling",false);
  if (cooling) {
    source_terms_enabled = true;
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
  auto &indcs = pmy_pack->pmesh->mb_indcs;
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
// Add Shearing box source terms in the momentum and energy equations for Hydro.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::AddShearingBox(DvceArray5D<Real> &u0, const DvceArray5D<Real> &w0,
                                   const Real bdt)
{ 
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
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j)
      {
        par_for_inner(member, is, ie+1, [&](const int i)
        {
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
      }
    );
  }
  // TODO: add 3D shearing box

  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real CoolFn()
//! \brief SPEX cooling curve, taken from Table 2 of Schure et al, A&A 508, 751 (2009)

Real CoolFn(Real temp)
{
  // original data from Shure et al. paper, covers 4.12 < logt < 8.16
  const float lhd[102] = {
      -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497, -21.6211, -21.6595,
      -21.6426, -21.5688, -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
      -20.8986, -20.8281, -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051,
      -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749, -20.6709, -20.6748,
      -20.7089, -20.8031, -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
      -21.4538, -21.5055, -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073,
      -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243, -21.8875, -21.9738,
      -22.0671, -22.1537, -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
      -22.3590, -22.3512, -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595,
      -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353, -22.5659, -22.5895,
      -22.6059, -22.6161, -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
      -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844,
      -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928};

  Real logt = std::log10(temp);

  //  for temperatures less than 10^4 K, use Koyama & Inutsuka
  if (logt <= 4.2) {
    Real temp = pow(10.0,logt);
    return (2.0e-19*exp(-1.184e5/(temp + 1.0e3)) + 2.8e-28*sqrt(temp)*exp(-92.0/temp));
  }

  // for temperatures above 10^8.15 use CGOLS fit
  if (logt > 8.15) return pow(10.0, (0.45*logt - 26.065));

  // in between values of 4.2 < log(T) < 8.15
  // linear interpolation of tabulated SPEX cooling rate

  int ipps  = static_cast<int>(25.0*logt) - 103;
  ipps = (ipps < 100)? ipps : 100;
  ipps = (ipps > 0 )? ipps : 0;
  float x0    = 4.12 + 0.04*static_cast<float>(ipps);

  float dx    = logt - x0;
  Real tcool = (lhd[ipps+1]*dx - lhd[ipps]*(dx - 0.04))*25.0;
  return pow(10.0,tcool);
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::AddCoolingTerm()
//! \brief Add Cooling & heating source terms in the energy equations for Hydro.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::AddCoolingTerm(DvceArray5D<Real> &u0, const DvceArray5D<Real> &w0,
                                   const Real bdt)
{
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real gamma = pmy_pack->phydro->peos->eos_data.gamma;
  Real gm1 = gamma - 1.0;

  par_for("cooling", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i)
    {
      // Typical normalization assumes:
      // number density unit ~ 1 cm^-3, velocity unit ~ 500km/s, length unit ~ 1kpc
      
      // Temperature in c.g.s unit, typically normalized by ~ 2e7 K
      Real temp = punit->temperature*w0(m,ITM,k,j,i)/w0(m,IDN,k,j,i)*gm1; 
      // Lambda_cooling in code unit, typically normalized by ~ 1e-22
      Real lambda_cooling = CoolFn(temp)/(punit->energy_density/punit->time); 
      // ISM heating in code unit, typically normalized by ~ 1e-22
      Real gamma_heating = 2.0e-26/(punit->energy_density/punit->time);

      u0(m,IEN,k,j,i) -= bdt * w0(m,IDN,k,j,i) * 
                         (w0(m,IDN,k,j,i) * lambda_cooling - gamma_heating);
    }
  );
  return;
}
