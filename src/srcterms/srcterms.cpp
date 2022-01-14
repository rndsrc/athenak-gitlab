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
#include <float.h>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
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
                                const EOS_Data &eos_data, const Real bdt){
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
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i){
    // temperature in cgs unit
    Real temp = 1.0;
    if (use_e){
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
//! \fn void SourceTerms::ISMCoolingNewTimeStep()
//! \brief Compute new time step for ISM cooling.

void SourceTerms::ISMCoolingNewTimeStep(const DvceArray5D<Real> &w0,
                                        const EOS_Data &eos_data){
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
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

  dtnew_cooling = std::numeric_limits<float>::max();

  // find smallest (e/cooling_rate) in each cell
  Kokkos::parallel_reduce("cooling_newdt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt){
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    // temperature in cgs unit
    Real temp = 1.0;
    Real eint = 1.0;
    if (use_e){
      temp = temp_unit*w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
      eint = w0(m,IEN,k,j,i);
    } else {
      temp = temp_unit*w0(m,ITM,k,j,i);
      eint = w0(m,ITM,k,j,i)*w0(m,IDN,k,j,i)/gm1;
    }
    
    Real lambda_cooling = ISMCoolFn(temp)/cooling_unit;
    Real gamma_heating = heating_rate/heating_unit;

    // add a tiny number
    Real cooling_heating = FLT_MIN + fabs(w0(m,IDN,k,j,i) *
                           (w0(m,IDN,k,j,i) * lambda_cooling - gamma_heating));

    min_dt = fmin((eint/cooling_heating), min_dt);
  }, Kokkos::Min<Real>(dtnew_cooling));
  return;
}
