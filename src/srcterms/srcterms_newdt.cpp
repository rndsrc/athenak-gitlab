//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms_newdt.cpp
//! \brief function to compute timestep for source terms across all MeshBlock(s) in a
//! MeshBlockPack

#include <float.h>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "srcterms.hpp"
#include "ismcooling.hpp"
#include "units/units.hpp"

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::NewTimeStep()
//! \brief Compute new timestep for source terms.

void SourceTerms::NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  dtnew = static_cast<Real>(std::numeric_limits<float>::max());

  if (ism_cooling) {
    Real use_e = eos_data.use_e;
    Real gamma = eos_data.gamma;
    Real gm1 = gamma - 1.0;
    Real heating_rate = hrate;
    Real temp_unit = pmy_pack->punit->temperature_cgs();
    Real n_unit = pmy_pack->punit->density_cgs()/pmy_pack->punit->mu()
                  /pmy_pack->punit->atomic_mass_unit_cgs;
    Real cooling_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                        /n_unit/n_unit;
    Real heating_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                        /n_unit;

    // find smallest (e/cooling_rate) in each cell
    Kokkos::parallel_reduce("srcterms_cooling_newdt",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real &dens = w0(m,IDN,k,j,i);
      Real temp = 1.0;
      Real eint = 1.0;
      if (use_e) {
        temp = temp_unit*w0(m,IEN,k,j,i)/dens*gm1;
        eint = w0(m,IEN,k,j,i);
      } else {
        temp = temp_unit*w0(m,ITM,k,j,i);
        eint = w0(m,ITM,k,j,i)*dens/gm1;
      }

      Real lambda_cooling = ISMCoolFn(temp)/cooling_unit;
      Real gamma_heating = heating_rate/heating_unit;

      // add a tiny number
      Real cooling_heating = FLT_MIN + fabs(dens*(dens*lambda_cooling - gamma_heating));

      min_dt = fmin((eint/cooling_heating), min_dt);
    }, Kokkos::Min<Real>(dtnew));
  }

  return;
}
