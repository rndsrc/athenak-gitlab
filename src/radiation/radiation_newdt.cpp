//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_newdt.cpp
//! \brief function to compute rad timestep across all MeshBlock(s) in a MeshBlockPack

#include <math.h>

#include <limits>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "radiation.hpp"

namespace radiation {

//----------------------------------------------------------------------------------------
// \!fn void Radiation::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlockPack for radiation problems

TaskStatus Radiation::NewTimeStep(Driver *pdriver, int stage) {
  if (stage != (pdriver->nexp_stages)) {
    return TaskStatus::complete; // only execute last stage
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  // setup indicies for Kokkos parallel reduce
  auto &mbsize = pmy_pack->pmb->mb_size;
  const int nm = (pmy_pack->nmb_thispack);

  // find smallest (dx/c) in each direction for radiation problems
  Kokkos::parallel_reduce("RadiationNudt",Kokkos::RangePolicy<>(DevExeSpace(), 0, nm),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
    int m = (idx);
    min_dt1 = fmin((mbsize.d_view(m).dx1), min_dt1);
    min_dt2 = fmin((mbsize.d_view(m).dx2), min_dt2);
    min_dt3 = fmin((mbsize.d_view(m).dx3), min_dt3);
  }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }

  return TaskStatus::complete;
}
} // namespace radiation
