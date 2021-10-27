//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_newdt.cpp
//  \brief function to compute radiation timestep across all MeshBlock(s) in a MeshBlockPack

#include <limits>
#include <math.h>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "radiation.hpp"

namespace radiation {

//----------------------------------------------------------------------------------------
// \!fn void Radiation::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlock for vacuum transfer

TaskStatus Radiation::NewTimeStep(Driver *pdriver, int stage)
{
  if (stage != (pdriver->nexp_stages)) {
    return TaskStatus::complete; // only execute last stage
  }

  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  // capture class variables for kernel
  auto &mbsize = pmy_pack->coord.coord_data.mb_size;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;

  Kokkos::parallel_reduce("radNudt2",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3)
    {
      int m = (idx)/nkji;
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
