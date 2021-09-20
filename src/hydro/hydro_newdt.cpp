//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_newdt.cpp
//  \brief function to compute hydro timestep across all MeshBlock(s) in a MeshBlockPack

#include <limits>
#include <math.h>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "hydro.hpp"

#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
// \!fn void Hydro::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlock for hydrodynamic problems

TaskStatus Hydro::NewTimeStep(Driver *pdriver, int stage)
{
  if (stage != (pdriver->nexp_stages)) {
    return TaskStatus::complete; // only execute last stage
  }
  
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  auto &eos = pmy_pack->phydro->peos->eos_data;
  auto &coord = pmy_pack->coord.coord_data;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  // capture class variables for kernel
  auto &w0_ = w0;
  auto &mbsize = pmy_pack->coord.coord_data.mb_size;
  auto &is_special_relativistic_ = is_special_relativistic;
  auto &is_general_relativistic_ = is_general_relativistic;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  if (pdriver->time_evolution == TimeEvolution::kinematic) {
    // find smallest (dx/v) in each direction for advection problems
    Kokkos::parallel_reduce("HydroNudt1",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3)
      {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      min_dt1 = fmin((mbsize.d_view(m).dx1/fabs(w0_(m,IVX,k,j,i))), min_dt1);
      min_dt2 = fmin((mbsize.d_view(m).dx2/fabs(w0_(m,IVY,k,j,i))), min_dt2);
      min_dt3 = fmin((mbsize.d_view(m).dx3/fabs(w0_(m,IVZ,k,j,i))), min_dt3);
    }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));
 
  } else {

    // find smallest dx/(v +/- C) in each direction for hydrodynamic problems
    Kokkos::parallel_reduce("HydroNudt2",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3)
      { 
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real max_dv1 = 0.0, max_dv2 = 0.0, max_dv3 = 0.0;

      if (is_general_relativistic_) {

        // ERM: THIS IS NOT ALWAYS CORRECT
        max_dv1 = 1.0;
        max_dv2 = 1.0;
        max_dv3 = 1.0;


	// ERM: Fastest characteristics in x-dir are
	// Note that both left and right going characteristics
	// have -betax.
	// -+ alpha* sqrt(gi^xx) - betax

	// Extract components of metric
	Real &x1min = coord.mb_size.d_view(m).x1min;
	Real &x1max = coord.mb_size.d_view(m).x1max;
	Real &x2min = coord.mb_size.d_view(m).x2min;
	Real &x2max = coord.mb_size.d_view(m).x2max;
	Real &x3min = coord.mb_size.d_view(m).x3min;
	Real &x3max = coord.mb_size.d_view(m).x3max;
	int nx1 = coord.mb_indcs.nx1;
	int nx2 = coord.mb_indcs.nx2;
	int nx3 = coord.mb_indcs.nx3;
	Real x1v,x2v,x3v;
	x1v = CellCenterX  (i-is, nx1, x1min, x1max);
	x2v = CellCenterX(j-js, nx2, x2min, x2max);
	x3v = CellCenterX(k-ks, nx3, x3min, x3max);

	Real g_[NMETRIC], gi_[NMETRIC];
	ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, false,
				coord.bh_spin, g_, gi_);

	const Real
	  &g_00 = g_[I00], &g_01 = g_[I01], &g_02 = g_[I02], &g_03 = g_[I03],
	  &g_10 = g_[I01], &g_11 = g_[I11], &g_12 = g_[I12], &g_13 = g_[I13],
	  &g_20 = g_[I02], &g_21 = g_[I12], &g_22 = g_[I22], &g_23 = g_[I23],
	  &g_30 = g_[I03], &g_31 = g_[I13], &g_32 = g_[I23], &g_33 = g_[I33];
	const Real
	  &g00 = gi_[I00], &g01 = gi_[I01], &g02 = gi_[I02], &g03 = gi_[I03],
	  &g10 = gi_[I01], &g11 = gi_[I11], &g12 = gi_[I12], &g13 = gi_[I13],
	  &g20 = gi_[I02], &g21 = gi_[I12], &g22 = gi_[I22], &g23 = gi_[I23],
	  &g30 = gi_[I03], &g31 = gi_[I13], &g32 = gi_[I23], &g33 = gi_[I33];
	Real alpha = std::sqrt(-1.0/g00);

	Real betax = -g01/g00;
	Real sqrtg = sqrt(fabs(gi_[I11] - gi_[I01]*gi_[I01]/gi_[I00]));
	max_dv1 = fmax(fabs(-alpha*sqrtg - betax),
	    	       fabs(alpha*sqrtg - betax));

	betax = -g02/g00;
	sqrtg = sqrt(fabs(gi_[I22] - gi_[I02]*gi_[I02]/gi_[I00]));
	max_dv2 = fmax(fabs(-alpha*sqrtg - betax),
	    	       fabs(alpha*sqrtg - betax));

	betax = -g03/g00;
	sqrtg = sqrt(fabs(gi_[I33] - gi_[I03]*gi_[I03]/gi_[I00]));
	max_dv3 = fmax(fabs(-alpha*sqrtg - betax),
	    	       fabs(alpha*sqrtg - betax));

      } else if (is_special_relativistic_) {
        Real v2 = SQR(w0_(m,IVX,k,j,i)) + SQR(w0_(m,IVY,k,j,i)) + SQR(w0_(m,IVZ,k,j,i));
        Real lf = sqrt(1.0 + v2);
        // FIXME ERM: Ideal fluid for now
        Real h = w0_(m,IDN,k,j,i) + (eos.gamma/(eos.gamma-1.)) * w0_(m,IPR,k,j,i);
        Real lm, lp;

        eos.WaveSpeedsSR(h, w0_(m,IPR,k,j,i), w0_(m,IVX,k,j,i)/lf, lf*lf, lp, lm);
        max_dv1 = fmax(fabs(lm), lp);

        eos.WaveSpeedsSR(h, w0_(m,IPR,k,j,i), w0_(m,IVX,k,j,i)/lf, lf*lf, lp, lm);
        max_dv2 = fmax(fabs(lm), lp);

        eos.WaveSpeedsSR(h, w0_(m,IPR,k,j,i), w0_(m,IVZ,k,j,i)/lf, lf*lf, lp, lm);
        max_dv3 = fmax(fabs(lm), lp);

      } else {
        Real cs;
        if (eos.is_ideal) {
          cs = eos.SoundSpeed(w0_(m,IPR,k,j,i),w0_(m,IDN,k,j,i));
        } else         {
          cs = eos.iso_cs;
        }
        max_dv1 = fabs(w0_(m,IVX,k,j,i)) + cs;
        max_dv2 = fabs(w0_(m,IVY,k,j,i)) + cs;
        max_dv3 = fabs(w0_(m,IVZ,k,j,i)) + cs;
      }

      min_dt1 = fmin((mbsize.d_view(m).dx1/max_dv1), min_dt1);
      min_dt2 = fmin((mbsize.d_view(m).dx2/max_dv2), min_dt2);
      min_dt3 = fmin((mbsize.d_view(m).dx3/max_dv3), min_dt3);
    }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));
 
  }

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }

  return TaskStatus::complete;
}
} // namespace hydro
