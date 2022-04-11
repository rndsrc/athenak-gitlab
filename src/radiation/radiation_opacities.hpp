//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_opacities.hpp
//! \brief implements functions for computing opacities

#include <math.h>

#include "athena.hpp"
#include "globals.hpp"

//----------------------------------------------------------------------------------------
//! \fn void OpacityFunction
//! \brief sets sigma_a, sigma_s, and sigma_p in the comoving frame

KOKKOS_INLINE_FUNCTION
void OpacityFunction(const Real dens, const Real temp,
                     const Real k_a, const Real k_s, const Real k_p,
                     bool cons_opacity, bool pow_opacity,
                     Real& sigma_a, Real& sigma_s, Real& sigma_p) {
  if (cons_opacity) {
    sigma_a = dens*k_a;
    sigma_s = dens*k_s;
    sigma_p = dens*k_p;
    return;
  }
  if (pow_opacity) {
    sigma_a = SQR(dens)*pow(temp, -3.5);
    sigma_s = 0.0;
    sigma_p = 0.0;
    return;
  }
}
