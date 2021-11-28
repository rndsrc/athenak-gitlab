#ifndef UTILS_UNITS_HPP_
#define UTILS_UNITS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file units.hpp
//! \brief prototypes of unit and constant classes

// Athena++ headers
#include "../athena.hpp"

//! \brief Physical constants defined in c.g.s.
namespace constants {
static const Real pc    = 3.085678e+18;    //Parsec
static const Real kpc   = 3.085678e+21;    //Kiloparsec
static const Real kms   = 1.0e+5;          //Kilometer per second
static const Real m_hydrogen    = 1.6733e-24;      //Mass of hydrogen
static const Real k_boltz    = 1.380658e-16;    //Boltzmann constant
} // namespace constants

//----------------------------------------------------------------------------------------
//! \class Units
//! \brief data and definitions of functions used to store and access units
//  Functions are implemented in units.cpp
class Units {
 public:
  Units(Real dunit, Real lunit, Real vunit);
  Units(Real dunit, Real lunit, Real vunit, Real mu0);

  void UpdateUnits(Real dunit, Real lunit, Real vunit);
  void UpdateUnits(Real dunit, Real lunit, Real vunit, Real mu0);

  void SetUnitsConstants();

  bool fixed_mu;

  // Units in physical units
  // for converting variables from code unit to physical (c.g.s) unit
  // i.e., var_in_physical_unit = var_in_code_unit * unit
  Real mass, length, time;
  Real volume, density, velocity;
  Real energy_density, pressure;
  Real magnetic_field;
  Real temperature, mu;

  // Units in code units
  // for converting variables from physical (c.g.s) unit to code unit
  // i.e., var_in_code_unit = var_in_physical_unit * unit_code
  Real gram_code, cm_code, second_code, dyne_code, erg_code, kelvin_code;

  Real pc_code;
  Real kpc_code;
  Real kms_code;
  Real m_hydrogen_code;
  Real k_boltz_code;
};
extern Units *punit;
#endif // UTILS_UNITS_HPP_
