//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file units.cpp

// Athena++ headers
#include "athena.hpp"
#include "units.hpp"

//----------------------------------------------------------------------------------------
//! \brief Units constructor without mu
Units::Units(Real dunit, Real lunit, Real vunit) {
  density = dunit;
  length = lunit;
  velocity = vunit;
  mu = 1.0;
  fixed_mu = false;

  SetUnitsConstants();
}

//----------------------------------------------------------------------------------------
//! \brief Units constructor with fixed mu
Units::Units(Real dunit, Real lunit, Real vunit, Real mu0) {
  density = dunit;
  length = lunit;
  velocity = vunit;
  mu = mu0;
  fixed_mu = true;

  SetUnitsConstants();
}

//----------------------------------------------------------------------------------------
//! \fn void Units::UpdateUnits()
//! \brief Update units without mu
void Units::UpdateUnits(Real dunit, Real lunit, Real vunit) {
  density = dunit;
  length = lunit;
  velocity = vunit;
  mu = 1.0;
  fixed_mu = false;

  SetUnitsConstants();
}

//----------------------------------------------------------------------------------------
//! \fn void Units::UpdateUnits()
//! \brief Update units with fixed mu
void Units::UpdateUnits(Real dunit, Real lunit, Real vunit, Real mu0) {
  density = dunit;
  length = lunit;
  velocity = vunit;
  mu = mu0;
  fixed_mu = true;

  SetUnitsConstants();
}

//----------------------------------------------------------------------------------------
//! \fn void Units::SetUnitsConstants()
//! \brief Set values of units in code
void Units::SetUnitsConstants() {
  // Set unit for converting variables from code unit to physical (c.g.s) unit
  // i.e., var_in_physical_unit = var_in_code_unit * unit
  volume = length*length*length;
  mass = density*volume;
  time = length/velocity;
  energy_density = pressure = density*velocity*velocity;
  magnetic_field = std::sqrt(4.*M_PI*pressure);
  temperature = pressure/density*mu*constants::m_hydrogen/constants::k_boltz;

  // Set unit_code for converting variables from physical (c.g.s) unit to code unit
  // i.e., var_in_code_unit = var_in_physical_unit * unit_code
  cm_code = 1.0/length;
  gram_code = 1.0/mass;
  second_code = 1.0/time;
  dyne_code = gram_code*cm_code/(second_code*second_code);
  erg_code = gram_code*cm_code*cm_code/(second_code*second_code);
  kelvin_code = 1.0/temperature;

  pc_code = constants::pc * cm_code;
  kpc_code = constants::kpc * cm_code;
  kms_code = constants::kms * cm_code/second_code;
  m_hydrogen_code = constants::m_hydrogen * gram_code;
  k_boltz_code = constants::k_boltz * erg_code/kelvin_code;
}

// Initialize the global Units object with default parameters
Real mu = 0.618;
Real dunit = mu*constants::m_hydrogen; // density in code units is number density
Real lunit = 1.0*constants::pc; // length in code units is parsec
Real vunit = 1.0*constants::kms; // velocity in code units is km/s
Units *punit = new Units(dunit, lunit, vunit, mu);
