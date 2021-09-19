//
// This file is part of Margherita, the light-weight EOS framework
//
//  Copyright (C) 2017, Elias Roland Most
//                      <emost@th.physik.uni-frankfurt.de>
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream>
#include "cold_pwpoly.hh"

#ifndef COLD_PWPOLY_IMP_HH
#define COLD_PWPOLY_IMP_HH

// ###################################################################################
// 	Private functions
// ###################################################################################

inline Cold_PWPoly::error_t Cold_PWPoly::check_range(double &rho) {
  error_t error;
  if (rho < rhomin) {
    error[0] = true;
    rho = rhomin;

    return error;

  } else if (rho > rhomax) {
    error[1] = true;
    rho = rhomax;
  }

  return error;
}

inline int Cold_PWPoly::find_piece__rho(double &rho, error_t &error) {
  error = check_range(rho);

  for (int nn = 1; nn < num_pieces; nn++) {
    if (rho <= rho_tab[nn] && rho > rho_tab[nn - 1]) {
      return nn - 1;
    }
  }
  // We are in the last piece
  return num_pieces - 1;
}

inline double Cold_PWPoly::press_cold_eps_cold__rho(double &eps_cold,
                                                    double &rho,
                                                    error_t &error) {
  auto index = find_piece__rho(rho, error);

  auto press_cold = k_tab[index] * pow(rho, gamma_tab[index]);

  eps_cold = eps_tab[index] + press_cold / (rho * (gamma_tab[index] - 1.));

  return press_cold;
}

inline double Cold_PWPoly::dpress_cold_drho__rho(double &rho, error_t &error) {
  auto index = find_piece__rho(rho, error);

  auto press_cold = k_tab[index] * pow(rho, gamma_tab[index]);
  return gamma_tab[index] * press_cold / rho;
}

inline double Cold_PWPoly::gamma_cold__rho(double &rho, error_t &error) {
  auto index = find_piece__rho(rho, error);
  return gamma_tab[index];
}

inline double Cold_PWPoly::gamma_cold_eps_tab__rho(double &eps_tabL,
                                                   double &rho,
                                                   error_t &error) {
  auto index = find_piece__rho(rho, error);
  eps_tabL = eps_tab[index];
  return gamma_tab[index];
}

inline double Cold_PWPoly::rho__press_cold(double &press_cold, error_t &error) {
  // Range check
  if (press_cold < P_tab[0]) {
    press_cold = P_tab[0];
    error[0] = true;
    return rho_tab[0];
  }

  // Find index in Press
  int index = num_pieces - 1;
  for (int nn = 1; nn < num_pieces; nn++) {
    if (press_cold <= P_tab[nn] && press_cold > P_tab[nn - 1]) {
      index = nn - 1;
    }
  }

  return pow(press_cold / k_tab[index], 1. / gamma_tab[index]);
}

  // Initialise pointers

#ifdef PWPOLY_SETUP

// These are specific to this class
std::array<double, Cold_PWPoly::max_num_pieces> Cold_PWPoly::k_tab;
std::array<double, Cold_PWPoly::max_num_pieces> Cold_PWPoly::gamma_tab;
std::array<double, Cold_PWPoly::max_num_pieces> Cold_PWPoly::rho_tab;
std::array<double, Cold_PWPoly::max_num_pieces> Cold_PWPoly::eps_tab;
std::array<double, Cold_PWPoly::max_num_pieces> Cold_PWPoly::P_tab;
int Cold_PWPoly::num_pieces = 1;

double Cold_PWPoly::rhomin;
double Cold_PWPoly::rhomax;

#endif

#endif
