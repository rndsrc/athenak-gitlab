//
// This file is part of Margherita, the light-weight EOS framework
//
//  Copyright (C) 2017, Elias Roland Most
//                      <emost@th.physik.uni-frankfurt.de>
//  Copyright (C) 2017, Ludwig Jens Papenfort
//                      <papenfort@th.physik.uni-frankfurt.de>
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

#include <array>
#include <bitset>
#include "../margherita.hh"

#ifndef COLD_PWPOLY_HH
#define COLD_PWPOLY_HH

class Cold_PWPoly {
 public:
  typedef std::bitset<2> error_t;

  static inline error_t rho_range();

 private:
  static inline error_t check_range(double &rho);

  static inline int find_piece__rho(double &rho, error_t &error);

 public:
  // General definitions for cold EOS

  static inline double dpress_cold_drho__rho(double &rho, error_t &error);

  static inline double press_cold_eps_cold__rho(double &eps_cold, double &rho,
                                                error_t &error);

  static inline double rho__press_cold(double &press_cold, error_t &error);

  // Specific to PWPoly

  static inline double gamma_cold__rho(double &rho, error_t &error);

  static inline double gamma_cold_eps_tab__rho(double &eps_tabL, double &rho,
                                               error_t &error);

  static constexpr int max_num_pieces = 30;

  static std::array<double, max_num_pieces> k_tab;
  static std::array<double, max_num_pieces> gamma_tab;
  static std::array<double, max_num_pieces> rho_tab;
  static std::array<double, max_num_pieces> eps_tab;
  static std::array<double, max_num_pieces> P_tab;
  static int num_pieces;

  static double rhomin;
  static double rhomax;
};
#endif
