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
#include "../utils/linear_interp.hh"
#include "../utils/spline.hh"

#ifndef COLD_TABLE_HH
#define COLD_TABLE_HH

template <int extra_vars = 0, template<typename,int> class interp_t = linear_interp_t>
class Cold_Table_t {
 public:
  using error_t = std::bitset<2>;

  using this_interp_t = interp_t<double, 2 + extra_vars>;

  static inline error_t rho_range();

  enum v_index { EPS = 0, PRESS, YE, TEMP, ENTROPY, CS2, NUM_VARS };

 private:
  static inline error_t check_range(double &rho);

 public:
  // General definitions for cold EOS

  static inline double dpress_cold_drho__rho(double &rho, error_t &error);

  static inline double press_cold_eps_cold__rho(double &eps_cold, double &rho,
                                                error_t &error);

  static inline double rho__press_cold(double &press_cold, error_t &error);

  // Hot Slice capabilities
  // Only used for ID
  static inline std::array<double, 2 + extra_vars> get_extra_quantities(
      double &rho, error_t &error);

  // Specific to Cold_Table
  static double energy_shift;

  static interp_t<double, 2 + extra_vars> lintp;

  static double press_min;
  static double press_max;
  static double rhomin;
  static double rhomax;
};
#endif
