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

#include <functional>
#include <iostream>
#include "../utils/brent.hh"
#include "../utils/linear_interp.hh"
#include "../utils/spline.hh"
#include "cold_table.hh"

#ifndef COLD_TABLE_IMP_HH
#define COLD_TABLE_IMP_HH

// ###################################################################################
// 	Private functions
// ###################################################################################

template <int extra_vars,template<typename,int> class interp_t>
inline typename Cold_Table_t<extra_vars,interp_t>::error_t
Cold_Table_t<extra_vars,interp_t>::check_range(double &rho) {
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

// ###################################################################################
// 	Public functions
// ###################################################################################

template <int extra_vars,template<typename,int> class interp_t>
inline double Cold_Table_t<extra_vars, interp_t>::press_cold_eps_cold__rho(
    double &eps_cold, double &rho, error_t &error) {
  error = check_range(rho);

  const auto res = lintp.interpolate(log(rho), v_index::PRESS, v_index::EPS);

  eps_cold = exp(res[v_index::EPS]) - energy_shift;
  return exp(res[v_index::PRESS]);
}

template <int extra_vars, template<typename,int> class interp_t>
inline double Cold_Table_t<extra_vars, interp_t>::dpress_cold_drho__rho(double &rho,
                                                              error_t &error) {
  error = check_range(rho);

  auto res = lintp.get_derivative(log(rho), v_index::PRESS);
  const auto res2 = lintp.interpolate(log(rho), v_index::PRESS);

  // res = dlog P/dlog \rho so need to multiply by P/rho
  return res[v_index::PRESS] * exp(res2[v_index::PRESS]) / rho;
}

template <int extra_vars,template<typename,int> class interp_t>
inline double Cold_Table_t<extra_vars,interp_t>::rho__press_cold(double &press_cold,
                                                        error_t &error) {
  // Range check
  if (press_cold < press_min) {
    press_cold = press_min;
    error[0] = true;
    return rhomin;
  }

  // Range check
  if (press_cold > press_max) {
    press_cold = press_max;
    error[1] = true;
    return rhomax;
  }

  const auto func = [&](const double &lrho) {
    const auto pL = lintp.interpolate(lrho, v_index::PRESS);
    return log(press_cold) - pL[v_index::PRESS];
  };

  auto lrho =
      zero_brent<>(1.001 * log(rhomin), 0.999 * log(rhomax), 1.0e-13, func);

  return exp(lrho);
}

// template<int extra_vars>
// inline double Cold_Table_t<extra_vars>::rho__h_cold(double &h_cold, error_t
// &error) {
//
//
//   const auto epsmin_vec = lintp.interpolate(log(rhomin),v_index::EPS);
//   const auto epsmax_vec = lintp.interpolate(log(rhomax),v_index::EPS);
//
//   const auto h_max = 1. + epsmax_vec[v_index::EPS] + press_max/rhomax;
//   const auto h_min = 1. + epsmin_vec[v_index::EPS] + press_min/rhomin;
//
//   // Range check
//   if (h_cold < h_min) {
//     h_cold = h_min;
//     error[0] = true;
//     return rhomin;
//   }
//
//   // Range check
//   if (h_cold > h_max) {
//     h_cold = h_max;
//     error[1] = true;
//     return rhomax;
//   }
//
//   const auto func = [&](const double &lrho){
//     				const auto pL =
//     lintp.interpolate(lrho,v_index::PRESS, v_index::EPS);
//  				return h_cold -( 1. + pL[v_index::EPS] +
//  pL[v_index::PRESS]/exp(lrho)); };
//
//   auto lrho= zero_brent<>(1.001*log(rhomin), 0.999*log(rhomax), 1.0e-13,
//   func);
//
//   return exp(lrho);
//
// }

template <int extra_vars,template<typename,int> class interp_t>
inline std::array<double, 2 + extra_vars>
Cold_Table_t<extra_vars,interp_t>::get_extra_quantities(double &rho, error_t &error) {
  error = check_range(rho);
  return lintp.interpolate_all(log(rho));
}

  // Initialise pointers

#ifdef COLDTABLE_SETUP

// These are specific to this class
template <int extra_vars,template<typename,int> class interp_t>
interp_t<double, 2 + extra_vars> Cold_Table_t<extra_vars,interp_t>::lintp;

template <int extra_vars,template<typename,int> class interp_t>
double Cold_Table_t<extra_vars,interp_t>::energy_shift = 0;

template <int extra_vars,template<typename,int> class interp_t>
double Cold_Table_t<extra_vars,interp_t>::press_max;
template <int extra_vars,template<typename,int> class interp_t>
double Cold_Table_t<extra_vars,interp_t>::press_min;

template <int extra_vars,template<typename,int> class interp_t>
double Cold_Table_t<extra_vars,interp_t>::rhomin;
template <int extra_vars,template<typename,int> class interp_t>
double Cold_Table_t<extra_vars,interp_t>::rhomax;

#endif

#endif
