/*
 * =====================================================================================
 *
 *       Filename:  hot_slice.hh
 *
 *    Description:  Computes a slice at beta equilibrium from the full 3D table
 *
 *        Version:  1.0
 *        Created:  04/05/2017 20:22:37
 *       Revision:  none
 *       Compiler:  g++,icpc
 *
 *         Author:  Elias Roland Most (ERM), emost@itp.uni-frankfurt.de
 *   Organization:  Goethe University Frankfurt
 *
 * =====================================================================================
 */

#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../utils/linear_interp.hh"
#include "../utils/spline.hh"

#include "../margherita.hh"
#define COLDTABLE_SETUP
#include "../Margherita_EOS.h"
#include "../Cold/cold_table_implementation.hh"

// This routine computes a beta equilibrium slice from
// the full 3D Table. IMPORTANT: The beta equilibrium
// slice features small oscillations, that break dP/drho>0.
// An easy fix seems to be to linearly downsample the slice
// to about 100 points and interpolate this by using a cubic spline
// interp.

void HotSlice_beta_eq(const int downsample_rho, const int num_cold_points,
                      double temp) {
  using var_i = Hot_Slice::v_index;

  const auto num_rho_points = std::max(
      4, std::min(static_cast<int>(EOS_Tabulated::alltables.num_points[0]),
                  downsample_rho));
  const auto delta_rho_initial = (EOS_Tabulated::alltables.xmax<0>() -
                                  EOS_Tabulated::alltables.xmin<0>()) /
                                 (num_rho_points - 1);

  std::array<std::unique_ptr<double[]>, var_i::NUM_VARS> vars;
  auto rho_ptr = std::unique_ptr<double[]>{new double[num_rho_points]};
  for (int i = 0; i < var_i::NUM_VARS; ++i)
    vars[i] = std::unique_ptr<double[]>{new double[num_rho_points]};

  // Compute beta_equilibrium for all table points at  T=temp

  for (int i = 0; i < num_rho_points; ++i) {
    auto rhoL = EOS_Tabulated::alltables.xmin<0>() + i * delta_rho_initial;
    rho_ptr[i] = rhoL;
    rhoL = exp(rhoL);
    EOS_Tabulated::error_type error;
    vars[var_i::PRESS][i] = EOS_Tabulated::press_eps_ye__beta_eq__rho_temp(
        vars[var_i::EPS][i], vars[var_i::YE][i], rhoL, temp, error);
    vars[var_i::TEMP][i] = temp;

    // Get other auxilliaries
    EOS_Tabulated::eps_csnd2_entropy__temp_rho_ye(
        vars[var_i::CS2][i], vars[var_i::ENTROPY][i], temp, rhoL,
        vars[var_i::YE][i], error);
  }

  Hot_Slice::rhomin = EOS_Tabulated::eos_rhomin;
  Hot_Slice::rhomax = EOS_Tabulated::eos_rhomax;

  Hot_Slice::press_min = vars[var_i::PRESS][0];
  Hot_Slice::press_max = vars[var_i::PRESS][num_rho_points - 1];

  // Shift eps to ensure positivity
  if (vars[var_i::EPS][0] < 0) {
    Hot_Slice::energy_shift = -2. * vars[var_i::EPS][0];
  }
  assert(Hot_Slice::energy_shift >= 0);

  for (int i = 0; i < num_rho_points; ++i) {
    vars[var_i::EPS][i] = log(vars[var_i::EPS][i] + Hot_Slice::energy_shift);
    vars[var_i::PRESS][i] = log(vars[var_i::PRESS][i]);
    vars[var_i::TEMP][i] = log(vars[var_i::TEMP][i]);
  }

  // Setup cubic spline for interpolation
  const auto spline = cubic_spline_t<double, var_i::NUM_VARS>(
      num_rho_points, std::move(rho_ptr), std::move(vars[0]),
      std::move(vars[1]), std::move(vars[2]), std::move(vars[3]),
      std::move(vars[4]), std::move(vars[5]));

  //

  std::array<std::unique_ptr<double[]>, var_i::NUM_VARS> vars_lin;

  auto rho_ptr_lin = std::unique_ptr<double[]>{new double[num_cold_points]};

  for (int i = 0; i < var_i::NUM_VARS; ++i)
    vars_lin[i] = std::unique_ptr<double[]>{new double[num_cold_points]};

  // Interpolate
  const auto delta_rho = (EOS_Tabulated::alltables.xmax<0>() -
                          EOS_Tabulated::alltables.xmin<0>()) /
                         (num_cold_points - 1);

  for (int i = 0; i < num_cold_points; ++i) {
    auto rhoL = log(EOS_Tabulated::eos_rhomin) + delta_rho * i;
    rho_ptr_lin[i] = rhoL;
    auto interp = spline.interpolate_all(rhoL);
    for (int nn = 0; nn < var_i::NUM_VARS; ++nn) {
      vars_lin[nn][i] = interp[nn];
    }
  }

  Hot_Slice::lintp = linear_interp_t<double, var_i::NUM_VARS>(
      num_cold_points, std::move(rho_ptr_lin), std::move(vars_lin[0]),
      std::move(vars_lin[1]), std::move(vars_lin[2]), std::move(vars_lin[3]),
      std::move(vars_lin[4]), std::move(vars_lin[5]));
}

/*
 * Compute beta equilibrium at constant entropy. This is significantly more
 * involved
 * as we first have to compute beta equilibrium for all temperatures and then
 * interpolate to get the right one matching the entropy. This is a lot simpler
 * to do
 * e.g. in python but for completeness we implement this also in Margherita
 */

void HotSlice_beta_eq_isentropic(const int downsample_rho,
                                 const int num_cold_points,
                                 const double entropy_value) {
  using var_i = Hot_Slice::v_index;

  auto num_rho_points = std::min(
      static_cast<int>(EOS_Tabulated::alltables.num_points[0]), downsample_rho);
  const auto delta_rho_initial = (EOS_Tabulated::alltables.xmax<0>() -
                                  EOS_Tabulated::alltables.xmin<0>()) /
                                 (num_rho_points - 1);

  const auto delta_temp = (EOS_Tabulated::alltables.xmax<1>() -
                           EOS_Tabulated::alltables.xmin<1>()) /
                          (EOS_Tabulated::alltables.num_points[1] - 1);

  std::array<std::unique_ptr<double[]>, var_i::NUM_VARS> vars;

  // Create arrays on heap
  auto rho_ptr = std::unique_ptr<double[]>{new double[num_rho_points]};
  auto temp_ptr = std::unique_ptr<double[]>{
      new double[EOS_Tabulated::alltables.num_points[1]]};
  for (int i = 0; i < var_i::NUM_VARS; ++i)
    vars[i] = std::unique_ptr<double[]>{new double[num_rho_points]};

  std::vector<std::unique_ptr<double[]>> entropies(num_rho_points);
  for (auto &v : entropies)
    v = std::unique_ptr<double[]>{
        new double[EOS_Tabulated::alltables.num_points[1]]};

  // Compute beta_equilibrium for all table points at all temperatures, but
  // store only entropies

  for (int i = 0; i < EOS_Tabulated::alltables.num_points[1]; ++i) {
    temp_ptr[i] = EOS_Tabulated::alltables.xmin<1>() + i * delta_temp;
    auto temp = exp(temp_ptr[i]);
    for (int rr = 0; rr < num_rho_points; ++rr) {
      auto rhoL = EOS_Tabulated::alltables.xmin<0>() + rr * delta_rho_initial;
      rho_ptr[rr] = rhoL;
      rhoL = exp(rhoL);
      auto ye = double{};
      auto eps = double{};
      // Compute beta equilibrium
      EOS_Tabulated::error_type error;
      EOS_Tabulated::press_eps_ye__beta_eq__rho_temp(eps, ye, rhoL, temp,
                                                     error);
      auto csnd2 = double{};
      // Now get entropy
      EOS_Tabulated::eps_csnd2_entropy__temp_rho_ye(csnd2, entropies[rr][i],
                                                    temp, rhoL, ye, error);
    }
  }

  // As we now have the entropies, invert them pointwise to find the temperature

  auto temp_intp =
      cubic_spline_t<double, 1>(EOS_Tabulated::alltables.num_points[1],
                                std::move(temp_ptr), std::move(entropies[0]));
  //  auto temp_intp =
  //  linear_interp_t<double,1>(EOS_Tabulated::ntemp,std::move(temp_ptr),std::move(entropies[0]));

  const auto entropy_func = [&temp_intp, &entropy_value](const double &tempL) {
    return temp_intp.interpolate<0>(tempL)[0] - entropy_value;
  };

  const auto maxtemp = EOS_Tabulated::alltables.xmax<1>();
  const auto mintemp = EOS_Tabulated::alltables.xmin<1>();

  for (int rr = 0; rr < num_rho_points; ++rr) {
    vars[var_i::TEMP][rr] =
        exp(zero_brent<>(mintemp, maxtemp, 1.0e-13, entropy_func));
    if (rr < num_rho_points - 1)
      temp_intp.replace_y(std::move(entropies[rr + 1]));
  }

  for (int i = 0; i < num_rho_points; ++i) {
    auto rhoL = exp(rho_ptr[i]);
    EOS_Tabulated::error_type error;
    vars[var_i::PRESS][i] = EOS_Tabulated::press_eps_ye__beta_eq__rho_temp(
        vars[var_i::EPS][i], vars[var_i::YE][i], rhoL, vars[var_i::TEMP][i],
        error);

    // Get other auxilliaries
    EOS_Tabulated::eps_csnd2_entropy__temp_rho_ye(
        vars[var_i::CS2][i], vars[var_i::ENTROPY][i], vars[var_i::TEMP][i],
        rhoL, vars[var_i::YE][i], error);
  }

  Hot_Slice::rhomin = EOS_Tabulated::eos_rhomin;
  Hot_Slice::rhomax = EOS_Tabulated::eos_rhomax;

  Hot_Slice::press_min = vars[var_i::PRESS][0];
  Hot_Slice::press_max = vars[var_i::PRESS][num_rho_points - 1];

  // Shift eps to ensure positivity
  if (vars[var_i::EPS][0] < 0) {
    Hot_Slice::energy_shift = -2. * vars[var_i::EPS][0];
  }
  assert(Hot_Slice::energy_shift >= 0);

  for (int i = 0; i < num_rho_points; ++i) {
    vars[var_i::EPS][i] = log(vars[var_i::EPS][i] + Hot_Slice::energy_shift);
    vars[var_i::PRESS][i] = log(vars[var_i::PRESS][i]);
    vars[var_i::TEMP][i] = log(vars[var_i::TEMP][i]);
  }

  // Setup cubic spline for interpolation
  const auto spline = cubic_spline_t<double, var_i::NUM_VARS>(
      num_rho_points, std::move(rho_ptr), std::move(vars[0]),
      std::move(vars[1]), std::move(vars[2]), std::move(vars[3]),
      std::move(vars[4]), std::move(vars[5]));

  //

  std::array<std::unique_ptr<double[]>, var_i::NUM_VARS> vars_lin;

  auto rho_ptr_lin = std::unique_ptr<double[]>{new double[num_cold_points]};

  for (int i = 0; i < var_i::NUM_VARS; ++i)
    vars_lin[i] = std::unique_ptr<double[]>{new double[num_cold_points]};

  // Interpolate

  const auto delta_rho = (EOS_Tabulated::alltables.xmax<0>() -
                          EOS_Tabulated::alltables.xmin<0>()) /
                         (num_cold_points - 1);

  for (int i = 0; i < num_cold_points; ++i) {
    auto rhoL = log(EOS_Tabulated::eos_rhomin) + delta_rho * i;
    rho_ptr_lin[i] = rhoL;
    auto interp = spline.interpolate_all(rhoL);
    for (int nn = 0; nn < var_i::NUM_VARS; ++nn) {
      vars_lin[nn][i] = interp[nn];
    }
  }

  Hot_Slice::lintp = linear_interp_t<double, var_i::NUM_VARS>(
      num_cold_points, std::move(rho_ptr_lin), std::move(vars_lin[0]),
      std::move(vars_lin[1]), std::move(vars_lin[2]), std::move(vars_lin[3]),
      std::move(vars_lin[4]), std::move(vars_lin[5]));
}

// void HotSlice_beta_eq_isentropic_ye_const(const int downsample_rho,
//                                  const int num_cold_points,
//                                  const double entropy_value,
// 				 const double ye_value) {
//   using var_i = Hot_Slice::v_index;
//
//   auto num_rho_points = std::min(EOS_Tabulated::nrho, downsample_rho);
//   const auto delta_rho_initial =
//       (EOS_Tabulated::logrho[EOS_Tabulated::nrho - 1] -
//        EOS_Tabulated::logrho[0]) /
//       (num_rho_points - 1);
//   const auto delta_temp = (EOS_Tabulated::logtemp[EOS_Tabulated::ntemp - 1] -
//                            EOS_Tabulated::logtemp[0]) /
//                           (EOS_Tabulated::ntemp - 1);
//
//   std::array<std::unique_ptr<double[]>, var_i::NUM_VARS> vars;
//
//   // Create arrays on heap
//   auto rho_ptr = std::unique_ptr<double[]>{new double[num_rho_points]};
//   auto temp_ptr = std::unique_ptr<double[]>{new
//   double[EOS_Tabulated::ntemp]}; for (int i = 0; i < var_i::NUM_VARS; ++i)
//     vars[i] = std::unique_ptr<double[]>{new double[num_rho_points]};
//
//   std::vector<std::unique_ptr<double[]>> entropies(num_rho_points);
//   for (auto &v : entropies)
//     v = std::unique_ptr<double[]>{new double[EOS_Tabulated::ntemp]};
//
//   // Compute beta_equilibrium for all table points at all temperatures, but
//   // store only entropies
//
//   for (int i = 0; i < EOS_Tabulated::ntemp; ++i) {
//     temp_ptr[i] = EOS_Tabulated::logtemp[0] + i * delta_temp;
//     auto temp = exp(temp_ptr[i]);
//     for (int rr = 0; rr < num_rho_points; ++rr) {
//
//       auto rhoL = EOS_Tabulated::logrho[0] + rr * delta_rho_initial;
//       rho_ptr[rr] = rhoL;
//       rhoL = exp(rhoL);
//       auto ye = double{};
//       auto eps = double{};
//       // Compute beta equilibrium
//       EOS_Tabulated::error_type error;
//       auto csnd2 = double{};
//       // Now get entropy
//       EOS_Tabulated::eps_csnd2_entropy__temp_rho_ye(csnd2, entropies[rr][i],
//                                                     temp, rhoL, ye_value,
//                                                     error);
//     }
//   }
//
//   // As we now have the entropies, invert them pointwise to find the
//   temperature
//
//   auto temp_intp = cubic_spline_t<double, 1>(
//       EOS_Tabulated::ntemp, std::move(temp_ptr), std::move(entropies[0]));
//   //  auto temp_intp =
//   //
//   linear_interp_t<double,1>(EOS_Tabulated::ntemp,std::move(temp_ptr),std::move(entropies[0]));
//
//   const auto entropy_func = [&temp_intp, &entropy_value](const double &tempL)
//   {
//     return temp_intp.interpolate<0>(tempL)[0] - entropy_value;
//   };
//
//   const auto maxtemp = EOS_Tabulated::logtemp[EOS_Tabulated::ntemp - 1];
//   const auto mintemp = EOS_Tabulated::logtemp[0];
//
//   for (int rr = 0; rr < num_rho_points; ++rr) {
//     vars[var_i::TEMP][rr] =
//         exp(zero_brent<>(mintemp, maxtemp, 1.0e-13, entropy_func));
//     if (rr < num_rho_points - 1)
//       temp_intp.replace_y(std::move(entropies[rr + 1]));
//   }
//
//   for (int i = 0; i < num_rho_points; ++i) {
//     auto rhoL = exp(rho_ptr[i]);
//     vars[var_i::YE][i] = ye_value;
//     EOS_Tabulated::error_type error;
//     vars[var_i::PRESS][i] = EOS_Tabulated::press__temp_rho_ye(
//          vars[var_i::TEMP][i],rhoL,ye_value error);
//
//     vars[var_i::EPS][i] = EOS_Tabulated::eps__temp_rho_ye(
//          vars[var_i::TEMP][i],rhoL,ye_value error);
//
//     // Get other auxilliaries
//     EOS_Tabulated::eps_csnd2_entropy__temp_rho_ye(
//         vars[var_i::CS2][i], vars[var_i::ENTROPY][i], vars[var_i::TEMP][i],
//         rhoL, vars[var_i::YE][i], error);
//   }
//
//   Hot_Slice::rhomin = EOS_Tabulated::eos_rhomin;
//   Hot_Slice::rhomax = EOS_Tabulated::eos_rhomax;
//
//   Hot_Slice::press_min = vars[var_i::PRESS][0];
//   Hot_Slice::press_max = vars[var_i::PRESS][num_rho_points - 1];
//
//   // Shift eps to ensure positivity
//   if (vars[var_i::EPS][0] < 0) {
//     Hot_Slice::energy_shift = -2. * vars[var_i::EPS][0];
//   }
//   assert(Hot_Slice::energy_shift>=0);
//
//   for (int i = 0; i < num_rho_points; ++i) {
//     vars[var_i::EPS][i] = log(vars[var_i::EPS][i] + Hot_Slice::energy_shift);
//     vars[var_i::PRESS][i] = log(vars[var_i::PRESS][i]);
//     vars[var_i::TEMP][i] = log(vars[var_i::TEMP][i]);
//   }
//
//   // Setup cubic spline for interpolation
//   const auto spline = cubic_spline_t<double, var_i::NUM_VARS>(
//       num_rho_points, std::move(rho_ptr), std::move(vars[0]),
//       std::move(vars[1]), std::move(vars[2]), std::move(vars[3]),
//       std::move(vars[4]), std::move(vars[5]));
//
//   //
//
//   std::array<std::unique_ptr<double[]>, var_i::NUM_VARS> vars_lin;
//
//   auto rho_ptr_lin = std::unique_ptr<double[]>{new double[num_cold_points]};
//
//   for (int i = 0; i < var_i::NUM_VARS; ++i)
//     vars_lin[i] = std::unique_ptr<double[]>{new double[num_cold_points]};
//
//   // Interpolate
//   const auto delta_rho = (EOS_Tabulated::logrho[EOS_Tabulated::nrho - 1] -
//                           EOS_Tabulated::logrho[0]) /
//                          (num_cold_points - 1);
//
//   for (int i = 0; i < num_cold_points; ++i) {
//     auto rhoL = log(EOS_Tabulated::eos_rhomin) + delta_rho * i;
//     rho_ptr_lin[i] = rhoL;
//     auto interp = spline.interpolate_all(rhoL);
//     for (int nn = 0; nn < var_i::NUM_VARS; ++nn) {
//       vars_lin[nn][i] = interp[nn];
//     }
//   }
//
//   Hot_Slice::lintp = linear_interp_t<double, var_i::NUM_VARS>(
//       num_cold_points, std::move(rho_ptr_lin), std::move(vars_lin[0]),
//       std::move(vars_lin[1]), std::move(vars_lin[2]), std::move(vars_lin[3]),
//       std::move(vars_lin[4]), std::move(vars_lin[5]));
// }
