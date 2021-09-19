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

#include <iostream>
#include "hybrid.hh"

#ifndef EOS_HYBRID_IMP_HH
#define EOS_HYBRID_IMP_HH

// ###################################################################################
// 	Private functions
// ###################################################################################

template <typename cold_eos>
inline double EOS_Hybrid<cold_eos>::eps_th__press_press_cold_rho(
    const double &press, const double &press_cold, const double &rho) {
  using namespace Margherita_helpers;
  return max(0., (press - press_cold) / (rho * (gamma_th_m1)));
}

template <typename cold_eos>
inline double EOS_Hybrid<cold_eos>::entropy__eps_th_rho(const double &eps_th,
                                                        const double &rho) {
  const auto eps_thL = std::max(eps_th, entropy_min);
  const double entropy = log(eps_thL * pow(rho, -gamma_th_m1)) / (gamma_th_m1);
  // This entropy diverges for small eps_th, so need to catch this here.
  //  if (std::isnan(entropy) || entropy < entropy_min)
  //    return entropy_min;
  return entropy;
}

template <typename cold_eos>
inline double EOS_Hybrid<cold_eos>::temp__eps_th(const double &eps_th) {
  using namespace Margherita_constants;
  using namespace Margherita_helpers;

  // Not really meaningfull to try to convert to MeV
  //  return max(0., mnuc_MeV * eps_th * (gamma_th_m1)); // FIXME: Check this
  return max(0., eps_th * (gamma_th_m1));  // FIXME: Check this
}

template <typename cold_eos>
inline double EOS_Hybrid<cold_eos>::eps_th__temp(const double &temp) {
  using namespace Margherita_constants;
  using namespace Margherita_helpers;

  // Not really meaningfull to try to convert to MeV
  //  return max(0., temp / (mnuc_MeV * (gamma_th_m1))); // FIXME: Check this
  return max(0., temp / ((gamma_th_m1)));  // FIXME: Check this
}

template <typename cold_eos>
inline double EOS_Hybrid<cold_eos>::csnd2__press_cold_eps_th_h_rho(
    const double &dpress_cold_drho, const double &press_cold,
    const double &eps_th, const double &h, const double &rho) {
  return (dpress_cold_drho + gamma_th_m1 * (gamma_th_m1 + 1.0) * eps_th) / h;
}

// ###################################################################################
// 	Public functions
// ###################################################################################
template <typename cold_eos>
double EOS_Hybrid<cold_eos>::press_temp__eps_rho_ye(double &temp, double &eps,
                                                    double &rho, double &ye,
                                                    error_type &error) {
  using namespace Margherita_constants;

  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);

  if (eps < eps_cold) eps = eps_cold;

  const double eps_th = eps - eps_cold;

  temp = temp__eps_th(eps_th);

  return press_cold + eps_th * rho * (gamma_th_m1);
}

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::press__eps_rho_ye(double &eps, double &rho,
                                               double &ye, error_type &error) {
  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);
  if (eps < eps_cold) eps = eps_cold;

  return press_cold + (eps - eps_cold) * rho * (gamma_th_m1);
}

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::press__temp_rho_ye(double &temp, double &rho,
                                                double &ye, error_type &error) {
  using namespace Margherita_constants;

  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);
  if (temp < 0.) temp = 0.;

  // Not really meaningfull to try to convert to MeV
  // return press_cold + rho * temp / mnuc_MeV; // FIXME Check this
  return press_cold + rho * temp;  // FIXME Check this
}

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::eps__temp_rho_ye(double &temp, double &rho,
                                              double &ye, error_type &error) {
  using namespace Margherita_constants;

  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);
  if (temp < 0.) temp = 0.;

  return eps_cold + eps_th__temp(temp);
}

//////////

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::press_cold__rho_ye(double &rho, double &ye,
                                                error_type &error) {
  double eps_cold;
  return cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);
}

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::eps_cold__rho_ye(double &rho, double &ye,
                                              error_type &error) {
  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);

  return eps_cold;
}

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::temp_cold__rho_ye(double &rho, double &ye,
                                               error_type &error) {
  return 0.0;
}
///////////

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::eps__press_temp_rho_ye(double &press, double &temp,
                                                    double &rho, double &ye,
                                                    error_type &error) {
  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);
  if (press < press_cold) press = press_cold;
  const double eps_th = eps_th__press_press_cold_rho(press, press_cold, rho);

  return eps_cold + eps_th;
}

template <typename cold_eos>
std::array<double, 2> EOS_Hybrid<cold_eos>::eps_range__rho_ye(
    double &rho, double &ye, error_type &error) {
  using namespace Margherita_helpers;

  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);
  std::array<double, 2> epsrange{{min(eps_cold, c2p_eps_max), c2p_eps_max}};

  return epsrange;
}

template <typename cold_eos>
std::array<double, 2> EOS_Hybrid<cold_eos>::entropy_range__rho_ye(
    double &rho, double &ye, error_type &error) {
  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);
  auto eps_th_max = c2p_eps_max - eps_cold;

  std::array<double, 2> range{{entropy__eps_th_rho(entropy_min, rho),
                               entropy__eps_th_rho(eps_th_max, rho)}};

  return range;
}

/////////////////

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::press_h_csnd2__eps_rho_ye(double &h, double &csnd2,
                                                       double &eps, double &rho,
                                                       double &ye,
                                                       error_type &error) {
  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);

  if (eps < eps_cold) eps = eps_cold;

  const double eps_th = eps - eps_cold;

  const double press = press_cold + eps_th * rho * (gamma_th_m1);

  h = 1. + eps + press / rho;

  auto dp_drho = cold_eos::dpress_cold_drho__rho(rho, error);
  csnd2 = csnd2__press_cold_eps_th_h_rho(dp_drho, press_cold, eps_th, h, rho);

  return press;
}

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::press_h_csnd2__temp_rho_ye(double &h,
                                                        double &csnd2,
                                                        double &temp,
                                                        double &rho, double &ye,
                                                        error_type &error) {
  using namespace Margherita_constants;

  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);

  if (temp < 0) temp = 0;
  const double eps_th = eps_th__temp(temp);

  // Not really meaningfull to try to convert to MeV
  // const double press = press_cold + rho * temp / mnuc_MeV; // FIXME Check
  // this
  const double press = press_cold + rho * temp;  // FIXME Check this

  h = 1. + eps_cold + eps_th + press / rho;

  auto dp_drho = cold_eos::dpress_cold_drho__rho(rho, error);
  csnd2 = csnd2__press_cold_eps_th_h_rho(dp_drho, press_cold, eps_th, h, rho);

  return press;
}

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::eps_h_csnd2__press_rho_ye(double &h, double &csnd2,
                                                       double &press,
                                                       double &rho, double &ye,
                                                       error_type &error) {
  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);
  if (press < press_cold) press = press_cold;

  const double eps_th = eps_th__press_press_cold_rho(press, press_cold, rho);

  h = 1. + eps_th + eps_cold + press / rho;

  auto dp_drho = cold_eos::dpress_cold_drho__rho(rho, error);
  csnd2 = csnd2__press_cold_eps_th_h_rho(dp_drho, press_cold, eps_th, h, rho);

  return eps_cold + eps_th;
}

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::press_eps_csnd2__temp_rho_ye(double &eps,
                                                        double &csnd2,
                                                        double &temp,
                                                        double &rho, double &ye,
                                                        error_type &error) {
  using namespace Margherita_constants;

  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);

  if (temp < 0) temp = 0;
  const double eps_th = eps_th__temp(temp);

  // Not really meaningfull to try to convert to MeV
  // const double press = press_cold + rho * temp / mnuc_MeV; // FIXME Check
  // this
  const double press = press_cold + rho * temp;  // FIXME Check this

  eps = eps_cold + eps_th;
  auto h = 1. + eps + press / rho;

  auto dp_drho = cold_eos::dpress_cold_drho__rho(rho, error);
  csnd2 = csnd2__press_cold_eps_th_h_rho(dp_drho, press_cold, eps_th, h, rho);

  return press;
}

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::press_h_csnd2_temp_entropy__eps_rho_ye(
    double &h, double &csnd2, double &temp, double &entropy, double &eps,
    double &rho, double &ye, error_type &error) {
  using namespace Margherita_constants;

  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);
  if (eps < eps_cold) eps = eps_cold;

  const double eps_th = eps - eps_cold;

  const double press = press_cold + eps_th * rho * (gamma_th_m1);

  h = 1. + eps + press / rho;

  auto dp_drho = cold_eos::dpress_cold_drho__rho(rho, error);
  csnd2 = csnd2__press_cold_eps_th_h_rho(dp_drho, press_cold, eps_th, h, rho);

  entropy = entropy__eps_th_rho(eps_th, rho);

  temp = temp__eps_th(eps_th);

  return press;
}

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::eps_csnd2_entropy__temp_rho_ye(
    double &csnd2, double &entropy, double &temp, double &rho, double &ye,
    error_type &error) {
  using namespace Margherita_constants;

  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);

  if (temp < 0) temp = 0;

  const double eps_th = eps_th__temp(temp);

  // Not really meaningfull to try to convert to MeV
  // const double press = press_cold + rho * temp / mnuc_MeV; // FIXME Check
  // this
  const double press = press_cold + rho * temp;  // FIXME Check this

  const double h = 1. + eps_cold + eps_th + press / rho;

  auto dp_drho = cold_eos::dpress_cold_drho__rho(rho, error);
  csnd2 = csnd2__press_cold_eps_th_h_rho(dp_drho, press_cold, eps_th, h, rho);

  entropy = entropy__eps_th_rho(eps_th, rho);

  return eps_cold + eps_th;
}

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::press_h_csnd2_temp_eps__entropy_rho_ye(
    double &h, double &csnd2, double &temp, double &eps, double &entropy,
    double &rho, double &ye, error_type &error) {
  using namespace Margherita_constants;

  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);

  //  if (entropy < 0)
  //    entropy = entropy_min;

  const double eps_th = exp(gamma_th_m1 * entropy) * pow(rho, gamma_th_m1);

  const double press = press_cold + eps_th * rho * (gamma_th_m1);

  eps = eps_cold + eps_th;

  h = 1. + eps + press / rho;

  auto dp_drho = cold_eos::dpress_cold_drho__rho(rho, error);
  csnd2 = csnd2__press_cold_eps_th_h_rho(dp_drho, press_cold, eps_th, h, rho);

  temp = temp__eps_th(eps_th);

  return press;
}

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::eps_h_csnd2_temp_entropy__press_rho_ye(
    double &h, double &csnd2, double &temp, double &entropy, double &press,
    double &rho, double &ye, error_type &error) {
  using namespace Margherita_constants;

  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);
  if (press < press_cold) press = press_cold;

  const double eps_th = eps_th__press_press_cold_rho(press, press_cold, rho);

  h = 1. + eps_th + eps_cold + press / rho;

  auto dp_drho = cold_eos::dpress_cold_drho__rho(rho, error);
  csnd2 = csnd2__press_cold_eps_th_h_rho(dp_drho, press_cold, eps_th, h, rho);

  entropy = entropy__eps_th_rho(eps_th, rho);

  // Not really meaningfull to try to convert to MeV
  // temp = mnuc_MeV * eps_th * (gamma_th_m1); // FIXME: Check this
  temp = temp__eps_th(eps_th);

  return eps_cold + eps_th;
}

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::press_eps_ye__beta_eq__rho_temp(
    double &eps, double &ye, double &rho, double &temp, error_type &error) {
  ye = 0;

  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);

  eps = eps_cold;
  temp = 0.;

  return press_cold;
}

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::rho__beta_eq__press_temp(double &press,
                                                      double &temp,
                                                      error_type &error) {
  return cold_eos::rho__press_cold(press, error);
}

template <typename cold_eos>
std::array<double, 4> EOS_Hybrid<cold_eos>::h_bound__rho1_rho2_ye(
    double &rho1, double &rho2, double &ye, error_type &error) {
  /*
   std::array<double,4> range;
   double eps_cold1;
   auto press_cold1 = cold_eos::press_cold_eps_cold__rho(eps_cold1,rho1,error);
   error_type error2;
   double eps_cold2;
   auto press_cold2 = cold_eos::press_cold_eps_cold__rho(eps_cold2,rho2,error);

   auto p_th2 = (c2p_eps_max-eps_cold2)

   h_bound[0] = 1.+ eps_cold1 + press_cold1/rho1;
   h_bound[0] = 1.+ c2p_eps_max + (press_cold1 + ())/rho1;
   */
  // Not implemented right now
  //
  std::array<double, 4> bounds{{1., 1.e99, 0, 1.e99}};

  return bounds;
}

// ###################################################################################
// 	Public functions -- EOS specific
// ###################################################################################

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::eps_th__press_rho(double &press, double &rho,
                                               error_type &error) {
  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);
  if (press < press_cold) press = press_cold;

  return eps_th__press_press_cold_rho(press, press_cold, rho);
}

template <typename cold_eos>
double EOS_Hybrid<cold_eos>::eps_th__eps_rho(double &eps, double &rho,
                                             error_type &error) {
  double eps_cold;
  auto press_cold = cold_eos::press_cold_eps_cold__rho(eps_cold, rho, error);
  if (eps < eps_cold) eps = eps_cold;

  return eps - eps_cold;
}

// Initialise pointers

// Gamma - 1 for the thermal part
template <typename cold_eos>
double EOS_Hybrid<cold_eos>::gamma_th_m1 = 0;
// Minimal value for the entropy, currently only for viscosity
template <typename cold_eos>
double EOS_Hybrid<cold_eos>::entropy_min = 1.e-10;
template <typename cold_eos>
double EOS_Hybrid<cold_eos>::c2p_h_min = 1.0;
template <typename cold_eos>
double EOS_Hybrid<cold_eos>::c2p_h_max = 1.e99;
template <typename cold_eos>
double EOS_Hybrid<cold_eos>::c2p_rho_atm = 0.;
template <typename cold_eos>
double EOS_Hybrid<cold_eos>::c2p_temp_atm = 0.;
template <typename cold_eos>
double EOS_Hybrid<cold_eos>::c2p_eps_atm = 0.;
template <typename cold_eos>
double EOS_Hybrid<cold_eos>::c2p_ye_atm = 0.0;
template <typename cold_eos>
double EOS_Hybrid<cold_eos>::c2p_eps_min = 0.;
template <typename cold_eos>
double EOS_Hybrid<cold_eos>::c2p_eps_max = 10.;
template <typename cold_eos>
double EOS_Hybrid<cold_eos>::c2p_press_max = 0.;  // Need to set this...
template <typename cold_eos>
bool EOS_Hybrid<cold_eos>::atm_beta_eq = true;  // Not really meaningful here

// template <typename cold_eos>
// double EOS_Hybrid<cold_eos>::temp_id = 0;

/*
template<typename cold_eos>
double * __restrict EOS_Hybrid<cold_eos>::h_max__rho_gf;
template<typename cold_eos>
double * __restrict EOS_Hybrid<cold_eos>::h_min__rho_gf;
template<typename cold_eos>
double * __restrict EOS_Hybrid<cold_eos>::press_min__rho_gf;
template<typename cold_eos>
double * __restrict EOS_Hybrid<cold_eos>::press_max__rho_gf;
*/
template <typename cold_eos>
double &EOS_Hybrid<cold_eos>::eos_rhomax = cold_eos::rhomax;
template <typename cold_eos>
double &EOS_Hybrid<cold_eos>::eos_rhomin = cold_eos::rhomin;
template <typename cold_eos>
double EOS_Hybrid<cold_eos>::eos_tempmax = 1.e99;
template <typename cold_eos>
double EOS_Hybrid<cold_eos>::eos_tempmin = 0;
template <typename cold_eos>
double EOS_Hybrid<cold_eos>::eos_yemax = 0;
template <typename cold_eos>
double EOS_Hybrid<cold_eos>::eos_yemin = 0;

#endif
