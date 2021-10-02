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
#include "../margherita.hh"

#ifndef EOS_HYBRID_HH
#define EOS_HYBRID_HH

template <typename cold_eos>
class EOS_Hybrid {
 public:
  typedef typename cold_eos::error_t error_type;

  typedef cold_eos cold_t;  // Convenience typedef to access the cold class
                            // after template instantiation

 private:
  static inline double eps_th__press_press_cold_rho(const double &press,
                                                    const double &press_cold,
                                                    const double &rho);
  static inline double entropy__eps_th_rho(const double &eps_th,
                                           const double &rho);
  static inline double temp__eps_th(const double &eps_th);
  static inline double eps_th__temp(const double &temp);
  static inline double csnd2__press_cold_eps_th_h_rho(
      const double &dpress_cold_drho, const double &press_cold,
      const double &eps_th, const double &h, const double &rho);

 public:
  // General accessors
  static double press__eps_rho_ye(double &eps, double &rho, double &ye,
                                  error_type &error);
  static double press_temp__eps_rho_ye(double &temp, double &eps, double &rho,
                                       double &ye, error_type &error);
  static double eps__temp_rho_ye(
      double &temp, double &rho, double &ye,
      error_type &error);  // This is ill-defined for polytropes
  static double press__temp_rho_ye(
      double &temp, double &rho, double &ye,
      error_type &error);  // This is ill-defined for polytropes
  static double press_cold__rho_ye(double &rho, double &ye, error_type &error);
  static double eps_cold__rho_ye(double &rho, double &ye, error_type &error);
  static double temp_cold__rho_ye(
      double &rho, double &ye,
      error_type &error);  // This is ill-defined for polytropes

  static double eps__press_temp_rho_ye(double &press, double &temp, double &rho,
                                       double &ye, error_type &error);

  static std::array<double, 2> eps_range__rho_ye(double &rho, double &ye,
                                                 error_type &error);

  static std::array<double, 2> entropy_range__rho_ye(double &rho, double &ye,
                                                     error_type &error);

  // Economical interface. We might want to include also the entropy
  static double press_h_csnd2__eps_rho_ye(double &h, double &csnd2, double &eps,
                                          double &rho, double &ye,
                                          error_type &error);
  static double press_h_csnd2__temp_rho_ye(double &h, double &csnd2,
                                           double &temp, double &rho,
                                           double &ye, error_type &error);
  static double eps_h_csnd2__press_rho_ye(double &h, double &csnd2,
                                          double &press, double &rho,
                                          double &ye, error_type &error);
  static double press_eps_csnd2__temp_rho_ye(double &eps, double &csnd2,
                                           double &temp, double &rho,
                                           double &ye, error_type &error);
  // Get everything
  static double press_h_csnd2_temp_entropy__eps_rho_ye(
      double &h, double &csnd2, double &temp, double &entropy, double &eps,
      double &rho, double &ye, error_type &error);

  static double eps_csnd2_entropy__temp_rho_ye(double &csnd2, double &entropy,
                                               double &temp, double &rho,
                                               double &ye, error_type &error);

  static double press_h_csnd2_temp_eps__entropy_rho_ye(
      double &h, double &csnd2, double &temp, double &eps, double &entropy,
      double &rho, double &ye, error_type &error);
  static double eps_h_csnd2_temp_entropy__press_rho_ye(
      double &h, double &csnd2, double &temp, double &entropy, double &press,
      double &rho, double &ye, error_type &error);
  // This function is here for ease of use
  // wth EOS_Tabulated, but really, we don't have beta equilibrium for
  // polytropes.
  static double press_eps_ye__beta_eq__rho_temp(double &eps, double &ye,
                                                double &rho, double &temp,
                                                error_type &error);

  static double rho__beta_eq__press_temp(double &press, double &temp,
                                         error_type &error);

  static std::array<double, 4> h_bound__rho1_rho2_ye(double &rho1, double &rho2,
                                                     double &ye,
                                                     error_type &error);

  // Specific function for this class
  static double eps_th__press_rho(double &press, double &rho,
                                  error_type &error);
  static double eps_th__eps_rho(double &eps, double &rho, error_type &error);

  // Gamma - 1 for the thermal part
  static double gamma_th_m1;
  // Minimal value for the entropy, currently only for viscosity
  static double entropy_min;

  static double &eos_rhomax, &eos_rhomin;
  static double eos_tempmin, eos_tempmax;
  static double eos_yemin, eos_yemax;

  // static double c2p_tempmin;
  // static double c2p_tempmax;
  static double c2p_ye_atm;
  static double c2p_rho_atm;
  static double c2p_temp_atm;
  static double c2p_eps_atm;
  static double c2p_eps_min;
  static double c2p_h_min;
  static double c2p_h_max;
  static double c2p_eps_max;
  static double c2p_press_max;
  static bool atm_beta_eq;

  static constexpr bool has_ye = false;

  /*
    static double * __restrict h_max__rho_gf;
    static double * __restrict h_min__rho_gf;
    static double * __restrict press_min__rho_gf;
    static double * __restrict press_max__rho_gf;
  */
};
#endif
