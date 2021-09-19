//
//  Copyright (C) 2017, Elias Roland Most
//  			<emost@th.physik.uni-frankfurt.de>
//  			Ludwig Jens Papenfort
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
//
//
//
//
// TODO: Check table bounds!
//
//
//

//#include "cctk.h"
#include <array>
#include <bitset>
#include <iostream>
#include "../margherita.hh"
#include "../utils/brent.hh"
#include "../utils/linear_interp_ND.hh"

#ifndef EOS_TABULATED_HH
#define EOS_TABULATED_HH

class EOS_Tabulated {
 public:
  // error definitions
  enum errors {
    NO_ERRORS = 0,
    RHO_TOO_HIGH,
    RHO_TOO_LOW,
    YE_TOO_HIGH,
    YE_TOO_LOW,
    TEMP_TOO_HIGH,
    TEMP_TOO_LOW,
    num_errors
  };
  // enum checked_vars { RHO, YE, TEMP };
  // return type of errors
  typedef std::bitset<errors::num_errors> error_type;

  enum EV {
    PRESS = 0,
    EPS,
    S,
    CS2,
    MUE,
    MUP,
    MUN,
    XA,
    XH,
    XN,
    XP,
    ABAR,
    ZBAR,
    NUM_VARS
  };

  static constexpr bool has_ye = true;

  static const double max_csnd2;  // = 0.99999999;
 private:
  template <bool check_temp>
  static inline error_type checkbounds(double &xrho, double &xtemp,
                                       double &xye);

  static inline double find_logtemp_from_eps(const double &lrho, double &eps,
                                             const double ye);

  static inline double find_logtemp_from_entropy(const double &lrho,
                                                 double &entropy,
                                                 const double ye);

 public:
  static inline int index_pmax(const int &irho, const int &iye);

  // General accessors
  static double press__eps_rho_ye(double &eps, double &rho, double &ye,
                                  error_type &error);
  static double press_temp__eps_rho_ye(double &temp, double &eps, double &rho,
                                       double &ye, error_type &error);
  static double press__temp_rho_ye(double &temp, double &rho, double &ye,
                                   error_type &error);
  static double eps__temp_rho_ye(double &temp, double &rho, double &ye,
                                 error_type &error);
  static double press_cold__rho_ye(double &rho, double &ye, error_type &error);
  static double eps_cold__rho_ye(double &rho, double &ye, error_type &error);
  static double temp_cold__rho_ye(const double &rho, const double &ye,
                                  error_type &error);

  static double eps__press_temp_rho_ye(const double &press, double &temp,
                                       double &rho, double &ye,
                                       error_type &error);

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
                                          const double &press, double &rho,
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
      double &eps, double &h, double &csnd2, double &temp, double &entropy,
      const double &press, double &rho, double &ye, error_type &error);

  static double press_eps_ye__beta_eq__rho_temp(double &eps, double &ye,
                                                double &rho, double &temp,
                                                error_type &error);

  static double mue_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye(
      double &mup, double &mun, double &Xa, double &Xh, double &Xn, double &Xp,
      double &Abar, double &Zbar, double &temp, double &rho, double &ye,
      error_type &error);
  /*
    static double dP_drho_dP_deps_deps_dT__temp_rho_ye(
                  double &dPdeps, double &depsdT,
                  double &temp, double &rho, double &ye, error_type &error);
  */
  // Table specific routines
  // Read in stellar collapse table
  static void readtable_scollapse(const char *nuceos_table_name,
                                  bool do_energy_shift, bool recompute_mu_nu);
  // Read in compose table
  static void readtable_compose(const char *nuceos_table_name);

  // Global variables
  static double temp0, temp1;
  static double energy_shift;

  // min and max values

  static double eos_rhomax, eos_rhomin;
  static double eos_tempmin, eos_tempmax;
  static double eos_yemin, eos_yemax;

  static double baryon_mass;
  // Con2Prim limits
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
  static bool extend_table_high;

  static linear_interp_uniform_ND_t<double, 3, EOS_Tabulated::EV::NUM_VARS> alltables;
};

#endif
