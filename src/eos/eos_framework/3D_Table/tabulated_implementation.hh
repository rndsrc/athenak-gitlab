//
//  Copyright (C) 2017, Elias Roland Most
//  			<emost@th.physik.uni-frankfurt.de>
//
//  This code was originally released under LGPLv2 and is
//  rereleased here under GPLv2 in accordance with LGPLv2. Please refer
//  to stellarcollapse.org and the einsteintoolkit.org for more information.
//
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

#include <array>
#include <bitset>
#include <functional>
#include <iostream>

#include "../utils/brent.hh"
#include "../utils/spline.hh"
#include "tabulated.hh"

#ifndef EOS_TABULATED_IMPL_HH
#define EOS_TABULATED_IMPL_HH

// Initialize storage for static members
double EOS_Tabulated::temp0;
double EOS_Tabulated::temp1;
double EOS_Tabulated::energy_shift;

double EOS_Tabulated::eos_rhomax;
double EOS_Tabulated::eos_rhomin;
double EOS_Tabulated::eos_tempmin;
double EOS_Tabulated::eos_tempmax;
double EOS_Tabulated::eos_yemin;
double EOS_Tabulated::eos_yemax;

double EOS_Tabulated::c2p_ye_atm;
double EOS_Tabulated::c2p_rho_atm;
double EOS_Tabulated::c2p_temp_atm;
double EOS_Tabulated::c2p_eps_atm;
double EOS_Tabulated::c2p_eps_min;
double EOS_Tabulated::c2p_eps_max;
double EOS_Tabulated::c2p_press_max;
double EOS_Tabulated::c2p_h_min;
double EOS_Tabulated::c2p_h_max;
bool EOS_Tabulated::atm_beta_eq = true;
bool EOS_Tabulated::extend_table_high = false;

const double EOS_Tabulated::max_csnd2 = 0.99999999;

double EOS_Tabulated::baryon_mass = Margherita_constants::mnuc_cgs;

linear_interp_uniform_ND_t<double, 3, EOS_Tabulated::EV::NUM_VARS>
    EOS_Tabulated::alltables;

// ###################################################################################
// 	Private functions
// ###################################################################################

template <bool check_temp>
EOS_Tabulated::error_type EOS_Tabulated::checkbounds(double &xrho,
                                                     double &xtemp,
                                                     double &xye) {
  // initialize error codes to zero
  error_type error_codes;

  if (xrho > eos_rhomax && !extend_table_high) {
    // This happens only inside the AH.
    // (assuming that the table has a sane range)
    error_codes[errors::RHO_TOO_HIGH] = true;
    xrho = eos_rhomax;
  } else if (xrho < eos_rhomin) {
    // Might happen in the atmosphere and will be caught later. This point is
    // most likely also incorrect and should be reset.
    xrho = eos_rhomin;
    error_codes[errors::RHO_TOO_LOW] = true;
  }
  if (xye > eos_yemax) {
    // This is a serious issue and should not happen.
    // Maybe we should even abort the simulation here...
    //    fprintf(stderr, "This is Margherita, your favorite EOS handler.\n Ye "
    //                    "exceeds the table maximum! So prepare to abort and
    //                    debug "
    //                    "your simulations.");
    // abort();
    xye = eos_yemax;
    error_codes[errors::YE_TOO_HIGH] = true;
  } else if (xye < eos_yemin) {
    // Ok, this can be fixed
    xye = eos_yemin;
    error_codes[errors::YE_TOO_LOW] = true;
  }
  if (check_temp) {
    if (xtemp > eos_tempmax) {
      // This should never happen...
      // But removing energy should be fine, so what about
      xtemp = eos_tempmax;
      error_codes[errors::TEMP_TOO_HIGH] = true;
    } else if (xtemp < eos_tempmin) {
      // Might happen in the atmosphere, let's reset
      xtemp = eos_tempmin;
      error_codes[errors::TEMP_TOO_LOW] = true;
    }
  }
  return error_codes;
  // Overall only abort on too high YE.
}

inline double EOS_Tabulated::find_logtemp_from_eps(const double &lrho,
                                                   double &eps,
                                                   const double ye) {
  auto const leps = log(eps + energy_shift);

  // Get eps ranges
  auto const varsmin =
      alltables.interpolate<EV::EPS>(lrho, alltables.xmin<1>(), ye);
  auto const varsmax =
      alltables.interpolate<EV::EPS>(lrho, alltables.xmax<1>(), ye);

  if (leps <= varsmin[0]) {
    eps = exp(varsmin[0]) - energy_shift;
    return alltables.xmin<1>();
  }
  if (leps >= varsmax[0]) {
    eps = exp(varsmax[0]) - energy_shift;
    return alltables.xmax<1>();
  }

  // Root finding interface closure
  auto const func = [&](double &lt) {
    auto const vars = alltables.interpolate<EV::EPS>(lrho, lt, ye);
    return leps - vars[0];
  };

  return zero_brent(alltables.xmin<1>(), alltables.xmax<1>(), 1.e-14, func);
}

inline double EOS_Tabulated::find_logtemp_from_entropy(const double &lrho,
                                                       double &entropy,
                                                       const double ye) {
  // Get eps ranges
  auto const varsmin =
      alltables.interpolate<EV::S>(lrho, alltables.xmin<1>(), ye);
  auto const varsmax =
      alltables.interpolate<EV::S>(lrho, alltables.xmax<1>(), ye);

  if (entropy <= varsmin[0]) {
    entropy = varsmin[0];
    return alltables.xmin<1>();
  }
  if (entropy >= varsmax[0]) {
    entropy = varsmax[0];
    return alltables.xmax<1>();
  }

  // Root finding interface closure
  auto const func = [&](double &lt) {
    auto const vars = alltables.interpolate<EV::S>(lrho, lt, ye);
    return entropy - vars[0];
  };

  return zero_brent(alltables.xmin<1>(), alltables.xmax<1>(), 1.e-14, func);
}

// ###################################################################################
// 	Public functions
// ###################################################################################
double EOS_Tabulated::press__eps_rho_ye(double &eps, double &rho, double &ye,
                                        error_type &error) {
  // Check if rho and Y_e lie inside the table, otherwise abort!
  double temp_tmp = 0;
  error = checkbounds<false>(rho, temp_tmp, ye);

  const double lrho = log(rho);

  const double ltemp = find_logtemp_from_eps(lrho, eps, ye);

  double press;

  auto const vars = alltables.interpolate<EV::PRESS>(lrho, ltemp, ye);

  return exp(vars[0]);
}

double EOS_Tabulated::press_temp__eps_rho_ye(double &temp, double &eps,
                                             double &rho, double &ye,
                                             error_type &error) {
  // Check if rho and Y_e lie inside the table, otherwise abort!
  error = checkbounds<true>(rho, temp, ye);

  const double lrho = log(rho);
  const double ltemp = find_logtemp_from_eps(lrho, eps, ye);

  temp = exp(ltemp);

  auto const vars = alltables.interpolate<EV::PRESS>(lrho, ltemp, ye);

  return exp(vars[0]);
}

double EOS_Tabulated::press__temp_rho_ye(double &temp, double &rho, double &ye,
                                         error_type &error) {
  error = checkbounds<true>(rho, temp, ye);

  const auto lrho = log(rho);
  const auto ltemp = log(temp);

  auto const vars = alltables.interpolate<EV::PRESS>(lrho, ltemp, ye);

  return exp(vars[0]);
}

double EOS_Tabulated::eps__temp_rho_ye(double &temp, double &rho, double &ye,
                                       error_type &error) {
  error = checkbounds<true>(rho, temp, ye);

  const double lrho = log(rho);
  const double ltemp = log(temp);

  auto const vars = alltables.interpolate<EV::EPS>(lrho, ltemp, ye);

  return exp(vars[0]) - energy_shift;
}

//////////

double EOS_Tabulated::press_cold__rho_ye(double &rho, double &ye,
                                         error_type &error) {
  double temp_tmp = 0;
  error = checkbounds<false>(rho, temp_tmp, ye);
  const double lrho = log(rho);
  const double ltemp = log(temp0);  // FIXME Check!

  auto const vars = alltables.interpolate<EV::PRESS>(lrho, ltemp, ye);

  return exp(vars[0]);
}

double EOS_Tabulated::eps_cold__rho_ye(double &rho, double &ye,
                                       error_type &error) {
  double temp_tmp = 0;
  error = checkbounds<false>(rho, temp_tmp, ye);

  const double lrho = log(rho);
  const double ltemp = log(temp0);

  auto const vars = alltables.interpolate<EV::EPS>(lrho, ltemp, ye);

  return exp(vars[0]) - energy_shift;
}

double EOS_Tabulated::temp_cold__rho_ye(const double &rho, const double &ye,
                                        error_type &error) {
  //  for (auto i = 0; i < error.size(); ++i)
  //    error[i] = errors::NO_ERRORS;
  return temp0;
}
///////////

double EOS_Tabulated::eps__press_temp_rho_ye(const double &press, double &temp,
                                             double &rho, double &ye,
                                             error_type &error) {
  assert(!"This routine should not be used. There is no monotonicity condition "
          "to enforce a succesfull inversion from eps(press). So you better "
          "rewrite your code to not require this call...");

  return 0;
}

std::array<double, 2> EOS_Tabulated::eps_range__rho_ye(double &rho, double &ye,
                                                       error_type &error) {
  double temp_tmp = 0;
  error = checkbounds<false>(rho, temp_tmp, ye);

  const double lrho = log(rho);

  auto const varsmin =
      alltables.interpolate<EV::EPS>(lrho, alltables.xmin<1>(), ye);
  auto const varsmax =
      alltables.interpolate<EV::EPS>(lrho, alltables.xmax<1>(), ye);

  return std::array<double, 2>{exp(varsmin[0]) - energy_shift,
                               exp(varsmax[0]) - energy_shift};
}

std::array<double, 2> EOS_Tabulated::entropy_range__rho_ye(double &rho,
                                                           double &ye,
                                                           error_type &error) {
  double temp_tmp = 0;
  error = checkbounds<false>(rho, temp_tmp, ye);

  const double lrho = log(rho);

  auto const varsmin =
      alltables.interpolate<EV::S>(lrho, alltables.xmin<1>(), ye);
  auto const varsmax =
      alltables.interpolate<EV::S>(lrho, alltables.xmax<1>(), ye);

  return std::array<double, 2>{varsmin[0], varsmax[0]};
}
/////////////////

double EOS_Tabulated::press_h_csnd2__eps_rho_ye(double &h, double &csnd2,
                                                double &eps, double &rho,
                                                double &ye, error_type &error) {
  using namespace Margherita_helpers;

  double temp_tmp = 0;
  error = checkbounds<false>(rho, temp_tmp, ye);

  const auto lrho = log(rho);
  const auto ltemp = find_logtemp_from_eps(lrho, eps, ye);

  auto const vars = alltables.interpolate<EV::PRESS, EV::CS2>(lrho, ltemp, ye);

  const auto press = exp(vars[0]);

  h = 1. + eps + press / rho;

  csnd2 = vars[1];  // min(vars[1] / h, max_csnd2);

  return press;
};

double EOS_Tabulated::press_h_csnd2__temp_rho_ye(double &h, double &csnd2,
                                                 double &temp, double &rho,
                                                 double &ye,
                                                 error_type &error) {
  using namespace Margherita_helpers;

  error = checkbounds<true>(rho, temp, ye);

  const double ltemp = log(temp);
  const double lrho = log(rho);

  auto const vars =
      alltables.interpolate<EV::PRESS, EV::CS2, EV::EPS>(lrho, ltemp, ye);

  const auto press = exp(vars[0]);
  const double eps = exp(vars[2]) - energy_shift;

  h = 1. + eps + press / rho;

  csnd2 = vars[1];  // min(vars[1] / h, max_csnd2);

  return press;
}

double EOS_Tabulated::eps_h_csnd2__press_rho_ye(double &h, double &csnd2,
                                                const double &press,
                                                double &rho, double &ye,
                                                error_type &error) {
  using namespace Margherita_helpers;

  assert(!"This routine should not be used. There is no monotonicity condition "
          "to enforce a succesfull inversion from eps(press). So you better "
          "rewrite your code to not require this call...");

  return 0;
}

double EOS_Tabulated::press_eps_csnd2__temp_rho_ye(double &eps, double &csnd2,
                                                 double &temp, double &rho,
                                                 double &ye,
                                                 error_type &error) {
  using namespace Margherita_helpers;

  error = checkbounds<true>(rho, temp, ye);

  const double ltemp = log(temp);
  const double lrho = log(rho);

  auto const vars =
      alltables.interpolate<EV::PRESS, EV::CS2, EV::EPS>(lrho, ltemp, ye);

  const auto press = exp(vars[0]);
  eps = exp(vars[2]) - energy_shift;


  csnd2 = vars[1];  

  return press;
}

double EOS_Tabulated::press_h_csnd2_temp_entropy__eps_rho_ye(
    double &h, double &csnd2, double &temp, double &entropy, double &eps,
    double &rho, double &ye, error_type &error) {
  using namespace Margherita_helpers;

  error = checkbounds<true>(rho, temp, ye);

  const double lrho = log(rho);
  const auto ltemp = find_logtemp_from_eps(lrho, eps, ye);

  auto const vars =
      alltables.interpolate<EV::PRESS, EV::CS2, EV::S>(lrho, ltemp, ye);

  const auto press = exp(vars[0]);
  temp = exp(ltemp);
  entropy = vars[2];

  h = 1. + eps + press / rho;

  csnd2 = vars[1];  // min(vars[1] / h, max_csnd2);

  return press;
}

double EOS_Tabulated::eps_csnd2_entropy__temp_rho_ye(double &csnd2,
                                                     double &entropy,
                                                     double &temp, double &rho,
                                                     double &ye,
                                                     error_type &error) {
  using namespace Margherita_helpers;

  error = checkbounds<true>(rho, temp, ye);

  const double lrho = log(rho);
  const double ltemp = log(temp);

  auto const vars = alltables.interpolate<EV::PRESS, EV::CS2, EV::S, EV::EPS>(
      lrho, ltemp, ye);

  const auto eps = exp(vars[3]) - energy_shift;

  const double h = 1. + eps + exp(vars[0]) / rho;

  csnd2 = vars[1];  // min(vars[1] / h, max_csnd2);

  entropy = vars[2];

  return eps;
}

double EOS_Tabulated::press_h_csnd2_temp_eps__entropy_rho_ye(
    double &h, double &csnd2, double &temp, double &eps, double &entropy,
    double &rho, double &ye, error_type &error) {
  using namespace Margherita_helpers;

  error = checkbounds<true>(rho, temp, ye);

  const auto lrho = log(rho);
  const auto ltemp = find_logtemp_from_entropy(lrho, entropy, ye);

  auto const vars =
      alltables.interpolate<EV::PRESS, EV::CS2, EV::EPS>(lrho, ltemp, ye);

  const auto press = exp(vars[0]);
  eps = exp(vars[2]) - energy_shift;
  temp = exp(ltemp);

  h = 1. + eps + press / rho;

  csnd2 = vars[1];  // min(vars[1] / h, max_csnd2);

  return press;
}

double EOS_Tabulated::eps_h_csnd2_temp_entropy__press_rho_ye(
    double &eps, double &h, double &csnd2, double &temp, double &entropy,
    const double &press, double &rho, double &ye, error_type &error) {
  using namespace Margherita_helpers;

  assert(!"This routine should not be used. There is no monotonicity condition "
          "to enforce a succesfull inversion from eps(press). So you better "
          "rewrite your code to not require this call...");

  return 0;
}

double EOS_Tabulated::press_eps_ye__beta_eq__rho_temp(double &eps, double &ye,
                                                      double &rho, double &temp,
                                                      error_type &error) {
  // Check if rho and Y_e lie inside the table, otherwise abort!
  error = checkbounds<true>(rho, temp, ye);

  const double lrho = log(rho);
  const double ltemp = log(temp);

  // Beta equilibrium requires that \mu_n =  \mu_p +\mu_e -\mu_nu
  // Since we assume that \mu_nu should be negligible we demand
  // \mu_n-\mu_p-\mu_e = \mu_hat =0

  // Root finding interface closure
  auto const func = [&](double &ye) {
    auto const vars =
        alltables.interpolate<EV::MUE, EV::MUP, EV::MUN>(lrho, ltemp, ye);
    return vars[0] + vars[1] - vars[2];
  };

  ye = zero_brent(alltables.xmin<2>(), alltables.xmax<2>(), 1.e-14, func);

  auto const vars = alltables.interpolate<EV::PRESS, EV::EPS>(lrho, ltemp, ye);

  eps = exp(vars[1]) - energy_shift;
  return exp(vars[0]);
}

double EOS_Tabulated::mue_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye(
    double &mup, double &mun, double &Xa, double &Xh, double &Xn, double &Xp,
    double &Abar, double &Zbar, double &temp, double &rho, double &ye,
    error_type &error) {
  error = checkbounds<true>(rho, temp, ye);

  const double lrho = log(rho);
  const double ltemp = log(temp);

  auto const vars =
      alltables.interpolate<EV::MUE, EV::MUP, EV::MUN, EV::XA, EV::XH, EV::XN,
                            EV::XP, EV::ABAR, EV::ZBAR>(lrho, ltemp, ye);

  mup = vars[1];
  mun = vars[2];

  Xa = vars[3];
  Xh = vars[4];
  Xn = vars[5];

  Xp = vars[6];
  Abar = vars[7];
  Zbar = vars[8];

  return vars[0];  // mue
};

  /*
  double EOS_Tabulated::dP_drho_dP_deps_deps_dT__temp_rho_ye(
                  double &dPdeps, double &depsdT,
                  double &temp, double &rho, double &ye, error_type &error){

    error = checkbounds<true>(rho, temp, ye);

    const double lrho = log(rho);
    const double ltemp = log(temp);

    auto const vars =
  alltables.interpolate<EV::DPDRHOE,EV::DPDERHO,EV::DEDT>(lrho,ltemp, ye);

    dPdeps = vars[1];
    depsdT = vars[2];

    return vars[0]; //dPdrho at constant eps

  }
  */

#include "tabulated_readtable_compose.hh"
#include "tabulated_readtable_scollapse.hh"

#endif
