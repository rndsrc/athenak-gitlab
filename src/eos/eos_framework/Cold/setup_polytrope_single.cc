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

/********************************
 * Setup piecewise polytropes
 *
 * Written in June 2016 by Elias R. Most
 * <emost@th.physik.uni-frankfurt.de>
 *
 * Sets up parameters for piecewiese
 * polytropes in IllinoisGRMHD.
 *
 * We do things here as in Whisky_Exp
 * (See Takami et al. https://arxiv.org/pdf/1412.3240v2.pdf)
 * (also see Read et al. https://arxiv.org/pdf/0812.2163v1.pdf)
 ********************************/

#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"

#define PWPOLY_SETUP

#include "../Hybrid/hybrid.hh"
#include "../Hybrid/hybrid_implementation.hh"
#include "cold_pwpoly.hh"
#include "cold_pwpoly_implementation.hh"

#include "../Margherita_EOS.h"

static constexpr double c_sq = 8.9875517873681764e+20;
static constexpr double rho_nuc = 6.1771669179018705e+17;
static constexpr double press_nuc = 5.5517607573860526e+38;

void Margherita_setup_polytrope_single(
    double K_EOS,
    std::vector<double> Rho_EOS,
    std::vector<double> Gamma_EOS,
    double gamma_th,
    double entropy_min,
    double pw_rhomin,
    double pw_rhomax,
    std::string Units
    ) 
{

  // First create Arrays..

  int const num_pieces = Gamma_EOS.size();

  Cold_PWPoly::num_pieces = num_pieces;
  assert(num_pieces <= Cold_PWPoly::max_num_pieces);
  EOS_Polytropic::gamma_th_m1 = (gamma_th - 1.);
  EOS_Polytropic::entropy_min = (entropy_min);

  Cold_PWPoly::rhomin = pw_rhomin;
  Cold_PWPoly::rhomax = pw_rhomax;

  // Unit system
  double rho_unit = 1.0;
  double K_unit = 1.0;

  if (Units== "geometrised") {
    if (Units == "cgs") {
      rho_unit = 1.0 / rho_nuc;
      K_unit = pow(rho_nuc, Gamma_EOS[0] - 1.0) / c_sq;
    } else {
      if (Units == "cgs_cgs_over_c2") {
        rho_unit = 1.0 / rho_nuc;
        K_unit = pow(rho_nuc, Gamma_EOS[0] - 1.0);
      } else {
	std::err << "Unit system not recognised!" << std::endl;
      }
    }
  }

  // CCTK_VInfo(CCTK_THORNSTRING, "K_unit=%.15e  ", K_unit);
//  CCTK_VInfo(CCTK_THORNSTRING, "Setting up piecewise polytrope");

  Cold_PWPoly::k_tab[0] = K_unit * K_EOS;
  Cold_PWPoly::gamma_tab[0] = Gamma_EOS[0];
  Cold_PWPoly::rho_tab[0] = 0.0;
  Cold_PWPoly::eps_tab[0] = 0.0;
  Cold_PWPoly::P_tab[0] = 0.0;

  // Setup piecewise polytrope
  for (int i = 1; i < num_pieces; ++i) {
    // Consistency check
    if (Rho_EOS[i] <= Rho_EOS[i - 1])
      std::err << "Rho_EOS must increase monotonically!" <<std::endl;

    Cold_PWPoly::gamma_tab[i] = Gamma_EOS[i];
    Cold_PWPoly::rho_tab[i] = rho_unit * Rho_EOS[i];
    Cold_PWPoly::k_tab[i] =
        Cold_PWPoly::k_tab[i - 1] *
        pow(Cold_PWPoly::rho_tab[i],
            Cold_PWPoly::gamma_tab[i - 1] - Cold_PWPoly::gamma_tab[i]);

    Cold_PWPoly::eps_tab[i] =
        Cold_PWPoly::eps_tab[i - 1] +
        Cold_PWPoly::k_tab[i - 1] *
            pow(Cold_PWPoly::rho_tab[i], Cold_PWPoly::gamma_tab[i - 1] - 1.0) /
            (Cold_PWPoly::gamma_tab[i - 1] - 1.0) -
        Cold_PWPoly::k_tab[i] *
            pow(Cold_PWPoly::rho_tab[i], Cold_PWPoly::gamma_tab[i] - 1.0) /
            (Cold_PWPoly::gamma_tab[i] - 1.0);
    Cold_PWPoly::P_tab[i] =
        Cold_PWPoly::k_tab[i] *
        pow(Cold_PWPoly::rho_tab[i], Cold_PWPoly::gamma_tab[i]);
  }

  // Output polytrope
  for (int nn = 0; nn < num_pieces; ++nn) {
    	printf(
        "nn= %d : K=%.15e , rho= %.15e, gamma= %.15e, eps= %.15e, P=.%15e ", nn,
        Cold_PWPoly::k_tab[nn], Cold_PWPoly::rho_tab[nn],
        Cold_PWPoly::gamma_tab[nn], Cold_PWPoly::eps_tab[nn],
        Cold_PWPoly::P_tab[nn]);
  }

  return;
}
