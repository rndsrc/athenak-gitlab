//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_spharms.cpp
//! \brief test the spherical harmonics basis functions for fpn

// C++ headers
#include <iostream>
#include <sstream>
#include <cmath>        // exp
#include <algorithm>    // max
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf.h>

// AthenaK headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid.hpp"
#include "adm/adm.hpp"

void ProblemGenerator::RadiationFEMNSpharms(ParameterInput *pin, const bool restart) {
  if (restart) return;

  std::cout << "----------------------------- " << std::endl;
  std::cout << "Tests for FPN basis functions " << std::endl;
  std::cout << "----------------------------- " << std::endl;

// ----------
  std::cout << "Test: FPN basis new vs old implementation" << std::endl;
  Real error = -42;
  Real phi = -3. * M_PI / 2. - 0.2;
  Real theta = 1.9889;

  for (int l = 0; l < 25; l++) {
    for (int m = -l; m <= l; m++) {
      Real fpn_new_basis = radiationfemn::fpn_basis_lm(l, m, phi, theta);
      Real fpn_basis_gsl = radiationfemn::fpn_basis_lm_alt(l, m, phi, theta);

      if (fabs(fpn_new_basis - fpn_basis_gsl) > error) {
        error = fabs(fpn_new_basis - fpn_basis_gsl);
      }

      std::cout << "l = " << l << ", m = " << m << ": " << fpn_new_basis << " " << fpn_basis_gsl << std::endl;
    }
  }
  std::cout << "Absolute error: " << error << std::endl;
  std::cout << std::endl;

  // ------------
  std::cout << "Test: Check associated Legendre implementation for l >= 0, |m| <= l with scipy implementation" << std::endl;
  Real error2 = -42;
  Real x = 0.8987;
  int N = 30;
  Real l_arr[] = {11, 4, 4, 26, 0, 4, 22, 13, 8, 23, 23, 10, 29, 20, 27, 19, 20, 11, 2, 18, 28, 11, 27, 24, 26, 14, 9, 26, 3, 26, 23, 27, 23, 14, 15, 27, 21, 2, 14, 1, 0, 26, 23,
                  2, 26, 6, 20, 20, 0, 3};
  Real m_arr[] = {7, 0, -1, -12, 0, 0, -22, -7, 7, 0, 11, 10, 20, 9, 18, 4, -15, 8, 1, 0, -8, 10, -17, -5, 8, -8, -6, 12, -3, 15, -14, -20, -13, -10, 10, 1, -4, 0, 6, 0, 0, -4,
                  -13, 1, 21, -2, -1, 12, 0, -1};
  Real answers[] = {-767068.0698190533, 0.20015731153474592, 0.13073694140932204, 2.1636171267985502e-18, 1.0, 0.20015731153474592, 2.8268141479831695e-36, 2.0019898541051555e-09,
                    -5684.655251828969, -0.22388166933330705, -113469147045446.45, 172341.7926871465, 1.4055020950276696e+25, -102558690792.88368, 2.132776678486919e+22,
                    -35413.397549732734, 5.412508224579258e-23, 1873706.56176514, -1.1824121787926132, 0.07035318127627838, -2.661385439814381e-13, 3252554.95084671,
                    4.406176734001765e-27, -2.7919927986855215e-08, 16705573721.962795, 5.648473516213318e-11, 1.0636075512287724e-07, 1.2980533080851954e+16,
                    0.0017573467245310269, -1.650902887446956e+19, 1.918964221063777e-21, 1.1369855638015587e-32, 9.846651817720011e-20, 4.3222553164519686e-14, 4795489134.8200865,
                    4.7053765235150085, -1.2419559894218114e-06, 0.7114925350000003, 2281588.7055624607, 0.8987, 1.0, 3.7804300687725965e-07, 9.846651817720011e-20,
                    -1.1824121787926132, -3.3908185064157403e+23, 0.012003960090949603, 0.009912456468071734, 54434511824144.58, 1.0, 0.16656155716630072};

  for (int i = 0; i < N; i++) {
    Real legval = radiationfemn::legendre(l_arr[i], m_arr[i], x);
    Real relerror = fabs(answers[i]) > 1e-14 ? fabs((legval - answers[i]) / answers[i]): fabs((legval - answers[i]));
    if (relerror > error2) {
      error2 = relerror;
    }
    std::cout << "l = " << l_arr[i] << ", m = " << m_arr[i] << ": " << legval << " " << answers[i] << std::endl;
  }
  std::cout << "Relative error: " << error2 << std::endl;
  std::cout << std::endl;

  // -------------
  std::cout << "Test: Check associated Legendre implementation for |m| > l (should be zero)" << std::endl;
  Real x1 = 0.23;
  Real error3 = -42;
  for (int m = 20; m < 35; m++) {
    for (int l = -(m); l <= (m - 1); l++) {
      Real legval = radiationfemn::legendre(l, m, x);
      if (fabs(legval) > error3) {
        error3 = fabs(legval);
      }
      std::cout << "l = " << l << ", m = " << m << ": " << legval << " " << "0" << std::endl;
    }
  }
  std::cout << "Absolute error: " << error3 << std::endl;
  std::cout << std::endl;

  // -----------
  std::cout << "Test: Check recurrence relations for associated Legendre functions" << std::endl;
  Real x2 = 0.9999;
  Real lwhere = -42;
  Real mwhere = -42;
  Real error4 = -42;
  for (int l = 0; l < 20; l++) {
    for (int m = -l; m <= l; m++) {
      Real legval = m * radiationfemn::legendre(l, m, x2) / sqrt(1 - x2 * x2);
      Real recur_legval1 = radiationfemn::recurrence_legendre(l, m, x2);
      Real recur_legval2 = radiationfemn::recurrence_legendre_alt(l, m, x2);
      Real rel_error = fabs(legval) > 1e-14 ? fabs(legval - recur_legval1) / fabs(recur_legval1) : fabs(legval - recur_legval1);
      Real rel_error2 = fabs(legval) > 1e-14 ? fabs(legval - recur_legval2) / fabs(recur_legval2) : fabs(legval - recur_legval2);
      if (rel_error > error4) {
        error4 = rel_error;
        lwhere = l;
        mwhere = m;
      }
      if (rel_error2 > error4) {
        error4 = rel_error2;
        lwhere = l;
        mwhere = m;
      }
      std::cout << "l = " << l << ", m = " << m << ": " << legval << " " << recur_legval1 << " " << recur_legval2 << std::endl;
    }
  }
  std::cout << "Relative error: (" << lwhere << ", " << mwhere << ") " << error4 << std::endl;
  std::cout << std::endl;

  // ------------
  std::cout << "Test: Check recurrence relation for derivatives of associated Legendre functions with scipy implementation" << std::endl;
  Real error5 = -42;
  x = 0.9543;
  N = 50;
  Real l_arr_der[] = {3, 21, 4, 4, 1, 6, 30, 1, 1, 5, 14, 3, 12, 1, 22, 2, 13, 16, 9, 7, 6, 22, 30, 1, 16, 5, 9, 15, 20, 6, 7, 13, 27, 26, 24, 19, 22, 29, 30, 12, 7, 2, 25, 19, 9,
                      9, 24, 15, 2, 15};
  Real m_arr_der[] = {3, -17, 4, -2, 1, 2, 10, 0, 1, -2, 14, -2, 12, 0, 0, 0, -8, 5, 8, 6, -5, -9, -21, 0, -13, 3, 1, 9, 20, -1, 4, 13, 24, -26, -19, -14, 16, -9, 13, -7, -4, -2,
                      3, -16, -7, 1, -8, -9, 1, -7};
  Real answers_der[] = {12.833678446477364, -3.7485313358964635e-27, -35.79658907705998, -0.18885751517075017, 3.1932374042960863, -205.8081964694436, -4575541402899365.0, 1.0,
                        3.1932374042960863, -0.15700256295482534, -1447338394.3152037, -0.2165081837500001, -20578422.92138207, 1.0, -4.331736109744453, 2.8629000000000007,
                        -3.641462521813354e-10, 8209233.363531861, -176658.00880122583, -5793.574402291702, -3.10289334851494e-05, -2.658855619591872e-12, -1.1604338894053653e-35,
                        1.0, -3.4647872930601917e-19, 299.16774845362556, -105.39268087143103, 5440353368.61422, -2207000067008977.2, 0.1570454386532748, -4532.723627758876,
                        166558928.94211188, -1.399080982140365e+24, -2.3612857317855684e-46, -2.5794445596378183e-31, -3.488453514888697e-21, -2.5367209978784794e+17,
                        -6.118025142041337e-13, 1.4769455048587022e+19, -1.669350482698397e-08, -0.0006813257016232078, -0.2385750000000001, -104770.0248203713,
                        -4.263312189412571e-25, -2.1672181765526562e-08, -105.39268087143103, -6.832905219781743e-11, -6.313263785449105e-12, 8.24536842365218,
                        -1.1752104590970943e-08};
  for (int i = 0; i < N; i++) {
    Real legval_der = radiationfemn::recurrence_derivative_legendre(l_arr_der[i], m_arr_der[i], x);
    Real answers_der_final = sqrt(1 - x*x) * answers_der[i];
    Real erval = fabs(answers_der_final) > 1e-14 ? fabs(legval_der - answers_der_final)/ fabs(answers_der_final) : fabs(legval_der - answers_der_final);
    if (erval > error5) {
      error5 = erval;
    }
    std::cout << "l = " << l_arr_der[i] << ", m = " << m_arr_der[i] << ": " << legval_der << " " << answers_der_final << std::endl;
  }
  std::cout << "Relative error: " << error5 << std::endl;
  std::cout << std::endl;

  std::cout << "--------" << std::endl;
  std::cout << "Results:" << std::endl;
  std::cout << "Test 1: " << error  << " (abs error)" <<std::endl;
  std::cout << "Test 2: " << error2 << " (rel error)" <<std::endl;
  std::cout << "Test 3: " << error3 << " (abs error)" <<std::endl;
  std::cout << "Test 4: " << error4 << " (rel error)" <<std::endl;
  std::cout << "Test 5: " << error5 << " (rel error)" <<std::endl;
  std::cout << "--------" << std::endl;
}