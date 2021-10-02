/*
 * =====================================================================================
 *
 *       Filename:  lorene_io.cpp
 *
 *    Description:  Read Lorene Table
 *
 *        Version:  1.0
 *        Created:  01/05/2017 23:21:18
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Elias Roland Most (ERM), most@fias.uni-frankfurt.de
 *   Organization:  Goethe University Frankfurt
 *
 * =====================================================================================
 */

#include <array>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../margherita.hh"

namespace MC = Margherita_constants;

static const std::array<std::string, Hot_Slice::v_index::NUM_VARS + 1>
    var_names{{"n_B [fm^{-3}]", "e [g/cm^3]", "p [dyn/cm^2]", "Y_e", "T [MeV]",
               "s [kB/m_B]", "c_s^2 [c]"}};

static const std::array<double, Hot_Slice::v_index::NUM_VARS + 1> conv{
    {MC::RHOGF * MC::mnuc_cgs * MC::cm3_to_fm3, MC::RHOGF, MC::PRESSGF, 1., 1.,
     1., 1.}};

static std::array<std::vector<double>, 3> Lorene_Table(
    const std::string &filename) {
  using namespace Margherita_constants;

  std::ifstream file(filename);
  // std::cout << std::setiosflags(std::ios::scientific) <<
  // std::setprecision(16);

  // Create vectors
  std::array<std::vector<double>, 3> vectors;

  // Skip lines
  constexpr double skip_lines = 9; //8;
  std::string line;
  for (int i = 0; i < skip_lines; i++) {
    std::getline(file, line);
  }

  while (std::getline(file, line)) {
    double n, e, p, dummy;
    file >> dummy >> n >> e >> p;
    vectors[0].push_back(n * RHOGF * mnuc_cgs * cm3_to_fm3);   // rho
    vectors[1].push_back(e / n / mnuc_cgs / cm3_to_fm3 - 1.);  // eps
    vectors[2].push_back(p * PRESSGF);                         // press
  }
  // Last entry is eof duplication, so remove
  vectors[0].pop_back();
  vectors[1].pop_back();
  vectors[2].pop_back();

  assert(vectors[0].size() == vectors[1].size());
  assert(vectors[0].size() == vectors[2].size());

  return vectors;
}

static std::array<std::vector<double>, 4> ID_Table(
    const std::string &filename) {
  using namespace Margherita_constants;

  std::ifstream file(filename);
  // std::cout << std::setiosflags(std::ios::scientific) <<
  // std::setprecision(16);

  // Create vectors
  std::array<std::vector<double>, 4> vectors;

  // Skip lines
  constexpr double skip_lines = 8;
  std::string line;
  for (int i = 0; i < skip_lines; i++) {
    std::getline(file, line);
  }

  while (std::getline(file, line)) {
    double ye, T, s, csnd2, dummy;
    file >> dummy >> ye >> T >> s >> csnd2;
    vectors[0].push_back(ye);
    vectors[1].push_back(T);
    vectors[2].push_back(s);
    vectors[3].push_back(csnd2);
  }
  // Last entry is eof duplication, so remove
  vectors[0].pop_back();
  vectors[1].pop_back();
  vectors[2].pop_back();
  vectors[3].pop_back();

  assert(vectors[0].size() == vectors[1].size());
  assert(vectors[0].size() == vectors[2].size());
  assert(vectors[0].size() == vectors[3].size());

  return vectors;
}

template <int begin = 0, int end = Hot_Slice::v_index::NUM_VARS>
static void write_table(std::ostream &file) {
  std::string whitespace{"    "};

  // Print header
  file << "#" << std::endl;
  file << "#" << std::endl;
  file << "#" << std::endl;
  file << "#" << std::endl;
  file << "#" << std::endl;
  file << Hot_Slice::lintp.size() << std::endl;
  file << "#" << std::endl;
  file << "# index";

  for (int i = begin; i <= end; ++i) file << whitespace << var_names[i];

  file << std::endl;
  file << "#" << std::endl;

  file << std::setiosflags(std::ios::scientific) << std::setprecision(16);

  /*
   auto rhoL = exp(Hot_Slice::lintp[0]);
   typename Hot_Slice::error_t error;
   auto interp = Hot_Slice::get_extra_quantities(rhoL,error);
   const auto press_0 = exp(interp[Hot_Slice::v_index::PRESS]);
  */
  // Write table
  for (int nn = 0; nn < Hot_Slice::lintp.size(); ++nn) {
    auto rhoL = exp(Hot_Slice::lintp[nn]);
    typename Hot_Slice::error_t error;
    auto interp = Hot_Slice::get_extra_quantities(rhoL, error);

    // We don't output eps but rho(1+eps), so need to modify
    interp[Hot_Slice::v_index::EPS] =
        rhoL * (1. + exp(interp[Hot_Slice::v_index::EPS]));
    interp[Hot_Slice::v_index::TEMP] = exp(interp[Hot_Slice::v_index::TEMP]);
    interp[Hot_Slice::v_index::PRESS] = exp(interp[Hot_Slice::v_index::PRESS]);

    file << 42;
    if (begin == 0) file << whitespace << rhoL / conv[0];

    for (int i = std::max(1, begin); i <= end; ++i) {
      file << whitespace << interp[i - 1] / conv[i];
    }
    file << std::endl;
  }
}
