/*
 * =====================================================================================
 *
 *       Filename:  hot_slice.cc
 *
 *    Description:  Setup hot slice
 *
 *        Version:  1.0
 *        Created:  07/05/2017 13:19:55
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Elias Roland Most (ERM), emost@itp.uni-frankfurt.de
 *   Organization:  Goethe University Frankfurt
 *
 * =====================================================================================
 */

#include "hot_slice.hh"
#include "../Margherita_EOS.h"
#include "lorene_io.hh"

#ifndef STANDALONE

#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"

#include <cassert>
#include <iostream>

extern "C" void Margherita_setup_hot_slice(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  if (!slice_is_isentropic) {
    HotSlice_beta_eq(downsample_num_points, cold_table_num_points, temp_slice);
  } else {
    HotSlice_beta_eq_isentropic(downsample_num_points, cold_table_num_points,
                                entropy_slice);
  }

  if (CCTK_MyProc(cctkGH) == 0) {
    std::ofstream file{"./eos.margherita"};
    assert(file.good());
    write_table<>(file);
    file.close();
  }

  return;
}
#endif
