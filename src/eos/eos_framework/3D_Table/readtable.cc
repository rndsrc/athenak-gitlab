
//
//  Copyright (C) 2017, Elias Roland Most
//  			<emost@th.physik.uni-frankfurt.de>
//  			Ludwig Jens Papenfort
//                      <papenfort@th.physik.uni-frankfurt.de>
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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "tabulated.hh"
#include "tabulated_implementation.hh"

#include <iostream>

#ifndef STANDALONE

extern "C" void Margherita_readtable_schedule(CCTK_ARGUMENTS) {

  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  if(CCTK_EQUALS(nuceos_table_type, "stellarcollapse"))
    EOS_Tabulated::readtable_scollapse(nuceos_table_name, do_energy_shift, recompute_mu_nu);
  if(CCTK_EQUALS(nuceos_table_type, "compose"))
    EOS_Tabulated::readtable_compose(nuceos_table_name);

  if(extend_table){
	EOS_Tabulated::extend_table_high = true;
  }

  std::cout << "Got rhomax,rhomin " << EOS_Tabulated::eos_rhomax << ","
            << EOS_Tabulated::eos_rhomin << std::endl;
}
#endif
