//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_weyl_scalars.cpp
//  \brief implementation of functions in the Z4c class to interpolate weyl scalar
//  and output the waveform

// C++ standard headers
#include <unistd.h>
#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#include "globals.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"

namespace z4c {

//----------------------------------------------------------------------------------------
// \!fn void Z4c::Z4cWeyl(MeshBlockPack *pmbp)
// \brief compute the weyl scalars given the adm variables and matter state
//
// This function operates only on the interior points of the MeshBlock
void Z4c::ADMQuantities(MeshBlockPack *pmbp) {
  // Spherical Grid for user-defined history
  auto &grids = pmbp->pz4c->adm_spherical_grids;
  auto &u_adm_ints = pmbp->pz4c->u_adm_ints;
  int nradii = grids.size();
  Real eadm_out[nradii][4];

  for (int g=0; g<nradii; ++g) {
    // Interpolate Adm Integrands to the surface
    grids[g]->InterpolateToSphere(4, u_adm_ints);
    Real radius = grids[g]->radius;
    Real eadm_ = 0.0;
    Real padmx_ = 0.0;
    Real padmy_ = 0.0;
    Real padmz_ = 0.0;

    for (int ip = 0; ip < grids[g]->nangles; ++ip) {
      Real weight = grids[g]->solid_angles.h_view(ip);
      Real data = grids[g]->interp_vals.h_view(ip,0);
      Real px = grids[g]->interp_vals.h_view(ip,1);
      Real py = grids[g]->interp_vals.h_view(ip,2);
      Real pz = grids[g]->interp_vals.h_view(ip,3);

      eadm_ += weight * data * radius * radius;
      padmx_ += weight * px * radius * radius;
      padmy_ += weight * py * radius * radius;
      padmz_ += weight * pz * radius * radius;
    }
    eadm_out[g][0] = eadm_;
    eadm_out[g][1] = padmx_;
    eadm_out[g][2] = padmy_;
    eadm_out[g][3] = padmz_;
  }

  // write output
  #if MPI_PARALLEL_ENABLED
  #endif

  if (0 == global_variable::my_rank) {
    for (int g=0; g<nradii; ++g) {
      // Output file names
      std::string filename = "adm_quantities/Eadm_";
      std::stringstream strObj;
      strObj << std::setfill('0') << std::setw(4) << grids[g]->radius;
      filename += strObj.str();
      filename += ".txt";

      // Check if the file already exists
      std::ifstream fileCheck(filename);
      bool fileExists = fileCheck.good();
      fileCheck.close();

      // If the file doesn't exist, create it
      if (!fileExists) {
        std::ofstream createFile(filename);
        createFile.close();

        // Open a file stream for writing header
        std::ofstream outFile;
        // append mode
        outFile.open(filename, std::ios::out | std::ios::app);
        // first append time
        outFile << "# 1:time" << "\t" << "2:Eadm" << "\t" << "3:Px"<< "\t" << "4:Py"<< "\t" << "5:Pz";
        outFile << '\n';

        // Close the file stream
        outFile.close();
      }

      // Open a file stream for writing header
      std::ofstream outFile;

      // append mode
      outFile.open(filename, std::ios::out | std::ios::app);

      // first append time
      outFile << pmbp->pmesh->time << "\t";

      // append waveform
      outFile << std::setprecision(15) << eadm_out[g][0] << '\t' << eadm_out[g][1] 
      << '\t' << eadm_out[g][2] << '\t' << eadm_out[g][3] << '\t';

      outFile << '\n';

      // Close the file stream
      outFile.close();
    }
  }
}
}  // namespace z4c
