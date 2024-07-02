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
  auto &eadm_out = pmbp->pz4c->eadm_out;

  // number of radii
  int nradii = grids.size();

  // maximum l; TODO(@hzhu): read in from input file
  int lmax = 8;
  bool bitant = false;

  Real ylmR,ylmI;
  for (int g=0; g<nradii; ++g) {
    // Interpolate Adm Integrands to the surface
    grids[g]->InterpolateToSphere(1, u_adm_ints);
    Real radius = grids[g]->radius;
    Real eadm_ = 0.0;
    for (int ip = 0; ip < grids[g]->nangles; ++ip) {
      Real weight = grids[g]->solid_angles.h_view(ip);
      Real data = grids[g]->interp_vals.h_view(ip,0);
      eadm_ += weight * data * radius * radius;
    }
    eadm_out(g,0) = eadm_;
  }

  // write output
  #if MPI_PARALLEL_ENABLED
  if (0 == global_variable::my_rank) {
    for (int g=0; g<nradii; ++g) {
      MPI_Reduce(MPI_IN_PLACE, &eadm_out(g,0), 1,
        MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    }
  } else {
    for (int g=0; g<nradii; ++g) {
      MPI_Reduce(&eadm_out(g,0), &eadm_out(g,0), 1,
        MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    }
  }
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
        outFile << "# 1:time" << "\t" << "2:Eadm";
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
      outFile << std::setprecision(15) << eadm_out(g,0) << '\t';

      outFile << '\n';

      // Close the file stream
      outFile.close();
    }
  }
}


}  // namespace z4c
