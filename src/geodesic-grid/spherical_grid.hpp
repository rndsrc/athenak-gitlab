#ifndef SPHERICAL_GRID_SPHERICAL_GRID_HPP_
#define SPHERICAL_GRID_SPHERICAL_GRID_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_grid.hpp
//  \brief definitions for SphericalGrid class

#include "athena.hpp"
#include "geodesic-grid/geodesic_grid.hpp"

//! \class SphericalGrid
//! \brief class to initialize a topological sphere (wrapping around GeodesicGrid class)
class SphericalGrid: public GeodesicGrid {
  public:
    // Creates a geodetic grid with nlev levels and radius rad
    SphericalGrid(MeshBlockPack *pmy_pack, int nlev, Real center[3],
                  bool rotate_g, bool fluxes, Real rad);
    Real radius;
    DualArray1D<Real> area;          // surface area of each face, for integration
    DualArray2D<Real> cart_rcoord;   // cartesian coordinates for grid points
    DualArray2D<int> interp_indcs;   // indices of MeshBlock and zones therein for interp
    DualArray2D<Real> interp_wghts;  // weights for interpolation
    DualArray2D<Real> interp_vals;   // container for data interpolated to sphere

    Real ctr[3];  // center of the sphere
    int nvars; // number of variables to be interpolated to sphere
    void SetPointwiseRadius(DualArray1D<Real> rad);  // set pointwise radius of sphere
    void SetInterpolationIndices();  // set indexing for interpolation
    void SetInterpolationWeights();  // set weights for interpolation
    void InterpToSphere(DvceArray5D<Real> &val);  // interpolate vars in val to sphere
  private:
    MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
};

#endif // SPHERICAL_GRID_SPHERICAL_GRID_HPP_