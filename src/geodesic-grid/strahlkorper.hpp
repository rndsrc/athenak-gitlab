#ifndef GEODESIC_GRID_strahlkorper_HPP_
#define GEODESIC_GRID_strahlkorper_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file strahlkorper.hpp
//  \brief definitions for strahlkorper class

#include "athena.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"

//----------------------------------------------------------------------------------------
//! \class Strahlkorper

class Strahlkorper: public SphericalGrid {
 public:
    // Creates a geodetic grid with nlev levels and radius rad
    Strahlkorper(MeshBlockPack *pmy_pack, int nlev, Real rad);
    ~Strahlkorper();

    DualArray1D<Real> pointwise_radius;
    DualArray3D<Real> basis_functions;
    int nlevel;
 private:
    void SetPointwiseRadius(DualArray1D<Real> rad_tmp, Real ctr[3]);  // set indexing for interpolation
    std::pair<double,double> SWSphericalHarm(int l, int m, int s, Real theta, Real phi);
    Real MakeReal(std::pair<double,double> (*func)(int, int, int, Real, Real), int l, int m, Real theta, Real phi);
    void EvaluateSphericalHarm();
};

#endif // GEODESIC_GRID_SPHERICAL_GRID_HPP_
