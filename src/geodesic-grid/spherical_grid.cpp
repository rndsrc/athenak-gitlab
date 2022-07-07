//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_grid.cpp
//  \brief Initializes a spherical grid

#include <cmath>
#include <list>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "spherical_grid.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

SphericalGrid::SphericalGrid(MeshBlockPack *ppack, int nlev, Real center[3],
                             bool rotate_g, bool fluxes, Real rad):
    GeodesicGrid(nlev,rotate_g,fluxes),
    pmy_pack(ppack),
    radius(rad),
    area("area",1),
    cart_rcoord("cart_rcoord",1,1),
    interp_indcs("interp_indcs",1,1),
    interp_wghts("interp_wghts",1,1),
    interp_vals("interp_vals",1,1) {
  // define center of spherical grid
  ctr[0] = center[0];
  ctr[1] = center[1];
  ctr[2] = center[2];

  // set number of variables
  if (pmy_pack->phydro != nullptr) {
    nvars = pmy_pack->phydro->nhydro + pmy_pack->phydro->nscalars;
  } else if (pmy_pack->pmhd != nullptr) {
    nvars = pmy_pack->pmhd->nmhd + pmy_pack->pmhd->nscalars;
  }

  // reallocate spherical grid arrays
  int ng = pmy_pack->pmesh->mb_indcs.ng;
  Kokkos::realloc(area,nangles);
  Kokkos::realloc(cart_rcoord,nangles,3);
  Kokkos::realloc(interp_indcs,nangles,4);
  Kokkos::realloc(interp_wghts,nangles,3);
  Kokkos::realloc(interp_vals,nangles,nvars);

  // NOTE(@pdmullen): by default, set positions and surface areas assuming constant
  // spherical radius for sphere. Can be overidden by calling
  // SphericalGrid::SetPointwiseRadius()
  for (int n=0; n<nangles; ++n) {
    cart_rcoord.h_view(n,0) = radius*cart_pos.h_view(n,0) + ctr[0];
    cart_rcoord.h_view(n,1) = radius*cart_pos.h_view(n,1) + ctr[1];
    cart_rcoord.h_view(n,2) = radius*cart_pos.h_view(n,2) + ctr[2];
    area.h_view(n) = SQR(radius)*solid_angles.h_view(n);
  }

  // sync dual arrays
  cart_rcoord.template modify<HostMemSpace>();
  cart_rcoord.template sync<DevExeSpace>();
  area.template modify<HostMemSpace>();
  area.template sync<DevExeSpace>();

  SetInterpolationIndices();
  SetInterpolationWeights();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphericalGrid::SetPointwiseRadius
//! \brief set radii, coordinate positions, and surface areas given pointwise radii
//         NOTE(@pdmullen): assumes that if rad_tmp was set in DevExeSpace that it will
//         be synced prior to passing to function

void SphericalGrid::SetPointwiseRadius(DualArray1D<Real> rad_tmp) {
  for (int n=0; n<nangles; ++n) {
    area.h_view(n) = SQR(rad_tmp.h_view(n))*solid_angles.h_view(n);
    cart_rcoord.h_view(n,0) = rad_tmp.h_view(n)*cart_pos.h_view(n,0) + ctr[0];
    cart_rcoord.h_view(n,1) = rad_tmp.h_view(n)*cart_pos.h_view(n,1) + ctr[1];
    cart_rcoord.h_view(n,2) = rad_tmp.h_view(n)*cart_pos.h_view(n,2) + ctr[2];
  }
  // sync dual arrays
  area.template modify<HostMemSpace>();
  area.template sync<DevExeSpace>();
  cart_rcoord.template modify<HostMemSpace>();
  cart_rcoord.template sync<DevExeSpace>();

  SetInterpolationIndices();
  SetInterpolationWeights();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphericalGrid::SetInterpolationIndices
//! \brief determine which MeshBlocks and MeshBlock zones therein that will be used in
//         interpolation onto the sphere

void SphericalGrid::SetInterpolationIndices() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &ng = indcs.ng;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang1 = nangles - 1;

  auto &rcoord = cart_rcoord;
  auto &iindcs = interp_indcs;
  for (int m=0; m<=nmb1; ++m) {
    // determine coordinate maxima for this MeshBlock
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;

    // determine grid cell spacings for this MeshBlock
    Real &dx1 = size.h_view(m).dx1;
    Real &dx2 = size.h_view(m).dx2;
    Real &dx3 = size.h_view(m).dx3;

    // Loop over all points to find those belonging to this spherical patch
    for (int n=0; n<=nang1; ++n) {
      // Default meshblock indices to -1
      if (m==0) { iindcs.h_view(n,0) = -1; }
      if ((rcoord.h_view(n,0) >= x1min && rcoord.h_view(n,0) <= x1max) &&
          (rcoord.h_view(n,1) >= x2min && rcoord.h_view(n,1) <= x2max) &&
          (rcoord.h_view(n,2) >= x3min && rcoord.h_view(n,2) <= x3max)) {
        // save MeshBlock and zone index for nearest position to spherical patch center
        iindcs.h_view(n,0) = m;
        iindcs.h_view(n,1) = (int) std::floor((rcoord.h_view(n,0)-(x1min+dx1/2.0))/dx1);
        iindcs.h_view(n,2) = (int) std::floor((rcoord.h_view(n,1)-(x2min+dx2/2.0))/dx2);
        iindcs.h_view(n,3) = (int) std::floor((rcoord.h_view(n,2)-(x3min+dx3/2.0))/dx3);
      }
    }
  }

  // sync dual arrays
  interp_indcs.template modify<DevExeSpace>();
  interp_indcs.template sync<HostMemSpace>();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphericalGrid::SetInterpolationWeights
//! \brief set weights used by Lagrangian interpolation

void SphericalGrid::SetInterpolationWeights() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &ng = indcs.ng;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  for (int n=0; n<nangles; ++n) {
    int &ii0 = iindcs.d_view(n,0);
    int &ii1 = iindcs.d_view(n,1);
    int &ii2 = iindcs.d_view(n,2);
    int &ii3 = iindcs.d_view(n,3);

    Real &x0 = cart_rcoord.h_view(n,0);
    Real &y0 = cart_rcoord.h_view(n,1);
    Real &z0 = cart_rcoord.h_view(n,2);

    Real &x1min = size.h_view(ii0).x1min;
    Real &x1max = size.h_view(ii0).x1max;
    Real &x2min = size.h_view(ii0).x2min;
    Real &x2max = size.h_view(ii0).x2max;
    Real &x3min = size.h_view(ii0).x3min;
    Real &x3max = size.h_view(ii0).x3max;
    for (int i=0; i<2*ng+2; ++i) {
      iwghts.h_view(i,0) = 1.;
      iwghts.h_view(i,1) = 1.;
      iwghts.h_view(i,2) = 1.;
      for (int j=0; j<2*ng+2; ++j) {
        if (j != i) {
          Real x1vpi = CellCenterX(ii1-ng+i, indcs.nx1, x1min, x1max);
          Real x1vpj = CellCenterX(ii1-ng+j, indcs.nx1, x1min, x1max);
          iwghts.h_view(i,0) *= (x0-x1vpj)/(x1vpi-x1vpj);
          Real x2vpi = CellCenterX(ii2-ng+i, indcs.nx2, x2min, x2max);
          Real x2vpj = CellCenterX(ii2-ng+j, indcs.nx2, x2min, x2max);
          iwghts.h_view(i,1) *= (y0-x2vpj)/(x2vpi-x2vpj);
          Real x3vpi = CellCenterX(ii3-ng+i, indcs.nx3, x3min, x3max);
          Real x3vpj = CellCenterX(ii3-ng+j, indcs.nx3, x3min, x3max);
          iwghts.h_view(i,2) *= (z0-x3vpj)/(x3vpi-x3vpj);
        }
      }
    }
  }

  // sync dual arrays
  interp_wghts.template modify<DevExeSpace>();
  interp_wghts.template sync<HostMemSpace>();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphericalGrid::InterpToSphere
//! \brief interpolate Cartesian data to surface of sphere

void SphericalGrid::InterpToSphere(DvceArray5D<Real> &val) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int &ng = pmy_pack->pmesh->mb_indcs.ng;
  int nang1 = nangles - 1;
  int nvar1 = nvars - 1;

  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  auto &ivals = interp_vals;
  par_for("int2sph",DevExeSpace(),0,nang1,0,nvar1,
  KOKKOS_LAMBDA(int n, int v) {
    int ii0 = iindcs.d_view(n,0);
    int ii1 = iindcs.d_view(n,1);
    int ii2 = iindcs.d_view(n,2);
    int ii3 = iindcs.d_view(n,3);
    Real int_value = 0.0;
    for (int i=0; i<2*ng+2; i++) {
      for (int j=0; j<2*ng+2; j++) {
        for (int k=0; k<2*ng+2; k++) {
          Real iwght = iwghts.d_view(i,0)*iwghts.d_view(j,1)*iwghts.d_view(k,2);
          int_value += iwght*val(ii0,v,ii3-(ng-k-ks),ii2-(ng-j-js),ii1-(ng-i-is));
        }
      }
    }
    ivals.d_view(n,v) = int_value;
  });

  // sync dual arrays
  interp_vals.template modify<DevExeSpace>();
  interp_vals.template sync<HostMemSpace>();

  return;
}
