//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_grid.cpp
//  \brief Tests the geodesic and spherical grid implementation

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"

#include "parameter_input.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "outputs/outputs.hpp"
#include "pgen.hpp"

void GeodesicGridFluxes(HistoryData *pdata, Mesh *pm);

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(Real spin, Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
static void TransformVector(Real spin, Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa1, Real *pa2, Real *pa3);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem
//  \brief Problem Generator for spherical grid test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &coord = pmbp->pcoord->coord_data;
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int nmb1 = pmbp->nmb_thispack - 1;

  // nlevels for geodesic mesh employed in this pgen
  int nlevel = pin->GetInteger("problem","nlevel");

  // User defined history function
  auto &radii = spherical_grids;
  radii.push_back(std::make_unique<SphericalGrid>(pmbp, nlevel, 2.0));
  radii.push_back(std::make_unique<SphericalGrid>(pmbp, nlevel, 3.0));
  radii.push_back(std::make_unique<SphericalGrid>(pmbp, nlevel, 4.0));
  user_hist_func = GeodesicGridFluxes;

  // geodesic mesh
  GeodesicGrid *pmy_geo = nullptr;
  pmy_geo = new GeodesicGrid(nlevel, true, true);

  // print number of angles
  if (global_variable::my_rank==0) {
    printf("\nnangles: %d\n", pmy_geo->nangles);
  }

  // guarantee sum of solid angles is 4 pi
  Real sum_angles = 0.0;
  int nang1 = pmy_geo->nangles - 1;
  auto &solid_angles_ = pmy_geo->solid_angles;
  for (int n=0; n<=nang1; ++n) {
    sum_angles += solid_angles_.h_view(n);
  }
  if (global_variable::my_rank==0) {
    printf("|sum_angles-four_pi|/four_pi: %24.16e\n",
           fabs(sum_angles-4.0*M_PI)/(4.0*M_PI));
  }

  // check that arc lengths are equivalent after computed separately for each angle
  auto posm = pmy_geo->cart_pos_mid;
  auto numn = pmy_geo->num_neighbors;
  auto indn = pmy_geo->ind_neighbors;
  auto arcl = pmy_geo->arc_lengths;
  for (int n=0; n<=nang1; ++n) {
    bool not_equal = false;
    for (int nb=0; nb<numn.h_view(n); ++nb) {
      Real this_arc = arcl.h_view(n,nb);
      bool match_not_found = true;
      for (int nnb=0; nnb<numn.h_view(indn.h_view(n,nb)); ++nnb) {
        Real neigh_arc = arcl.h_view(indn.h_view(n,nb),nnb);
        if (posm.h_view(n,nb,0) == posm.h_view(indn.h_view(n,nb),nnb,0) &&
            posm.h_view(n,nb,1) == posm.h_view(indn.h_view(n,nb),nnb,1) &&
            posm.h_view(n,nb,2) == posm.h_view(indn.h_view(n,nb),nnb,2)) {
          if (this_arc != neigh_arc) { not_equal = true; }
          match_not_found = false;
        }
      }
      if (not_equal || match_not_found) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Error in geodesic grid arc lengths" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  // check that unit flux lengths are equivalent after computed separately for each angle
  auto uvec = pmy_geo->unit_flux;
  for (int n=0; n<=nang1; ++n) {
    bool not_equal = false;
    for (int nb=0; nb<numn.h_view(n); ++nb) {
      Real this_uzeta = uvec.h_view(n,nb,0);
      Real this_upsi  = uvec.h_view(n,nb,1);
      bool match_not_found = true;
      for (int nnb=0; nnb<numn.h_view(indn.h_view(n,nb)); ++nnb) {
        Real neigh_uzeta = uvec.h_view(indn.h_view(n,nb),nnb,0);
        Real neigh_upsi  = uvec.h_view(indn.h_view(n,nb),nnb,1);
        if (posm.h_view(n,nb,0) == posm.h_view(indn.h_view(n,nb),nnb,0) &&
            posm.h_view(n,nb,1) == posm.h_view(indn.h_view(n,nb),nnb,1) &&
            posm.h_view(n,nb,2) == posm.h_view(indn.h_view(n,nb),nnb,2)) {
          if (this_uzeta != -1*neigh_uzeta) { not_equal = true; }
          if (this_upsi != -1*neigh_upsi) { not_equal = true; }
          match_not_found = false;
        }
      }
      if (not_equal || match_not_found) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Error in geodesic grid arc lengths" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  delete pmy_geo;

  // spherical mesh
  SphericalGrid *pmy_sphere = nullptr;
  pmy_sphere = new SphericalGrid(pmbp, nlevel, 3.0);

  // Compute \int sqrt(-gdet) dtheta dphi for spherical KS r=2 surface and spin a=0.5.
  // NOTE(@pdmullen): Notice that in this unit test, the SphericalGrid object was built
  // assuming zero spin (see the input file).  Here, we impose a spin a=0.5.
  // Inherited GeodesicGrid polar positions are assumed to be spherical KS.  Area
  // calulcation does not use interp_coord
  Real my_spin = 0.5;
  Real surface_area = 0.0;
  for (int n=0; n<pmy_sphere->nangles; ++n) {
    surface_area += (SQR(pmy_sphere->radius) +
                     SQR(my_spin*cos(pmy_sphere->polar_pos.h_view(n,0))))*
                     pmy_sphere->solid_angles.h_view(n);
  }
  Real rks2_area = (4.0/3.0)*M_PI*(SQR(my_spin) + 3.0*SQR(pmy_sphere->radius));
  if (global_variable::my_rank==0) {
    printf("|surface_area-analytic|/analytic: %24.16e\n",
           fabs(surface_area-rks2_area)/(rks2_area));
  }

  // Test mass flux in CKS
  DvceArray5D<Real> w0_;
  if (pmbp->phydro != nullptr) {
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    w0_ = pmbp->pmhd->w0;
  }
  Real &spin = coord.bh_spin;
  par_for("set_w0_cks", DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // Compute metric
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v,x2v,x3v,false,spin,glower,gupper);

    // Calculate Boyer-Lindquist coordinates of cell
    Real r, theta, phi;
    GetBoyerLindquistCoordinates(spin, x1v, x2v, x3v, &r, &theta, &phi);

    Real my_ur = 1.0/SQR(r);
    Real u0(0.0), u1(0.0), u2(0.0), u3(0.0);
    TransformVector(spin, my_ur, 0.0, 0.0, x1v, x2v, x3v, &u1, &u2, &u3);

    Real tmp = glower[1][1]*u1*u1 + 2.0*glower[1][2]*u1*u2 + 2.0*glower[1][3]*u1*u3
             + glower[2][2]*u2*u2 + 2.0*glower[2][3]*u2*u3
             + glower[3][3]*u3*u3;
    Real gammasq = 1.0 + tmp;
    Real b = glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
    u0 = (-b - sqrt(fmax(SQR(b) - glower[0][0]*gammasq, 0.0)))/glower[0][0];

    w0_(m,IDN,k,j,i) = 1.0;
    w0_(m,IEN,k,j,i) = 1.0;
    w0_(m,IVX,k,j,i) = u1 - gupper[0][1]/gupper[0][0] * u0;
    w0_(m,IVY,k,j,i) = u2 - gupper[0][2]/gupper[0][0] * u0;
    w0_(m,IVZ,k,j,i) = u3 - gupper[0][3]/gupper[0][0] * u0;
  });

  int nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
  pmy_sphere->InterpolateToSphere(nvars, w0_);

  // check that mass flux is 4 pi
  Real gr_flux = 0.0;
  for (int n=0; n<pmy_sphere->nangles; ++n) {
    // extract interpolated primitives
    Real &int_dn = pmy_sphere->interp_vals.h_view(n,IDN);
    Real &int_vx = pmy_sphere->interp_vals.h_view(n,IVX);
    Real &int_vy = pmy_sphere->interp_vals.h_view(n,IVY);
    Real &int_vz = pmy_sphere->interp_vals.h_view(n,IVZ);

    // coordinate data
    Real x1 = pmy_sphere->interp_coord.h_view(n,0);
    Real x2 = pmy_sphere->interp_coord.h_view(n,1);
    Real x3 = pmy_sphere->interp_coord.h_view(n,2);
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1,x2,x3,false,spin,glower,gupper);
    Real alpha = sqrt(-1.0/gupper[0][0]);

    // compute u^r
    Real a2 = SQR(spin);
    Real rad2 = SQR(x1)+SQR(x2)+SQR(x3);
    Real r = pmy_sphere->radius;
    Real r2 = SQR(r);
    Real drdx = r*x1/(2.0*r2 - rad2 + a2);
    Real drdy = r*x2/(2.0*r2 - rad2 + a2);
    Real drdz = (r*x3 + a2*x3/r)/(2.0*r2-rad2+a2);
    Real q = glower[1][1]*int_vx*int_vx + glower[2][2]*int_vy*int_vy +
             glower[3][3]*int_vz*int_vz +
             2.0*glower[1][2]*int_vx*int_vy + 2.0*glower[1][3]*int_vx*int_vz +
             2.0*glower[2][3]*int_vy*int_vz;
    Real gamma = sqrt(1.0 + q);
    Real int_u0 = gamma/alpha;
    Real int_u1 = int_vx - alpha*gamma*gupper[0][1];
    Real int_u2 = int_vy - alpha*gamma*gupper[0][2];
    Real int_u3 = int_vz - alpha*gamma*gupper[0][3];
    Real int_ur = drdx*int_u1 + drdy*int_u2 + drdz*int_u3;

    // compute mass flux
    gr_flux += int_dn*int_ur*
               (SQR(pmy_sphere->radius) +
                SQR(spin*cos(pmy_sphere->polar_pos.h_view(n,0))))*
                pmy_sphere->solid_angles.h_view(n);
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &gr_flux, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (global_variable::my_rank==0) {
    printf("|GR flux-analytic|/analytic: %24.16e\n",
           fabs(gr_flux-4.0*M_PI)/(4.0*M_PI));
  }

  delete pmy_sphere;

  return;
}


//----------------------------------------------------------------------------------------
// Function for returning fluxes through geodesic grid

void GeodesicGridFluxes(HistoryData *pdata, Mesh *pm) {
  // manually specify primitive variables here to test History infrastructure
  MeshBlockPack *pmbp = pm->pmb_pack;
  DvceArray5D<Real> w0_;
  if (pmbp->phydro != nullptr) {
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    w0_ = pmbp->pmhd->w0;
  }
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int nmb1 = pmbp->nmb_thispack - 1;
  par_for("set_w0", DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
    Real vr = 1.0/SQR(rad);

    w0_(m,IDN,k,j,i) = 1.0;
    w0_(m,IEN,k,j,i) = 1.0;
    w0_(m,IVX,k,j,i) = vr*x1v/rad;
    w0_(m,IVY,k,j,i) = vr*x2v/rad;
    w0_(m,IVZ,k,j,i) = vr*x3v/rad;
  });

  auto &radii = pm->pgen->spherical_grids;
  int nradii = radii.size();
  int nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
  Real mass_flux[3] = {0.0};
  for (int r=0; r<nradii; ++r) {
    radii[r]->InterpolateToSphere(nvars, w0_);
    for (int n=0; n<radii[r]->nangles; ++n) {
      Real &int_dn = radii[r]->interp_vals.h_view(n,IDN);
      Real &int_vx = radii[r]->interp_vals.h_view(n,IVX);
      Real &int_vy = radii[r]->interp_vals.h_view(n,IVY);
      Real &int_vz = radii[r]->interp_vals.h_view(n,IVZ);
      Real &theta = radii[r]->polar_pos.h_view(n,0);
      Real &phi = radii[r]->polar_pos.h_view(n,1);
      Real int_vr = (int_vx*cos(phi)*sin(theta) +
                     int_vy*sin(phi)*sin(theta) +
                     int_vz*cos(theta));
      mass_flux[r] += (int_dn*int_vr*
                       SQR(radii[r]->radius)*radii[r]->solid_angles.h_view(n));
    }
  }

  // set number of and names of history variables for hydro
  pdata->nhist = 3;
  pdata->label[0] = "mdot2";
  pdata->label[1] = "mdot3";
  pdata->label[2] = "mdot4";
  pdata->hdata[0] = mass_flux[0];
  pdata->hdata[1] = mass_flux[1];
  pdata->hdata[2] = mass_flux[2];

  // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }

  return;
}

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(Real spin, Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi) {
    Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
    Real r = fmax((sqrt( SQR(rad) - SQR(spin) + sqrt(SQR(SQR(rad)-SQR(spin))
                        + 4.0*SQR(spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
    *pr = r;
    *ptheta = (fabs(x3/r) < 1.0) ? acos(x3/r) : acos(copysign(1.0, x3));
    *pphi = atan2( (r*x2-spin*x1)/(SQR(r)+SQR(spin)),
                   (spin*x2+r*x1)/(SQR(r)+SQR(spin)) );
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from Boyer-Lindquist to desired coordinates
// Inputs:
//   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist coordinates
//   x1,x2,x3: Cartesian Kerr-Schild coordinates of point
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired coordinates
// Notes:
//   Schwarzschild coordinates match Boyer-Lindquist when a = 0

KOKKOS_INLINE_FUNCTION
static void TransformVector(Real spin, Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa1, Real *pa2, Real *pa3) {
  Real rad = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real r = fmax((sqrt( SQR(rad) - SQR(spin) + sqrt(SQR(SQR(rad)-SQR(spin))
                      + 4.0*SQR(spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  Real delta = SQR(r) - 2.0*r + SQR(spin);
  *pa1 = a1_bl * ( (r*x1+spin*x2)/(SQR(r) + SQR(spin)) - x2*spin/delta) +
         a2_bl * x1*x3/r * sqrt((SQR(r) + SQR(spin))/(SQR(x1) + SQR(x2))) -
         a3_bl * x2;
  *pa2 = a1_bl * ( (r*x2-spin*x1)/(SQR(r) + SQR(spin)) + x1*spin/delta) +
         a2_bl * x2*x3/r * sqrt((SQR(r) + SQR(spin))/(SQR(x1) + SQR(x2))) +
         a3_bl * x1;
  *pa3 = a1_bl * x3/r -
         a2_bl * r * sqrt((SQR(x1) + SQR(x2))/(SQR(r) + SQR(spin)));
  return;
}
