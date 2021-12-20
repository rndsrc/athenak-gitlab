//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_geom.cpp
//  \brief Initializes angular mesh and coordinate frame data.  Taken from gr_rad branch.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"
#include "radiation.hpp"
#include "radiation_tetrad.hpp"

#include <cmath>
  
namespace radiation {

KOKKOS_INLINE_FUNCTION
void DeviceUnitFluxDir(int ic1, int ic2, int nlvl,
                       DualArray4D<Real> ah_norm,
                       DualArray2D<Real> ap_norm,
                       Real *dtheta, Real *dphi);
KOKKOS_INLINE_FUNCTION
void DeviceGetGridCartPosition(int lm, int nlvl,
                               DualArray4D<Real> ah_norm,
                               DualArray2D<Real> ap_norm,
                               Real *x, Real *y, Real *z);
KOKKOS_INLINE_FUNCTION
void DeviceGetGridCartPositionMid(int lm, int nb, int nlvl,
                                  DualArray4D<Real> ah_norm,
                                  DualArray2D<Real> ap_norm,
                                  Real *x, Real *y, Real *z);
KOKKOS_INLINE_FUNCTION
void DeviceGreatCircleParam(Real zeta1, Real zeta2, Real psi1, Real psi2,
                            Real *apar, Real *psi0);

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::InitMesh()
//! \brief Initialize angular mesh

void Radiation::InitAngularMesh()
{
  int nlev = nlevels;
  Real sin_ang = 2.0/sqrt(5.0);
  Real cos_ang = 1.0/sqrt(5.0);
  Real p1[3] = {0.0, 0.0, 1.0};
  Real p2[3] = {sin_ang, 0.0, cos_ang};
  Real p3[3] = {sin_ang*cos(0.2*M_PI),  sin_ang*sin(0.2*M_PI),  -cos_ang};
  Real p4[3] = {sin_ang*cos(-0.4*M_PI), sin_ang*sin(-0.4*M_PI),  cos_ang};
  Real p5[3] = {sin_ang*cos(-0.2*M_PI), sin_ang*sin(-0.2*M_PI), -cos_ang};
  Real p6[3] = {0.0, 0.0, -1.0};

  // get coordinates of each face center, i.e., the normal component 
  auto amesh_normals_ = amesh_normals;

  // start with poles, which we can set explicitly
  auto ameshp_normals_ = ameshp_normals;
  ameshp_normals_.h_view(0,0) = 0.0;
  ameshp_normals_.h_view(0,1) = 0.0;
  ameshp_normals_.h_view(0,2) = 1.0;
  ameshp_normals_.h_view(1,0) = 0.0;
  ameshp_normals_.h_view(1,1) = 0.0;
  ameshp_normals_.h_view(1,2) = -1.0;

  // now move on and start by filling in one of the five patches
  // we only fill in the center (ignoring ghost values)
  int row_index = 1;
  for (int l=0; l<nlev; ++l) {
    int col_index = 1;
    for (int m=l; m<nlev; ++m) {
      Real x = ((m-l+1)*p2[0] + (nlev-m-1)*p1[0] + l*p4[0])/(Real)(nlev);
      Real y = ((m-l+1)*p2[1] + (nlev-m-1)*p1[1] + l*p4[1])/(Real)(nlev);
      Real z = ((m-l+1)*p2[2] + (nlev-m-1)*p1[2] + l*p4[2])/(Real)(nlev);
      Real norm = sqrt(SQR(x) + SQR(y) + SQR(z));
      amesh_normals_.h_view(0,row_index,col_index,0) = x/norm;
      amesh_normals_.h_view(0,row_index,col_index,1) = y/norm;
      amesh_normals_.h_view(0,row_index,col_index,2) = z/norm;
      col_index += 1;
    }
    for (int m=nlev-l; m<nlev; ++m) {
      Real x = ((nlev-l)*p2[0] + (m-nlev+l+1)*p5[0]
                + (nlev-m-1)*p4[0])/(Real)(nlev);
      Real y = ((nlev-l)*p2[1] + (m-nlev+l+1)*p5[1]
                + (nlev-m-1)*p4[1])/(Real)(nlev);
      Real z = ((nlev-l)*p2[2] + (m-nlev+l+1)*p5[2]
                + (nlev-m-1)*p4[2])/(Real)(nlev);
      Real norm = sqrt(SQR(x) + SQR(y) + SQR(z));
      amesh_normals_.h_view(0,row_index,col_index,0) = x/norm;
      amesh_normals_.h_view(0,row_index,col_index,1) = y/norm;
      amesh_normals_.h_view(0,row_index,col_index,2) = z/norm;
      col_index += 1;
    }
    for (int m=l; m<nlev; ++m) {
      Real x = ((m-l+1)*p3[0] + (nlev-m-1)*p2[0] + l*p5[0])/(Real)(nlev);
      Real y = ((m-l+1)*p3[1] + (nlev-m-1)*p2[1] + l*p5[1])/(Real)(nlev);
      Real z = ((m-l+1)*p3[2] + (nlev-m-1)*p2[2] + l*p5[2])/(Real)(nlev);
      Real norm = sqrt(SQR(x) + SQR(y) + SQR(z));
      amesh_normals_.h_view(0,row_index,col_index,0) = x/norm;
      amesh_normals_.h_view(0,row_index,col_index,1) = y/norm;
      amesh_normals_.h_view(0,row_index,col_index,2) = z/norm;
      col_index += 1;
    }
    for (int m=nlev-l; m<nlev; ++m) {
      Real x = ((nlev-l)*p3[0] + (m-nlev+l+1)*p6[0]
                + (nlev-m-1)*p5[0])/(Real)(nlev);
      Real y = ((nlev-l)*p3[1] + (m-nlev+l+1)*p6[1]
                + (nlev-m-1)*p5[1])/(Real)(nlev);
      Real z = ((nlev-l)*p3[2] + (m-nlev+l+1)*p6[2]
                + (nlev-m-1)*p5[2])/(Real)(nlev);
      Real norm = sqrt(SQR(x) + SQR(y) + SQR(z));
      amesh_normals_.h_view(0,row_index,col_index,0) = x/norm;
      amesh_normals_.h_view(0,row_index,col_index,1) = y/norm;
      amesh_normals_.h_view(0,row_index,col_index,2) = z/norm;
      col_index += 1;
    }
    row_index += 1;
  }

  // now fill the other four patches by rotating the first one. only set 
  // the internal (non-ghost) values
  for (int patch=1; patch<5; ++patch) {
    for (int l=1; l<1+nlev; ++l) {
      for (int m=1; m<1+2*nlev; ++m) {
        Real x0 = amesh_normals_.h_view(0,l,m,0);
        Real y0 = amesh_normals_.h_view(0,l,m,1);
        Real z0 = amesh_normals_.h_view(0,l,m,2);
        amesh_normals_.h_view(patch,l,m,0) = (x0*cos(patch*0.4*M_PI)
                                              + y0*sin(patch*0.4*M_PI));
        amesh_normals_.h_view(patch,l,m,1) = (y0*cos(patch*0.4*M_PI)
                                              - x0*sin(patch*0.4*M_PI));
        amesh_normals_.h_view(patch,l,m,2) = z0;
      }
    }
  }

  // TODO(@gnwong, @pdmullen) maybe figure out how to make this a
  // neat function or remove entirely
  auto blocks_n = amesh_normals_;
  for (int i=0; i<3; ++i) {
    for (int bl=0; bl<5; ++bl) {
      for (int k=0; k<nlev; ++k) {
        blocks_n.h_view(bl,0,k+1,i)           = blocks_n.h_view((bl+4)%5,k+1,1,i);
        blocks_n.h_view(bl,0,k+nlev+1,i)      = blocks_n.h_view((bl+4)%5,nlev,k+1,i);
        blocks_n.h_view(bl,k+1,2*nlev+1,i)    = blocks_n.h_view((bl+4)%5,nlev,k+nlev+1,i);
        blocks_n.h_view(bl,k+2,0,i)           = blocks_n.h_view((bl+1)%5,1,k+1,i);
        blocks_n.h_view(bl,nlev+1,k+1,i)      = blocks_n.h_view((bl+1)%5,1,k+nlev+1,i);
        blocks_n.h_view(bl,nlev+1,k+nlev+1,i) = blocks_n.h_view((bl+1)%5,k+2,2*nlev,i);
      }
      blocks_n.h_view(bl,1,0,i)           = ameshp_normals_.h_view(0,i);
      blocks_n.h_view(bl,nlev+1,2*nlev,i) = ameshp_normals_.h_view(1,i);
      blocks_n.h_view(bl,0,2*nlev+1,i)    = blocks_n.h_view(bl,0,2*nlev,i);
    }
  }

  amesh_normals_.template modify<HostMemSpace>();
  ameshp_normals_.template modify<HostMemSpace>();

  amesh_normals_.template sync<DevExeSpace>();
  ameshp_normals_.template sync<DevExeSpace>();

  // generate 2d -> 1d map
  auto amesh_indices_ = amesh_indices;
  auto ameshp_indices_ = ameshp_indices;
  for (int patch=0; patch<5; ++patch) {
    for (int l = 0; l < nlev; ++l) {
      for (int m = 0; m < 2*nlev; ++m) {
        // set center (non-ghost) values
        amesh_indices_.h_view(patch,l+1,m+1) = patch*2*SQR(nlev) + l*2*nlev + m;
      }
    }
  }

  ameshp_indices_.h_view(0) = 5*2*SQR(nlev);
  ameshp_indices_.h_view(1) = 5*2*SQR(nlev) + 1;

  // TODO(@gnwong, @pdmullen) maybe figure out how to make this a
  // neat function or remove entirely
  auto blocks_i = amesh_indices_;
  for (int bl=0; bl<5; ++bl) {
    for (int k=0; k<nlev; ++k) {
      blocks_i.h_view(bl,0,k+1)           = blocks_i.h_view((bl+4)%5,k+1,1);
      blocks_i.h_view(bl,0,k+nlev+1)      = blocks_i.h_view((bl+4)%5,nlev,k+1);
      blocks_i.h_view(bl,k+1,2*nlev+1)    = blocks_i.h_view((bl+4)%5,nlev,k+nlev+1);
      blocks_i.h_view(bl,k+2,0)           = blocks_i.h_view((bl+1)%5,1,k+1);
      blocks_i.h_view(bl,nlev+1,k+1)      = blocks_i.h_view((bl+1)%5,1,k+nlev+1);
      blocks_i.h_view(bl,nlev+1,k+nlev+1) = blocks_i.h_view((bl+1)%5,k+2,2*nlev);
    }
    blocks_i.h_view(bl,1,0)           = ameshp_indices_.h_view(0);
    blocks_i.h_view(bl,nlev+1,2*nlev) = ameshp_indices_.h_view(1);
    blocks_i.h_view(bl,0,2*nlev+1)    = blocks_i.h_view(bl,0,2*nlev);
  }

  amesh_indices_.template modify<HostMemSpace>();
  ameshp_indices_.template modify<HostMemSpace>();

  amesh_indices.template sync<DevExeSpace>();
  ameshp_indices.template sync<DevExeSpace>();

  // set up geometric factors and neighbor information arrays
  auto solid_angle_ = solid_angle;
  auto num_neighbors_ = num_neighbors;
  auto ind_neighbors_ = ind_neighbors;
  auto arc_lengths_ = arc_lengths;

  for (int lm=0; lm<nangles; ++lm) {
    Real dual_edge[6];
    int neighbors[6];
    solid_angle_.h_view(lm) = ComputeWeightAndDualEdges(lm, dual_edge);
    num_neighbors_.h_view(lm) = GetNeighbors(lm, neighbors);
    for (int nb=0; nb<6; ++nb) {
      // TODO(@gnwong, @pdmullen) is it necessary to save this information?
      ind_neighbors_.h_view(lm, nb) = neighbors[nb];
      arc_lengths_.h_view(lm, nb) = dual_edge[nb];
    }
  }

  solid_angle_.template modify<HostMemSpace>();
  num_neighbors_.template modify<HostMemSpace>();
  ind_neighbors_.template modify<HostMemSpace>();
  arc_lengths_.template modify<HostMemSpace>();

  solid_angle_.template sync<DevExeSpace>();
  num_neighbors_.template sync<DevExeSpace>();
  ind_neighbors_.template sync<DevExeSpace>();
  arc_lengths_.template sync<DevExeSpace>();

  auto xi_mn_ = xi_mn;
  auto eta_mn_ = eta_mn;
  for (int lm=0; lm<nangles; ++lm) {
    Real xi_coord[6];
    Real eta_coord[6];
    ComputeXiEta(lm, xi_coord, eta_coord);
    for (int nb = 0; nb < 6; ++nb) {
      xi_mn_.h_view(lm,nb) = xi_coord[nb];
      eta_mn_.h_view(lm,nb) = eta_coord[nb];
    }
  }

  xi_mn_.template modify<HostMemSpace>();
  eta_mn_.template modify<HostMemSpace>();

  xi_mn_.template sync<DevExeSpace>();
  eta_mn_.template sync<DevExeSpace>();

  // TODO(@gnwong, @pdmullen) make this prettier
  Real rotangles[2];
  OptimalAngles(rotangles);
  if (rotate_geo) {
    RotateGrid(rotangles[0], rotangles[1]);
  }

  auto nh_c_ = nh_c;
  auto nh_f_ = nh_f;

  for (int lm=0; lm<nangles; ++lm) {
    Real x, y, z;
    GetGridCartPosition(lm, &x,&y,&z);
    nh_c_.h_view(lm,0) = 1.0;
    nh_c_.h_view(lm,1) = x;
    nh_c_.h_view(lm,2) = y;
    nh_c_.h_view(lm,3) = z;
    int nn = num_neighbors_.h_view(lm);
    for (int nb=0; nb<nn; ++nb) {
      Real xm, ym, zm;
      GetGridCartPositionMid(lm, ind_neighbors_.h_view(lm,nb), &xm,&ym,&zm);
      nh_f_.h_view(lm,nb,0) = 1.0;
      nh_f_.h_view(lm,nb,1) = xm;
      nh_f_.h_view(lm,nb,2) = ym;
      nh_f_.h_view(lm,nb,3) = zm;
    }
        if (nn==5) {
      nh_f_.h_view(lm,5,0) = std::numeric_limits<Real>::quiet_NaN();
      nh_f_.h_view(lm,5,1) = std::numeric_limits<Real>::quiet_NaN();
      nh_f_.h_view(lm,5,2) = std::numeric_limits<Real>::quiet_NaN();
      nh_f_.h_view(lm,5,3) = std::numeric_limits<Real>::quiet_NaN();
    }
  }

  nh_c_.template modify<HostMemSpace>();
  nh_f_.template modify<HostMemSpace>();

  nh_c_.template sync<DevExeSpace>();
  nh_f_.template sync<DevExeSpace>();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::InitCoordinateFrame()
//! \brief Initialize frame related quantities.

void Radiation::InitCoordinateFrame()
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;

  int nangles_ = nangles;
  int nlev_ = nlevels;

  int &nmb = pmy_pack->nmb_thispack;
  auto coord = pmy_pack->coord.coord_data;

  auto nmu_ = nmu;
  auto n_mu_ = n_mu;
  auto nh_c_ = nh_c;
  auto nh_f_ = nh_f;

  par_for("rad_nl_nu", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real &x1min = coord.mb_size.d_view(m).x1min;
      Real &x1max = coord.mb_size.d_view(m).x1max;
      int nx1 = coord.mb_indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = coord.mb_size.d_view(m).x2min;
      Real &x2max = coord.mb_size.d_view(m).x2max;
      int nx2 = coord.mb_indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = coord.mb_size.d_view(m).x3min;
      Real &x3max = coord.mb_size.d_view(m).x3max;
      int nx3 = coord.mb_indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
      ComputeTetrad(x1v, x2v, x3v, coord.snake, coord.bh_mass, coord.bh_spin,
                    e, e_cov, omega);

      for (int lm=0; lm<nangles_; ++lm) {
        Real n0 = 0.0;
        Real n1 = 0.0;
        Real n2 = 0.0;
        Real n3 = 0.0;
        Real n_0 = 0.0;
        Real n_1 = 0.0;
        Real n_2 = 0.0;
        Real n_3 = 0.0;
        for (int d=0; d<4; ++d) {
          n0 += e[d][0]*nh_c_.d_view(lm,d);
          n1 += e[d][1]*nh_c_.d_view(lm,d);
          n2 += e[d][2]*nh_c_.d_view(lm,d);
          n3 += e[d][3]*nh_c_.d_view(lm,d);
          n_0 += e_cov[d][0]*nh_c_.d_view(lm,d);
          n_1 += e_cov[d][1]*nh_c_.d_view(lm,d);
          n_2 += e_cov[d][2]*nh_c_.d_view(lm,d);
          n_3 += e_cov[d][3]*nh_c_.d_view(lm,d);
        }
        nmu_(m,lm,k,j,i,0) = n0;
        nmu_(m,lm,k,j,i,1) = n1;
        nmu_(m,lm,k,j,i,2) = n2;
        nmu_(m,lm,k,j,i,3) = n3;
        n_mu_(m,lm,k,j,i,0) = n_0;
        n_mu_(m,lm,k,j,i,1) = n_1;
        n_mu_(m,lm,k,j,i,2) = n_2;
        n_mu_(m,lm,k,j,i,3) = n_3;
      }
    }
  );

  // Calculate n^1 n_mu
  auto n1_n_0_ = n1_n_0;

  par_for("rad_n1_n_0", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, n1,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real &x1min = coord.mb_size.d_view(m).x1min;
      Real &x1max = coord.mb_size.d_view(m).x1max;
      int nx1 = coord.mb_indcs.nx1;
      Real x1f = LeftEdgeX(i-is, nx1, x1min, x1max);

      Real &x2min = coord.mb_size.d_view(m).x2min;
      Real &x2max = coord.mb_size.d_view(m).x2max;
      int nx2 = coord.mb_indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = coord.mb_size.d_view(m).x3min;
      Real &x3max = coord.mb_size.d_view(m).x3max;
      int nx3 = coord.mb_indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
      ComputeTetrad(x1f, x2v, x3v, coord.snake, coord.bh_mass, coord.bh_spin,
                    e, e_cov, omega);

      for (int lm=0; lm<nangles_; ++lm) {
        Real n1 = 0.0;
        Real n_0 = 0.0;
        for (int d = 0; d < 4; ++d) {
          n1 += e[d][1]*nh_c_.d_view(lm,d);
          n_0 += e_cov[d][0]*nh_c_.d_view(lm,d);
        }
        n1_n_0_(m,lm,k,j,i) = n1*n_0;
      }
    }
  );

  // Calculate n^2 n_mu
  auto n2_n_0_ = n2_n_0;

  par_for("rad_n2_n_0", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, n2, 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real &x1min = coord.mb_size.d_view(m).x1min;
      Real &x1max = coord.mb_size.d_view(m).x1max;
      int nx1 = coord.mb_indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = coord.mb_size.d_view(m).x2min;
      Real &x2max = coord.mb_size.d_view(m).x2max;
      int nx2 = coord.mb_indcs.nx2;
      Real x2f = LeftEdgeX(j-js, nx2, x2min, x2max);

      Real &x3min = coord.mb_size.d_view(m).x3min;
      Real &x3max = coord.mb_size.d_view(m).x3max;
      int nx3 = coord.mb_indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
      ComputeTetrad(x1v, x2f, x3v, coord.snake, coord.bh_mass, coord.bh_spin,
                    e, e_cov, omega);

      for (int lm=0; lm<nangles_; ++lm) {
        Real n2 = 0.0;
        Real n_0 = 0.0;
        for (int d = 0; d < 4; ++d) {
          n2 += e[d][2]*nh_c_.d_view(lm,d);
          n_0 += e_cov[d][0]*nh_c_.d_view(lm,d);
        }
        n2_n_0_(m,lm,k,j,i) = n2*n_0;
      }
    }
  );


  // Calculate n^3 n_mu
  auto n3_n_0_ = n3_n_0;

  par_for("rad_n3_n_0", DevExeSpace(), 0, (nmb-1), 0, n3, 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real &x1min = coord.mb_size.d_view(m).x1min;
      Real &x1max = coord.mb_size.d_view(m).x1max;
      int nx1 = coord.mb_indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = coord.mb_size.d_view(m).x2min;
      Real &x2max = coord.mb_size.d_view(m).x2max;
      int nx2 = coord.mb_indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = coord.mb_size.d_view(m).x3min;
      Real &x3max = coord.mb_size.d_view(m).x3max;
      int nx3 = coord.mb_indcs.nx3;
      Real x3f = LeftEdgeX(k-ks, nx3, x3min, x3max);

      Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
      ComputeTetrad(x1v, x2v, x3f, coord.snake, coord.bh_mass, coord.bh_spin,
                    e, e_cov, omega);

      for (int lm=0; lm<nangles_; ++lm) {
        Real n3 = 0.0;
        Real n_0 = 0.0;
        for (int d = 0; d < 4; ++d) {
          n3 += e[d][3]*nh_c_.d_view(lm,d);
          n_0 += e_cov[d][0]*nh_c_.d_view(lm,d);
        }
        n3_n_0_(m,lm,k,j,i) = n3*n_0;
      }
    }
  );

  // Calculate n^angle n_0
  auto num_neighbors_ = num_neighbors;
  auto ind_neighbors_ = ind_neighbors;
  auto na_n_0_ = na_n_0;
  auto amesh_normals_ = amesh_normals;
  auto ameshp_normals_ = ameshp_normals;

  par_for("rad_na_n_0", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real &x1min = coord.mb_size.d_view(m).x1min;
      Real &x1max = coord.mb_size.d_view(m).x1max;
      int nx1 = coord.mb_indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = coord.mb_size.d_view(m).x2min;
      Real &x2max = coord.mb_size.d_view(m).x2max;
      int nx2 = coord.mb_indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = coord.mb_size.d_view(m).x3min;
      Real &x3max = coord.mb_size.d_view(m).x3max;
      int nx3 = coord.mb_indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
      ComputeTetrad(x1v, x2v, x3v, coord.snake, coord.bh_mass, coord.bh_spin,
                    e, e_cov, omega);

      for (int lm=0; lm<nangles_; ++lm) {
        for (int nb=0; nb<num_neighbors_.d_view(lm); ++nb) {
            Real zeta_f = acos(nh_f_.d_view(lm,nb,3));
            Real psi_f  = atan2(nh_f_.d_view(lm,nb,2), nh_f_.d_view(lm,nb,1));

            Real na1 = 0.0;
            Real na2 = 0.0;
            for (int n = 0; n < 4; ++n) {
              for (int p = 0; p < 4; ++p) {
                na1 += (1.0/sin(zeta_f)*nh_f_.d_view(lm,nb,n)*nh_f_.d_view(lm,nb,p)
                        * (nh_f_.d_view(lm,nb,0)*omega[3][n][p]
                        -  nh_f_.d_view(lm,nb,3)*omega[0][n][p]));
                na2 += (1.0/SQR(sin(zeta_f))*nh_f_.d_view(lm,nb,n)*nh_f_.d_view(lm,nb,p)
                        * (nh_f_.d_view(lm,nb,2)*omega[1][n][p]
                        -  nh_f_.d_view(lm,nb,1)*omega[n][p][2]));
              }
            }

            Real unit_zeta, unit_psi;
            DeviceUnitFluxDir(lm,ind_neighbors_.d_view(lm,nb),nlev_,
                              amesh_normals_,ameshp_normals_,&unit_zeta,&unit_psi);

            Real na = na1*unit_zeta + na2*unit_psi;
            Real n_0 = 0.0;
            for (int n = 0; n < 4; ++n) {
              n_0 += e_cov[n][0]*nh_f_.d_view(lm,nb,n);
            }
            na_n_0_(m,lm,k,j,i,nb) = na*n_0;
        }
      }
    }
  );

  // Calculate norm_to_tet
  auto norm_to_tet_ = norm_to_tet;

  par_for("rad_norm_to_tet", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real &x1min = coord.mb_size.d_view(m).x1min;
      Real &x1max = coord.mb_size.d_view(m).x1max;
      int nx1 = coord.mb_indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = coord.mb_size.d_view(m).x2min;
      Real &x2max = coord.mb_size.d_view(m).x2max;
      int nx2 = coord.mb_indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = coord.mb_size.d_view(m).x3min;
      Real &x3max = coord.mb_size.d_view(m).x3max;
      int nx3 = coord.mb_indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real g_[NMETRIC], gi_[NMETRIC];
      ComputeMetricAndInverse(x1v, x2v, x3v, true, coord.snake,
                              coord.bh_mass, coord.bh_spin, g_, gi_);
      Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
      ComputeTetrad(x1v, x2v, x3v, coord.snake, coord.bh_mass, coord.bh_spin,
                    e, e_cov, omega);

      // Set Minkowski metric
      Real eta[4][4] = {0.0};
      eta[0][0] = -1.0;
      eta[1][1] = 1.0;
      eta[2][2] = 1.0;
      eta[3][3] = 1.0;

      // Calculate normal-to-coordinate transformation
      Real norm_to_coord[4][4] = {0.0};
      Real alpha = 1.0/sqrt(-gi_[I00]);
      norm_to_coord[0][0] = 1.0/alpha;
      norm_to_coord[1][0] = -alpha*gi_[I01];
      norm_to_coord[2][0] = -alpha*gi_[I02];
      norm_to_coord[3][0] = -alpha*gi_[I03];
      norm_to_coord[1][1] = 1.0;
      norm_to_coord[2][2] = 1.0;
      norm_to_coord[3][3] = 1.0;

      for (int d1=0; d1<4; ++d1) {
        for (int d2=0; d2<4; ++d2) {
          norm_to_tet_(m,d1,d2,k,j,i) = 0.0;
          for (int p=0; p<4; ++p) {
            for (int q=0; q<4; ++q) {
              norm_to_tet_(m,d1,d2,k,j,i) += eta[d1][p]*e_cov[p][q]*norm_to_coord[q][d2];
            }
          }
        }
      }
    }
  );

  return;
}

// TODO(@gnwong, @pdmullen) implement new function that returns num neighbors using
// if statements instead of also computing the neighbors too
int Radiation::GetNeighbors(int lm, int neighbors[6])
const {
  int num_neighbors;
  int nlev = nlevels;
  auto amesh_indices_ = amesh_indices;
 
  // handle north pole 
  if (lm==10*nlev*nlev) {
    for (int bl = 0; bl < 5; ++bl) {
      neighbors[bl] = amesh_indices_.h_view(bl,1,1);
    }
    neighbors[5] = not_a_patch;
    num_neighbors = 5;
  } else if (lm == 10*nlev*nlev + 1) {  // handle south pole
    for (int bl = 0; bl < 5; ++bl) {
      neighbors[bl] = amesh_indices_.h_view(bl,nlev,2*nlev);
    }
    neighbors[5] = not_a_patch;
    num_neighbors = 5;
  } else {
    int ibl0 =  lm / (2*nlev*nlev);
    int ibl1 = (lm % (2*nlev*nlev)) / (2*nlev);
    int ibl2 = (lm % (2*nlev*nlev)) % (2*nlev);
    neighbors[0] = amesh_indices_.h_view(ibl0, ibl1+1, ibl2+2);
    neighbors[1] = amesh_indices_.h_view(ibl0, ibl1+2, ibl2+1);
    neighbors[2] = amesh_indices_.h_view(ibl0, ibl1+2, ibl2);
    neighbors[3] = amesh_indices_.h_view(ibl0, ibl1+1, ibl2);
    neighbors[4] = amesh_indices_.h_view(ibl0, ibl1  , ibl2+1);

    // TODO(@gnwong, @pdmullen) check carefully, see if it can be inline optimized
    if (lm % (2*nlev*nlev) == nlev-1 || lm % (2*nlev*nlev) == 2*nlev-1) {
      neighbors[5] = not_a_patch;
      num_neighbors = 5;
    } else {
      neighbors[5] = amesh_indices_.h_view(ibl0, ibl1, ibl2+2);
      num_neighbors = 6;
    }
  }
  return num_neighbors;
}

Real Radiation::ComputeWeightAndDualEdges(int lm, Real length[6])
const {
  int nvec[6];
  int nnum = GetNeighbors(lm, nvec);
  Real x0, y0, z0;
  GetGridCartPosition(lm, &x0,&y0,&z0);
  Real weight = 0.0;
  for (int nb = 0; nb < nnum; ++nb) {
    Real xn1, yn1, zn1;
    Real xn2, yn2, zn2;
    Real xn3, yn3, zn3;
    GetGridCartPosition(nvec[(nb + nnum - 1)%nnum],&xn1,&yn1,&zn1);
    GetGridCartPosition(nvec[nb],                  &xn2,&yn2,&zn2);
    GetGridCartPosition(nvec[(nb + 1)%nnum],       &xn3,&yn3,&zn3);
    Real xc1, yc1, zc1;
    Real xc2, yc2, zc2;
    CircumcenterNormalized(x0,xn1,xn2,y0,yn1,yn2,z0,zn1,zn2,&xc1,&yc1,&zc1);
    CircumcenterNormalized(x0,xn2,xn3,y0,yn2,yn3,z0,zn2,zn3,&xc2,&yc2,&zc2);
    Real scalprod_c1 = x0*xc1 + y0*yc1 + z0*zc1;
    Real scalprod_c2 = x0*xc2 + y0*yc2 + z0*zc2;
    Real scalprod_12 = xc1*xc2 + yc1*yc2 + zc1*zc2;
    Real numerator = fabs(x0*(yc1*zc2-yc2*zc1) +
                            y0*(xc2*zc1-xc1*zc2) +
                            z0*(xc1*yc2-yc1*xc2));
    Real denominator = 1.0+scalprod_c1+scalprod_c2+scalprod_12;
    weight += 2.0*atan(numerator/denominator);
    length[nb] = acos(scalprod_12);
  }
  if (nnum == 5) {
    length[5] = std::numeric_limits<Real>::quiet_NaN();
  }
  
  return weight;
}

void Radiation::GetGridCartPosition(int lm, Real *x, Real *y, Real *z)
const {
  auto nlevels_ = nlevels;
  auto amesh_normals_ = amesh_normals;
  auto ameshp_normals_ = ameshp_normals;
  int ibl0 =  lm / (2*nlevels_*nlevels_);
  int ibl1 = (lm % (2*nlevels_*nlevels_)) / (2*nlevels_);
  int ibl2 = (lm % (2*nlevels_*nlevels_)) % (2*nlevels_);
  if (ibl0 == 5) {
    *x = ameshp_normals_.h_view(ibl2, 0);
    *y = ameshp_normals_.h_view(ibl2, 1);
    *z = ameshp_normals_.h_view(ibl2, 2);
  } else {
    *x = amesh_normals_.h_view(ibl0,ibl1+1,ibl2+1,0);
    *y = amesh_normals_.h_view(ibl0,ibl1+1,ibl2+1,1);
    *z = amesh_normals_.h_view(ibl0,ibl1+1,ibl2+1,2);
  }
}

void Radiation::GetGridCartPositionMid(int lm, int nb, Real *x, Real *y, Real *z)
const {
  Real x1, y1, z1;
  Real x2, y2, z2;
  GetGridCartPosition(lm,&x1,&y1,&z1);
  GetGridCartPosition(nb,&x2,&y2,&z2);
  Real xm = 0.5*(x1+x2);
  Real ym = 0.5*(y1+y2);
  Real zm = 0.5*(z1+z2);
  Real norm = sqrt(SQR(xm)+SQR(ym)+SQR(zm));
  *x = xm/norm;
  *y = ym/norm;
  *z = zm/norm;
}

void Radiation::OptimalAngles(Real ang[2])
const {
  int nzeta = 200;
  int npsi = 200;
  Real maxangle = ArcLength(0,1);
  Real deltazeta = maxangle/nzeta;
  Real deltapsi = M_PI/npsi;
  Real zeta;
  Real psi;
  Real vx, vy, vz;
  Real vrx, vry, vrz;
  Real vmax = 0.0;
  for (int l=0; l<nzeta; ++l) {
    zeta = (l+1)*deltazeta;
    for (int k = 0; k < npsi; ++k) {
      psi = (k+1)*deltapsi;
      Real kx = - sin(psi);
      Real ky = cos(psi);
      Real vmin_curr = 1.0;
      for (int i=0; i<nangles; ++i) {
        GetGridCartPosition(i,&vx,&vy,&vz);
        vrx = vx*cos(zeta)+ky*vz*sin(zeta)+kx*(kx*vx+ky*vy)*(1.0-cos(zeta));
        vry = vy*cos(zeta)-kx*vz*sin(zeta)+ky*(kx*vx+ky*vy)*(1.0-cos(zeta));
        vrz = vz*cos(zeta)+(kx*vy-ky*vx)*sin(zeta);
        if (fabs(vrx) < vmin_curr) {
          vmin_curr = fabs(vrx);
        }
        if (fabs(vry) < vmin_curr) {
          vmin_curr = fabs(vry);
        }
        if (fabs(vrz) < vmin_curr) {
          vmin_curr = fabs(vrz);
        }
      }
      if (vmin_curr > vmax) {
        vmax = vmin_curr;
        ang[0] = zeta;
        ang[1] = psi;
      }
    }
  }
}

void Radiation::RotateGrid(Real zeta, Real psi)
{
  Real kx = -sin(psi);
  Real ky = cos(psi);
  Real vx, vy, vz;
  Real vrx, vry, vrz;
  int nlev = nlevels;
  auto amesh_normals_ = amesh_normals;
  auto ameshp_normals_ = ameshp_normals;
  for (int bl=0; bl<5; ++bl) {
    for (int l=0; l<nlev; ++l) {
      for (int m=0; m<2*nlev; ++m) {
        vx = amesh_normals_.h_view(bl, l+1, m+1, 0);
        vy = amesh_normals_.h_view(bl, l+1, m+1, 1);
        vz = amesh_normals_.h_view(bl, l+1, m+1, 2);
        vrx = (vx*cos(zeta)+ky*vz*sin(zeta)
               + kx*(kx*vx+ky*vy)*(1.0-cos(zeta)));
        vry = (vy*cos(zeta)-kx*vz*sin(zeta)
               + ky*(kx*vx+ky*vy)*(1.0-cos(zeta)));
        vrz = (vz*cos(zeta)+(kx*vy-ky*vx)*sin(zeta));
        amesh_normals_.h_view(bl, l+1, m+1, 0) = vrx;
        amesh_normals_.h_view(bl, l+1, m+1, 1) = vry;
        amesh_normals_.h_view(bl, l+1, m+1, 2) = vrz;
      }
    }
  }
  for (int pl=0; pl<2; ++pl) {
    vx = ameshp_normals_.h_view(pl, 0);
    vy = ameshp_normals_.h_view(pl, 1);
    vz = ameshp_normals_.h_view(pl, 2);
    vrx = vx*cos(zeta)+ky*vz*sin(zeta)+kx*(kx*vx+ky*vy)*(1.0-cos(zeta));
    vry = vy*cos(zeta)-kx*vz*sin(zeta)+ky*(kx*vx+ky*vy)*(1.0-cos(zeta));
    vrz = vz*cos(zeta)+(kx*vy-ky*vx)*sin(zeta);
    ameshp_normals_.h_view(pl, 0) = vrx;
    ameshp_normals_.h_view(pl, 1) = vrx;
    ameshp_normals_.h_view(pl, 2) = vrx;
  }
 
  auto blocks_n = amesh_normals_;
  // TODO (@gnwong, @pdmullen) make this prettier?
  for (int i=0; i<3; ++i) {
    for (int bl=0; bl<5; ++bl) {
      for (int k=0; k<nlev; ++k) {
        blocks_n.h_view(bl,0,k+1,i)           = blocks_n.h_view((bl+4)%5,k+1,1,i);
        blocks_n.h_view(bl,0,k+nlev+1,i)      = blocks_n.h_view((bl+4)%5,nlev,k+1,i);
        blocks_n.h_view(bl,k+1,2*nlev+1,i)    = blocks_n.h_view((bl+4)%5,nlev,k+nlev+1,i);
        blocks_n.h_view(bl,k+2,0,i)           = blocks_n.h_view((bl+1)%5,1,k+1,i);
        blocks_n.h_view(bl,nlev+1,k+1,i)      = blocks_n.h_view((bl+1)%5,1,k+nlev+1,i);
        blocks_n.h_view(bl,nlev+1,k+nlev+1,i) = blocks_n.h_view((bl+1)%5,k+2,2*nlev,i);
      }
      blocks_n.h_view(bl,1,0,i)           = ameshp_normals_.h_view(0,i);
      blocks_n.h_view(bl,nlev+1,2*nlev,i) = ameshp_normals_.h_view(1,i);
      blocks_n.h_view(bl,0,2*nlev+1,i)    = blocks_n.h_view(bl,0,2*nlev,i);
    }
  }
}

void Radiation::ComputeXiEta(int lm, Real xi[6], Real eta[6])
const {
  Real x0, y0, z0;
  GetGridCartPosition(lm, &x0,&y0,&z0);
  int nvec[6];
  int nn = GetNeighbors(lm, nvec);
  Real a_angle = 0;
  for (int nb = 0; nb < nn; ++nb) {
    Real xn1, yn1, zn1;
    Real xn2, yn2, zn2;
    GetGridCartPosition(nvec[nb],         &xn1,&yn1,&zn1);
    GetGridCartPosition(nvec[(nb + 1)%nn],&xn2,&yn2,&zn2);
    Real n1_x = y0*zn1 - yn1*z0;
    Real n1_y = z0*xn1 - zn1*x0;
    Real n1_z = x0*yn1 - xn1*y0;
    Real n2_x = y0*zn2 - yn2*z0;
    Real n2_y = z0*xn2 - zn2*x0;
    Real n2_z = x0*yn2 - xn2*y0;
    Real norm1 = sqrt(SQR(n1_x)+SQR(n1_y)+SQR(n1_z));
    Real norm2 = sqrt(SQR(n2_x)+SQR(n2_y)+SQR(n2_z));
    Real cos_a = fmin((n1_x*n2_x+n1_y*n2_y+n1_z*n2_z)/(norm1*norm2),1.0);
    Real scalprod_c1 = x0*xn1 + y0*yn1 + z0*zn1;
    Real c_len = acos(scalprod_c1);
    xi[nb] = c_len*cos(a_angle);
    eta[nb] = c_len*sin(a_angle);
    a_angle += acos(cos_a);
  }
  if (nn==5) {
    xi[5] = std::numeric_limits<Real>::quiet_NaN();
    eta[5] = std::numeric_limits<Real>::quiet_NaN();
  }
}

Real Radiation::ArcLength(int ic1, int ic2)
const {
  Real x1, y1, z1;
  GetGridCartPosition(ic1,&x1,&y1,&z1);
  Real x2, y2, z2;
  GetGridCartPosition(ic2,&x2,&y2,&z2);
  return acos(x1*x2+y1*y2+z1*z2);
}

void Radiation::CircumcenterNormalized(Real x1, Real x2, Real x3,
                                       Real y1, Real y2, Real y3,
                                       Real z1, Real z2, Real z3,
                                       Real *x, Real *y, Real *z)
const {
  Real a = sqrt((x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+(z3-z2)*(z3-z2));
  Real b = sqrt((x1-x3)*(x1-x3)+(y1-y3)*(y1-y3)+(z1-z3)*(z1-z3));
  Real c = sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
  Real denom = 1.0/((a+c+b)*(a+c-b)*(a+b-c)*(b+c-a));
  Real x_c = (x1*SQR(a)*(SQR(b)+SQR(c)-SQR(a))+x2*SQR(b)*(SQR(c)+SQR(a)-SQR(b))
              + x3*SQR(c)*(SQR(a)+SQR(b)-SQR(c)))*denom;
  Real y_c = (y1*SQR(a)*(SQR(b)+SQR(c)-SQR(a))+y2*SQR(b)*(SQR(c)+SQR(a)-SQR(b))
              + y3*SQR(c)*(SQR(a)+SQR(b)-SQR(c)))*denom;
  Real z_c = (z1*SQR(a)*(SQR(b)+SQR(c)-SQR(a))+z2*SQR(b)*(SQR(c)+SQR(a)-SQR(b))
              + z3*SQR(c)*(SQR(a)+SQR(b)-SQR(c)))*denom;
  Real norm_c = sqrt(SQR(x_c)+SQR(y_c)+SQR(z_c));
  *x = x_c/norm_c;
  *y = y_c/norm_c;
  *z = z_c/norm_c;
}

KOKKOS_INLINE_FUNCTION
void DeviceUnitFluxDir(int ic1, int ic2, int nlvl,
                       DualArray4D<Real> ah_norm,
                       DualArray2D<Real> ap_norm,
                       Real *dtheta, Real *dphi)
{
  Real x, y, z;
  DeviceGetGridCartPosition(ic1,nlvl,ah_norm,ap_norm,&x,&y,&z);
  Real zeta1 = acos(z);
  Real psi1 = atan2(y,x);

  Real xm, ym, zm;
  DeviceGetGridCartPositionMid(ic1,ic2,nlvl,ah_norm,ap_norm,&xm,&ym,&zm);
  Real zetam = acos(zm);
  Real psim = atan2(ym,xm);

  if (fabs(psim-psi1) < 1.0e-10 ||
      fabs(fabs(zm)-1) < 1.0e-10 ||
      fabs(fabs(cos(zeta1))-1) < 1.0e-10) {
    *dtheta = copysign(1.0,zetam-zeta1);
    *dphi = 0.0;
  } else {
    Real a_par, p_par;
    DeviceGreatCircleParam(zeta1,zetam,psi1,psim,&a_par,&p_par);
    Real zeta_deriv = (a_par*sin(psim-p_par)
                       / (1.0+a_par*a_par*cos(psim-p_par)*cos(psim-p_par)));
    Real denom = 1.0/sqrt(zeta_deriv*zeta_deriv+sin(zetam)*sin(zetam));
    Real signfactor = copysign(1.0,psim-psi1)*copysign(1.0,M_PI-fabs(psim-psi1));
    *dtheta = signfactor*zeta_deriv*denom;
    *dphi   = signfactor*denom;
  }
}

KOKKOS_INLINE_FUNCTION
void DeviceGetGridCartPosition(int lm, int nlvl,
                               DualArray4D<Real> ah_norm,
                               DualArray2D<Real> ap_norm,
                               Real *x, Real *y, Real *z)
{
  int ibl0 =  lm / (2*nlvl*nlvl);
  int ibl1 = (lm % (2*nlvl*nlvl)) / (2*nlvl);
  int ibl2 = (lm % (2*nlvl*nlvl)) % (2*nlvl);
  if (ibl0 == 5) {
    *x = ap_norm.d_view(ibl2, 0);
    *y = ap_norm.d_view(ibl2, 1);
    *z = ap_norm.d_view(ibl2, 2);
  } else {
    *x = ah_norm.d_view(ibl0,ibl1+1,ibl2+1,0);
    *y = ah_norm.d_view(ibl0,ibl1+1,ibl2+1,1);
    *z = ah_norm.d_view(ibl0,ibl1+1,ibl2+1,2);
  }
}

KOKKOS_INLINE_FUNCTION
void DeviceGetGridCartPositionMid(int lm, int nb, int nlvl,
                                  DualArray4D<Real> ah_norm,
                                  DualArray2D<Real> ap_norm,
                                  Real *x, Real *y, Real *z)
{
  Real x1, y1, z1;
  Real x2, y2, z2;
  DeviceGetGridCartPosition(lm,nlvl,ah_norm,ap_norm,&x1,&y1,&z1);
  DeviceGetGridCartPosition(nb,nlvl,ah_norm,ap_norm,&x2,&y2,&z2);
  Real xm = 0.5*(x1+x2);
  Real ym = 0.5*(y1+y2);
  Real zm = 0.5*(z1+z2);
  Real norm = sqrt(SQR(xm)+SQR(ym)+SQR(zm));
  *x = xm/norm;
  *y = ym/norm;
  *z = zm/norm;
}


KOKKOS_INLINE_FUNCTION
void DeviceGreatCircleParam(Real zeta1, Real zeta2, Real psi1, Real psi2,
                            Real *apar, Real *psi0)
{
  Real atilde = (sin(psi2)/tan(zeta1)-sin(psi1)/tan(zeta2))/sin(psi2-psi1);
  Real btilde = (cos(psi2)/tan(zeta1)-cos(psi1)/tan(zeta2))/sin(psi1-psi2);
  *psi0 = atan2(btilde, atilde);
  *apar = sqrt(atilde*atilde+btilde*btilde);
}

} // namespace radiation
