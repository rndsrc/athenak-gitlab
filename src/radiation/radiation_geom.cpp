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

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::InitMesh()
//! \brief Initialize angular mesh

void Radiation::InitAngularMesh() {

  Real SinAng = 2.0/std::sqrt(5.0);
  Real CosAng = 1.0/std::sqrt(5.0);
  Real P1[3] = {0.0, 0.0, 1.0};
  Real P2[3] = {SinAng, 0.0, CosAng};
  Real P3[3] = {SinAng*std::cos(0.2*M_PI), SinAng*std::sin(0.2*M_PI), -CosAng};
  Real P4[3] = {SinAng*std::cos(-0.4*M_PI), SinAng*std::sin(-0.4*M_PI), CosAng};
  Real P5[3] = {SinAng*std::cos(-0.2*M_PI), SinAng*std::sin(-0.2*M_PI), -CosAng};
  Real P6[3] = {0.0, 0.0, -1.0};  

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

  for (int l=0; l<nlevels; ++l) {

    int col_index = 1;
    
    for (int m = l; m < nlevels; ++m){
      Real x = ((m-l+1)*P2[0] + (nlevels-m-1)*P1[0] + l*P4[0])/(float)(nlevels);
      Real y = ((m-l+1)*P2[1] + (nlevels-m-1)*P1[1] + l*P4[1])/(float)(nlevels);
      Real z = ((m-l+1)*P2[2] + (nlevels-m-1)*P1[2] + l*P4[2])/(float)(nlevels);
      
      Real norm = std::sqrt(x*x + y*y + z*z);

      amesh_normals_.h_view(0,row_index,col_index,0) = x/norm;
      amesh_normals_.h_view(0,row_index,col_index,1) = y/norm;
      amesh_normals_.h_view(0,row_index,col_index,2) = z/norm;

      col_index += 1;
    }
    
    for (int m = nlevels-l; m < nlevels; ++m){
      Real x = ((nlevels-l)*P2[0] + (m-nlevels+l+1)*P5[0] + (nlevels-m-1)*P4[0])/(float)(nlevels);
      Real y = ((nlevels-l)*P2[1] + (m-nlevels+l+1)*P5[1] + (nlevels-m-1)*P4[1])/(float)(nlevels);
      Real z = ((nlevels-l)*P2[2] + (m-nlevels+l+1)*P5[2] + (nlevels-m-1)*P4[2])/(float)(nlevels);
      
      Real norm = std::sqrt(x*x + y*y + z*z);
     
      amesh_normals_.h_view(0,row_index,col_index,0) = x/norm;
      amesh_normals_.h_view(0,row_index,col_index,1) = y/norm;
      amesh_normals_.h_view(0,row_index,col_index,2) = z/norm;
      
      col_index += 1;
    }
    
    for (int m = l; m < nlevels; ++m){
      Real x = ((m-l+1)*P3[0] + (nlevels-m-1)*P2[0] + l*P5[0])/(float)(nlevels);
      Real y = ((m-l+1)*P3[1] + (nlevels-m-1)*P2[1] + l*P5[1])/(float)(nlevels);
      Real z = ((m-l+1)*P3[2] + (nlevels-m-1)*P2[2] + l*P5[2])/(float)(nlevels);
      
      Real norm = std::sqrt(x*x + y*y + z*z);
     
      amesh_normals_.h_view(0,row_index,col_index,0) = x/norm;
      amesh_normals_.h_view(0,row_index,col_index,1) = y/norm;
      amesh_normals_.h_view(0,row_index,col_index,2) = z/norm;

      col_index += 1;
    }
    
    for (int m = nlevels-l; m < nlevels; ++m){
      Real x = ((nlevels-l)*P3[0] + (m-nlevels+l+1)*P6[0] + (nlevels-m-1)*P5[0])/(float)(nlevels);
      Real y = ((nlevels-l)*P3[1] + (m-nlevels+l+1)*P6[1] + (nlevels-m-1)*P5[1])/(float)(nlevels);
      Real z = ((nlevels-l)*P3[2] + (m-nlevels+l+1)*P6[2] + (nlevels-m-1)*P5[2])/(float)(nlevels);
      
      Real norm = std::sqrt(x*x + y*y + z*z);
     
     amesh_normals_.h_view(0,row_index,col_index,0) = x/norm;
     amesh_normals_.h_view(0,row_index,col_index,1) = y/norm;
     amesh_normals_.h_view(0,row_index,col_index,2) = z/norm;

      col_index += 1;
    }
    
    row_index += 1;
  }

  // now fill the other four patches by rotating the first one. only set 
  // the internal (non-ghost) values
  for (int patch = 1; patch < 5; ++patch){
    for (int l=1; l<1+nlevels; ++l) {
      for (int m=1; m<1+2*nlevels; ++m) {

        Real x0 = amesh_normals_.h_view(0,l,m,0);
        Real y0 = amesh_normals_.h_view(0,l,m,1);
        Real z0 = amesh_normals_.h_view(0,l,m,2);

        amesh_normals_.h_view(patch,l,m,0) = x0 * std::cos(patch*0.4*M_PI) + y0 * std::sin(patch*0.4*M_PI);
        amesh_normals_.h_view(patch,l,m,1) = y0 * std::cos(patch*0.4*M_PI) - x0 * std::sin(patch*0.4*M_PI);
        amesh_normals_.h_view(patch,l,m,2) = z0;
      }
    }
  }

  // TODO maybe figure out how to make this a neat function / remove entirely
  {
    auto blocks = amesh_normals_;
    auto nlev = nlevels;
    for (int i=0; i<3; ++i) { 
      for (int bl=0; bl<5; ++bl){
        for (int k=0; k<nlev; ++k){
          blocks.h_view(bl,0,k+1,i)           = blocks.h_view((bl+4)%5,k+1,1,i);
          blocks.h_view(bl,0,k+nlev+1,i)      = blocks.h_view((bl+4)%5,nlev,k+1,i);
          blocks.h_view(bl,k+1,2*nlev+1,i)    = blocks.h_view((bl+4)%5,nlev,k+nlev+1,i);
          blocks.h_view(bl,k+2,0,i)           = blocks.h_view((bl+1)%5,1,k+1,i);
          blocks.h_view(bl,nlev+1,k+1,i)      = blocks.h_view((bl+1)%5,1,k+nlev+1,i);
          blocks.h_view(bl,nlev+1,k+nlev+1,i) = blocks.h_view((bl+1)%5,k+2,2*nlev,i);
        }
        blocks.h_view(bl,1,0,i)           = ameshp_normals_.h_view(0,i);
        blocks.h_view(bl,nlev+1,2*nlev,i) = ameshp_normals_.h_view(1,i);
        blocks.h_view(bl,0,2*nlev+1,i)    = blocks.h_view(bl,0,2*nlev,i);
      }
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
    for (int l = 0; l < nlevels; ++l){
      for (int m = 0; m < 2*nlevels; ++m){
        // set center (non-ghost) values
        amesh_indices_.h_view(patch,l+1,m+1) = patch*2*nlevels*nlevels + l*2*nlevels + m;
      }
    }
  }

  ameshp_indices_.h_view(0) = 5*2*nlevels*nlevels;
  ameshp_indices_.h_view(1) = 5*2*nlevels*nlevels + 1;

  // TODO maybe figure out how to make this a neat function / remove entirely
  {
    auto blocks = amesh_indices_;
    auto nlev = nlevels;
    for (int bl=0; bl<5; ++bl){
      for (int k=0; k<nlev; ++k){
        blocks.h_view(bl,0,k+1)           = blocks.h_view((bl+4)%5,k+1,1);
        blocks.h_view(bl,0,k+nlev+1)      = blocks.h_view((bl+4)%5,nlev,k+1);
        blocks.h_view(bl,k+1,2*nlev+1)    = blocks.h_view((bl+4)%5,nlev,k+nlev+1);
        blocks.h_view(bl,k+2,0)           = blocks.h_view((bl+1)%5,1,k+1);
        blocks.h_view(bl,nlev+1,k+1)      = blocks.h_view((bl+1)%5,1,k+nlev+1);
        blocks.h_view(bl,nlev+1,k+nlev+1) = blocks.h_view((bl+1)%5,k+2,2*nlev);
      }
      blocks.h_view(bl,1,0)           = ameshp_indices_.h_view(0);
      blocks.h_view(bl,nlev+1,2*nlev) = ameshp_indices_.h_view(1);
      blocks.h_view(bl,0,2*nlev+1)    = blocks.h_view(bl,0,2*nlev);
    }
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

    double dual_edge[6];
    int neighbors[6];

    solid_angle_.h_view(lm) = ComputeWeightAndDualEdges(lm, dual_edge);
    num_neighbors_.h_view(lm) = GetNeighbors(lm, neighbors);

    for (int nb=0; nb<6; ++nb) {
      ind_neighbors_.h_view(lm, nb) = neighbors[nb];  // TODO is it necessary to save this information?
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
    double xi_coord[6];
    double eta_coord[6];

    ComputeXiEta(lm, xi_coord, eta_coord);

    for(int nb = 0; nb < 6; ++nb){
      xi_mn_.h_view(lm,nb) = xi_coord[nb];
      eta_mn_.h_view(lm,nb) = eta_coord[nb];
    }
  }

  xi_mn_.template modify<HostMemSpace>();
  eta_mn_.template modify<HostMemSpace>();

  xi_mn_.template sync<DevExeSpace>();
  eta_mn_.template sync<DevExeSpace>();

  // TODO FIXME make this prettier
  double rotangles[2];
  OptimalAngles(rotangles);
  RotateGrid(rotangles[0], rotangles[1]);

  auto nh_c_ = nh_c;
  auto nh_f_ = nh_f;

  for (int lm=0; lm<nangles; ++lm) {

    double x,y,z;
    GetGridCartPosition(lm, &x,&y,&z);
    
    nh_c_.h_view(lm,0) = 1.0;
    nh_c_.h_view(lm,1) = x;
    nh_c_.h_view(lm,2) = y;
    nh_c_.h_view(lm,3) = z;
   
    int nn = num_neighbors_.h_view(lm);
    for (int nb = 0; nb < nn; ++nb){
      double xm, ym, zm;

      GetGridCartPositionMid(lm, ind_neighbors_.h_view(lm,nb), &xm,&ym,&zm);
      
      nh_f_.h_view(lm,nb,0) = 1.0;
      nh_f_.h_view(lm,nb,1) = xm;
      nh_f_.h_view(lm,nb,2) = ym;
      nh_f_.h_view(lm,nb,3) = zm;
    }
    
    if (nn == 5) {
      nh_f_.h_view(lm,5,0) = std::numeric_limits<double>::quiet_NaN();
      nh_f_.h_view(lm,5,1) = std::numeric_limits<double>::quiet_NaN();
      nh_f_.h_view(lm,5,2) = std::numeric_limits<double>::quiet_NaN();
      nh_f_.h_view(lm,5,3) = std::numeric_limits<double>::quiet_NaN();
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

void Radiation::InitCoordinateFrame() {
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;

  int &nmb = pmy_pack->nmb_thispack;
  auto coord = pmy_pack->coord.coord_data;

  auto nmu_ = nmu;
  auto n0_n_mu_ = n0_n_mu;
  auto nh_c_ = nh_c;
  auto nh_f_ = nh_f;

  par_for("rad_n0_n_0", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
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

      for (int lm=0; lm<nangles; ++lm) {
        Real n0 = 0.;
        Real n1 = 0.;
        Real n2 = 0.;
        Real n3 = 0.;
        Real n_0 = 0.;
        Real n_1 = 0.;
        Real n_2 = 0.;
        Real n_3 = 0.;
        for (int d=0; d<4; ++d) {
          n0 += e[d][0] * nh_c_.d_view(lm,d);
          n1 += e[d][1] * nh_c_.d_view(lm,d);
          n2 += e[d][2] * nh_c_.d_view(lm,d);
          n3 += e[d][3] * nh_c_.d_view(lm,d);
          n_0 += e_cov[d][0] * nh_c_.d_view(lm,d);
          n_1 += e_cov[d][1] * nh_c_.d_view(lm,d);
          n_2 += e_cov[d][2] * nh_c_.d_view(lm,d);
          n_3 += e[d][3] * nh_c_.d_view(lm,d);
        }
        nmu_(m,lm,k,j,i,0) = n0;
        nmu_(m,lm,k,j,i,1) = n1;
        nmu_(m,lm,k,j,i,2) = n2;
        nmu_(m,lm,k,j,i,3) = n3;
        n0_n_mu_(m,lm,k,j,i,0) = n0 * n_0;
        n0_n_mu_(m,lm,k,j,i,1) = n0 * n_1;
        n0_n_mu_(m,lm,k,j,i,2) = n0 * n_2;
        n0_n_mu_(m,lm,k,j,i,3) = n0 * n_3;
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

      for (int lm=0; lm<nangles; ++lm) {
        Real n1 = 0.0;
        Real n_0 = 0.0;
        for (int d = 0; d < 4; ++d) {
          n1 += e[d][1] * nh_c_.d_view(lm,d);
          n_0 += e_cov[d][0] * nh_c_.d_view(lm,d);
        }
        n1_n_0_(m,lm,k,j,i) = n1 * n_0;
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

      for (int lm=0; lm<nangles; ++lm) {
        Real n2 = 0.0;
        Real n_0 = 0.0;
        for (int d = 0; d < 4; ++d) {
          n2 += e[d][2] * nh_c_.d_view(lm,d);
          n_0 += e_cov[d][0] * nh_c_.d_view(lm,d);
        }
        n2_n_0_(m,lm,k,j,i) = n2 * n_0;
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

      for (int lm=0; lm<nangles; ++lm) {
        Real n3 = 0.0;
        Real n_0 = 0.0;
        for (int d = 0; d < 4; ++d) {
          n3 += e[d][3] * nh_c_.d_view(lm,d);
          n_0 += e_cov[d][0] * nh_c_.d_view(lm,d);
        }
        n3_n_0_(m,lm,k,j,i) = n3 * n_0;
      }
    }
  );

  // Calculate n^angle n_0
  auto num_neighbors_ = num_neighbors;
  auto ind_neighbors_ = ind_neighbors;
  auto na_n_0_ = na_n_0;

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

      for (int lm=0; lm<nangles; ++lm) {
        for (int nb=0; nb<num_neighbors_.d_view(lm); ++nb) {

            Real zeta_f = std::acos(nh_f_.d_view(lm,nb,3));
            Real psi_f  = std::atan2(nh_f_.d_view(lm,nb,2), nh_f_.d_view(lm,nb,1));
        
            Real na1 = 0.0;
            Real na2 = 0.0;
            for (int n = 0; n < 4; ++n) {
              for (int p = 0; p < 4; ++p) {
                na1 += 1.0 / std::sin(zeta_f) * nh_f_.d_view(lm,nb,n) * nh_f_.d_view(lm,nb,p) 
                    * (nh_f_.d_view(lm,nb,0) * omega[3][n][p] - nh_f_.d_view(lm,nb,3) * omega[0][n][p]);
                na2 += 1.0 / SQR(std::sin(zeta_f)) * nh_f_.d_view(lm,nb,n) * nh_f_.d_view(lm,nb,p)
                    * (nh_f_.d_view(lm,nb,2) * omega[1][n][p] - nh_f_.d_view(lm,nb,1) * omega[n][p][2]);
              }
            }
            
            Real unit_zeta, unit_psi;
            UnitFluxDir(lm,ind_neighbors_.d_view(lm,nb),&unit_zeta,&unit_psi);
            
            Real na = na1*unit_zeta + na2*unit_psi;
          
            Real n_0 = 0.0;
            for (int n = 0; n < 4; ++n) {
              n_0 += e_cov[n][0] * nh_f_.d_view(lm,nb,n);
            }
            na_n_0_(m,lm,k,j,i,nb) = na * n_0;

        }
      }

    }
  );
  
  return;
}

// TODO FIXME implement new function that returns num neighbors using if statements 
// instead of also computing the neighbors too
int Radiation::GetNeighbors(int lm, int neighbors[6]) const {
  int num_neighbors;

  auto nlev_ = nlevels;
  auto amesh_indices_ = amesh_indices;
 
  // handle north pole 
  if (lm == 10*nlev_*nlev_){

    for(int bl = 0; bl < 5; ++bl){
      neighbors[bl] = amesh_indices_.d_view(bl,1,1);
    }

    neighbors[5] = NOT_A_PATCH;
    num_neighbors = 5;

  // handle south pole
  } else if (lm == 10*nlev_*nlev_ + 1){

    for(int bl = 0; bl < 5; ++bl){
      neighbors[bl] = amesh_indices_.d_view(bl,nlev_,2*nlev_);
    }

    neighbors[5] = NOT_A_PATCH;
    num_neighbors = 5;
    
  } else {
  
    int ibl0 =  lm / (2*nlev_*nlev_);
    int ibl1 = (lm % (2*nlev_*nlev_)) / (2*nlev_);
    int ibl2 = (lm % (2*nlev_*nlev_)) % (2*nlev_);
  
    neighbors[0] = amesh_indices_.d_view(ibl0, ibl1+1, ibl2+2);
    neighbors[1] = amesh_indices_.d_view(ibl0, ibl1+2, ibl2+1);
    neighbors[2] = amesh_indices_.d_view(ibl0, ibl1+2, ibl2);
    neighbors[3] = amesh_indices_.d_view(ibl0, ibl1+1, ibl2);
    neighbors[4] = amesh_indices_.d_view(ibl0, ibl1  , ibl2+1);
   
    // TODO check carefully, see if it can be inline optimized 
    if (lm % (2*nlev_*nlev_) == nlev_-1 || lm % (2*nlev_*nlev_) == 2*nlev_-1){
      neighbors[5] = NOT_A_PATCH;
      num_neighbors = 5;
    } else {
      neighbors[5] = amesh_indices_.d_view(ibl0, ibl1, ibl2+2);
      num_neighbors = 6;
    }
    
  }

  return num_neighbors;
}

double Radiation::ComputeWeightAndDualEdges(int lm, double length[6]) const {
  // TODO FIXME: how safe? assert n >= 0 && n < numverts?

  int nvec[6];
  int nnum = GetNeighbors(lm, nvec);
  
  double x0, y0, z0;
  GetGridCartPosition(lm, &x0,&y0,&z0);

  //fprintf(stderr, "%d -> %g %g %g %d\n", lm, x0,y0,z0, nnum);
  
  double weight = 0.0;
  
  for (int nb = 0; nb < nnum; ++nb){
  
    double xn1, yn1, zn1;
    double xn2, yn2, zn2;
    double xn3, yn3, zn3;
    
    GetGridCartPosition(nvec[(nb + nnum - 1)%nnum],&xn1,&yn1,&zn1);
    GetGridCartPosition(nvec[nb],                  &xn2,&yn2,&zn2);
    GetGridCartPosition(nvec[(nb + 1)%nnum],       &xn3,&yn3,&zn3);
    
    double xc1, yc1, zc1;
    double xc2, yc2, zc2;
    
    CircumcenterNormalized(x0,xn1,xn2,y0,yn1,yn2,z0,zn1,zn2,&xc1,&yc1,&zc1);
    CircumcenterNormalized(x0,xn2,xn3,y0,yn2,yn3,z0,zn2,zn3,&xc2,&yc2,&zc2);
    
    double scalprod_c1 = x0*xc1 + y0*yc1 + z0*zc1;
    double scalprod_c2 = x0*xc2 + y0*yc2 + z0*zc2;
    double scalprod_12 = xc1*xc2 + yc1*yc2 + zc1*zc2;
            
    double numerator = std::abs(x0*(yc1*zc2-yc2*zc1)+y0*(xc2*zc1-xc1*zc2)+z0*(xc1*yc2-yc1*xc2));
    double denominator = 1.0+scalprod_c1+scalprod_c2+scalprod_12;
        
    weight += 2.0*std::atan(numerator/denominator);
    
    length[nb] = std::acos(scalprod_12);
  
  }
  
  if (nnum == 5){
    length[5] = std::numeric_limits<double>::quiet_NaN();
  }
  
  return weight;
}

void Radiation::GetGridCartPosition(int lm, double *x, double *y, double *z) const {
  // TODO FIXME: should we assert n >= 0 and n < numverties? how safe are we?

  auto nlevels_ = nlevels;

  auto amesh_normals_ = amesh_normals;
  auto ameshp_normals_ = ameshp_normals;

  int ibl0 =  lm / (2*nlevels_*nlevels_);
  int ibl1 = (lm % (2*nlevels_*nlevels_)) / (2*nlevels_);
  int ibl2 = (lm % (2*nlevels_*nlevels_)) % (2*nlevels_);
  
  if (ibl0 == 5){ 
    *x = ameshp_normals_.d_view(ibl2, 0);  // TODO is it okay for this to be [dh]_view?
    *y = ameshp_normals_.d_view(ibl2, 1);
    *z = ameshp_normals_.d_view(ibl2, 2);
  } else {
    *x = amesh_normals_.d_view(ibl0,ibl1+1,ibl2+1,0);
    *y = amesh_normals_.d_view(ibl0,ibl1+1,ibl2+1,1);
    *z = amesh_normals_.d_view(ibl0,ibl1+1,ibl2+1,2);
  }
}

void Radiation::GetGridCartPositionMid(int lm, int nb, double *x, double *y, double *z) const {
  // TODO FIXME: should we assert?  (n good, nb good, (n,nb) are neighbors)

  double x1, y1, z1;
  double x2, y2, z2;
  
  GetGridCartPosition(lm,&x1,&y1,&z1);
  GetGridCartPosition(nb,&x2,&y2,&z2);
  
  double xm = 0.5*(x1+x2);
  double ym = 0.5*(y1+y2);
  double zm = 0.5*(z1+z2);
      
  double norm = std::sqrt(xm*xm+ym*ym+zm*zm);
  
  *x = xm/norm;
  *y = ym/norm;
  *z = zm/norm;
}

void Radiation::CircumcenterNormalized(double x1, double x2, double x3, 
                                       double y1, double y2, double y3, 
                                       double z1, double z2, double z3, 
                                       double *x, double *y, double *z) const {
  double a = std::sqrt((x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+(z3-z2)*(z3-z2));
  double b = std::sqrt((x1-x3)*(x1-x3)+(y1-y3)*(y1-y3)+(z1-z3)*(z1-z3));
  double c = std::sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
  double denom = 1.0/((a+c+b)*(a+c-b)*(a+b-c)*(b+c-a));
  double x_c = (x1*a*a*(b*b+c*c-a*a)+x2*b*b*(c*c+a*a-b*b)+x3*c*c*(a*a+b*b-c*c))*denom;
  double y_c = (y1*a*a*(b*b+c*c-a*a)+y2*b*b*(c*c+a*a-b*b)+y3*c*c*(a*a+b*b-c*c))*denom;
  double z_c = (z1*a*a*(b*b+c*c-a*a)+z2*b*b*(c*c+a*a-b*b)+z3*c*c*(a*a+b*b-c*c))*denom;
  double norm_c = std::sqrt(x_c*x_c+y_c*y_c+z_c*z_c);
  *x = x_c/norm_c;
  *y = y_c/norm_c;
  *z = z_c/norm_c;
}


void Radiation::GetGridPositionPolar(int ic, double * theta, double * phi) const {
  //assert(ic >= 0 && ic < NumVertices());
  // TODO REWRITE ?
  
  double x_, y_, z_;
  GetGridCartPosition(ic,&x_,&y_,&z_);
  
  *theta = std::acos(z_);
  *phi   = std::atan2(y_,x_);
}

void Radiation::GreatCircleParam(double zeta1, double zeta2, double psi1, double psi2, double * apar, double * psi0) const {
  Real atilde = (std::sin(psi2)/std::tan(zeta1)-std::sin(psi1)/std::tan(zeta2))/std::sin(psi2-psi1);
  Real btilde = (std::cos(psi2)/std::tan(zeta1)-std::cos(psi1)/std::tan(zeta2))/std::sin(psi1-psi2);
  *psi0 = std::atan2(btilde, atilde);
  *apar = std::sqrt(atilde*atilde+btilde*btilde);
}

void Radiation::UnitFluxDir(int ic1, int ic2, double * dtheta, double * dphi) const {
  //assert(ic1 >= 0 && ic1 < NumVertices());
  //assert(ic2 >= 0 && ic2 < NumVertices());
  //assert(AreNeighbors(ic1, ic2));
  
  double zeta1, psi1;
  
  GetGridPositionPolar(ic1,&zeta1,&psi1);
  
  double xm, ym, zm;
  GetGridCartPositionMid(ic1,ic2,&xm,&ym,&zm);
  
  double zetam = std::acos(zm);
  double psim  = std::atan2(ym,xm);
  
  double a_par, p_par;
  
  if (std::abs(psim-psi1) < 1.0e-10 || std::abs(std::abs(zm)-1) < 1.0e-10 || std::abs(std::abs(std::cos(zeta1))-1) < 1.0e-10){
    *dtheta = std::copysign(1.0,zetam-zeta1);
    *dphi = 0.0;
  }
  else{
    GreatCircleParam(zeta1,zetam,psi1,psim,&a_par,&p_par);
    double zeta_deriv = a_par*std::sin(psim-p_par)/(1.0+a_par*a_par*std::cos(psim-p_par)*std::cos(psim-p_par));
    double denom = 1.0/std::sqrt(zeta_deriv*zeta_deriv+std::sin(zetam)*std::sin(zetam));
    double signfactor = std::copysign(1.0,psim-psi1)*std::copysign(1.0,M_PI-std::abs(psim-psi1));
    *dtheta = signfactor * zeta_deriv * denom;
    *dphi   = signfactor * denom;
  }

}

void Radiation::OptimalAngles(double ang[2]) const {
  int nzeta = 200;
  int npsi = 200;
  
  double maxangle = ArcLength(0,1);

  double deltazeta = maxangle/nzeta;
  double deltapsi = M_PI/npsi;
  
  double zeta;
  double psi;
  
  double vx, vy, vz;
  double vrx, vry, vrz;

  double vmax = 0.0;
  
  for(int l = 0; l < nzeta; l++){
    zeta = (l+1)*deltazeta;
    for(int k = 0; k < npsi; k++){
      psi = (k+1)*deltapsi;
      
      double kx = - std::sin(psi);
      double ky = std::cos(psi);
      double vmin_curr = 1.0;
      
      for (int i = 0; i < nangles; ++i){
        GetGridCartPosition(i,&vx,&vy,&vz);
        vrx = vx*std::cos(zeta)+ky*vz*std::sin(zeta)+kx*(kx*vx+ky*vy)*(1.0-std::cos(zeta));
        vry = vy*std::cos(zeta)-kx*vz*std::sin(zeta)+ky*(kx*vx+ky*vy)*(1.0-std::cos(zeta));
        vrz = vz*std::cos(zeta)+(kx*vy-ky*vx)*std::sin(zeta);
        
        if(std::abs(vrx) < vmin_curr){
          vmin_curr = std::abs(vrx);
        }
        if(std::abs(vry) < vmin_curr){
          vmin_curr = std::abs(vry);
        }
        if(std::abs(vrz) < vmin_curr){
          vmin_curr = std::abs(vrz);
        }
      }
      
      if(vmin_curr > vmax){
        vmax = vmin_curr;
        ang[0] = zeta;
        ang[1] = psi;
      }
    
    }
  }
  
}

void Radiation::RotateGrid(double zeta, double psi) {
  double kx = -std::sin(psi);
  double ky = std::cos(psi);
  double vx, vy, vz;
  double vrx, vry, vrz;

  auto nlev_ = nlevels;
  auto amesh_normals_ = amesh_normals;
  auto ameshp_normals_ = ameshp_normals;
  
  for (int bl = 0; bl < 5; ++bl){
    for (int l = 0; l < nlev_; ++l){
      for (int m = 0; m < 2*nlev_; ++m){
        
        vx = amesh_normals_.h_view(bl, l+1, m+1, 0);
        vy = amesh_normals_.h_view(bl, l+1, m+1, 1);
        vz = amesh_normals_.h_view(bl, l+1, m+1, 2);

        vrx = vx*std::cos(zeta)+ky*vz*std::sin(zeta)+kx*(kx*vx+ky*vy)*(1.0-std::cos(zeta));
        vry = vy*std::cos(zeta)-kx*vz*std::sin(zeta)+ky*(kx*vx+ky*vy)*(1.0-std::cos(zeta));
        vrz = vz*std::cos(zeta)+(kx*vy-ky*vx)*std::sin(zeta);

        amesh_normals_.h_view(bl, l+1, m+1, 0) = vrx;
        amesh_normals_.h_view(bl, l+1, m+1, 1) = vry;
        amesh_normals_.h_view(bl, l+1, m+1, 2) = vrz;
      }
    }
  }
  for (int pl = 0; pl < 2; ++pl) {

    vx = ameshp_normals_.h_view(pl, 0);
    vy = ameshp_normals_.h_view(pl, 1);
    vz = ameshp_normals_.h_view(pl, 2);

    vrx = vx*std::cos(zeta)+ky*vz*std::sin(zeta)+kx*(kx*vx+ky*vy)*(1.0-std::cos(zeta));
    vry = vy*std::cos(zeta)-kx*vz*std::sin(zeta)+ky*(kx*vx+ky*vy)*(1.0-std::cos(zeta));
    vrz = vz*std::cos(zeta)+(kx*vy-ky*vx)*std::sin(zeta);
    
    ameshp_normals_.h_view(pl, 0) = vrx;
    ameshp_normals_.h_view(pl, 1) = vrx;
    ameshp_normals_.h_view(pl, 2) = vrx;
  }
 
  // TODO make this prettier?
  {
    auto blocks = amesh_normals_;
    auto nlev = nlevels;
    for (int i=0; i<3; ++i) {
      for (int bl=0; bl<5; ++bl){
        for (int k=0; k<nlev; ++k){
          blocks.h_view(bl,0,k+1,i)           = blocks.h_view((bl+4)%5,k+1,1,i);
          blocks.h_view(bl,0,k+nlev+1,i)      = blocks.h_view((bl+4)%5,nlev,k+1,i);
          blocks.h_view(bl,k+1,2*nlev+1,i)    = blocks.h_view((bl+4)%5,nlev,k+nlev+1,i);
          blocks.h_view(bl,k+2,0,i)           = blocks.h_view((bl+1)%5,1,k+1,i);
          blocks.h_view(bl,nlev+1,k+1,i)      = blocks.h_view((bl+1)%5,1,k+nlev+1,i);
          blocks.h_view(bl,nlev+1,k+nlev+1,i) = blocks.h_view((bl+1)%5,k+2,2*nlev,i);
        }
        blocks.h_view(bl,1,0,i)           = ameshp_normals_.h_view(0,i);
        blocks.h_view(bl,nlev+1,2*nlev,i) = ameshp_normals_.h_view(1,i);
        blocks.h_view(bl,0,2*nlev+1,i)    = blocks.h_view(bl,0,2*nlev,i);
      }
    }
  }
}

void Radiation::ComputeXiEta(int lm, double xi[6], double eta[6]) const {
  // TODO could assert here

  double x0, y0, z0;
  GetGridCartPosition(lm, &x0,&y0,&z0);
  
  int nvec[6];
  int nn = GetNeighbors(lm, nvec);
  
  double A_angle = 0;
  
  for (int nb = 0; nb < nn; ++nb){
  
    double xn1, yn1, zn1;
    double xn2, yn2, zn2;
    
    GetGridCartPosition(nvec[nb],         &xn1,&yn1,&zn1);
    GetGridCartPosition(nvec[(nb + 1)%nn],&xn2,&yn2,&zn2);
          
    double n1_x = y0*zn1 - yn1*z0;
    double n1_y = z0*xn1 - zn1*x0;
    double n1_z = x0*yn1 - xn1*y0;
 
    double n2_x = y0*zn2 - yn2*z0;
    double n2_y = z0*xn2 - zn2*x0;
    double n2_z = x0*yn2 - xn2*y0;

    double norm1 = std::sqrt(n1_x*n1_x+n1_y*n1_y+n1_z*n1_z);
    double norm2 = std::sqrt(n2_x*n2_x+n2_y*n2_y+n2_z*n2_z);

    double cosA = std::min((n1_x*n2_x+n1_y*n2_y+n1_z*n2_z)/(norm1*norm2),1.0);

    double scalprod_c1 = x0*xn1 + y0*yn1 + z0*zn1;

    double c_len = std::acos(scalprod_c1);
    
    xi[nb] = c_len*std::cos(A_angle);
    eta[nb] = c_len*std::sin(A_angle);
          
    A_angle += std::acos(cosA);
    
  }
  
  if (nn == 5){
    xi[5] = std::numeric_limits<double>::quiet_NaN();
    eta[5] = std::numeric_limits<double>::quiet_NaN();
  }

}


double Radiation::ArcLength(int ic1, int ic2) const {

  double x1, y1, z1;
  GetGridCartPosition(ic1,&x1,&y1,&z1);

  double x2, y2, z2;
  GetGridCartPosition(ic2,&x2,&y2,&z2);

  return std::acos(x1*x2+y1*y2+z1*z2);
}

} // namespace radiation
