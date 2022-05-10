//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_geom.cpp
//  \brief Initializes angular mesh and orthonormal tetrad

#include <math.h>
#include <float.h>
#include <limits.h>

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

namespace radiation {
//----------------------------------------------------------------------------------------
// inline functions for constructing geodesic mesh

KOKKOS_INLINE_FUNCTION
void GridCartPosition(int n, int nlvl,
                      DualArray4D<Real> anorm, DualArray2D<Real> apnorm,
                      Real *x, Real *y, Real *z);

KOKKOS_INLINE_FUNCTION
void GridCartPositionMid(int n, int nb, int nlvl,
                         DualArray4D<Real> anorm, DualArray2D<Real> apnorm,
                         Real *x, Real *y, Real *z);

KOKKOS_INLINE_FUNCTION
Real ComputeWeightAndDualEdges(int n, int nlvl, DualArray4D<Real> anorm,
                               DualArray2D<Real> apnorm, DualArray3D<Real> aind,
                               Real length[6]);

KOKKOS_INLINE_FUNCTION
void UnitFluxDir(Real zetav, Real psiv, Real zetaf, Real psif,
                 Real *dtheta, Real *dphi);

KOKKOS_INLINE_FUNCTION
int Neighbors(int n, int nlvl, DualArray3D<Real> aind, int neighbors[6]);

KOKKOS_INLINE_FUNCTION
void OptimalAngles(int nangles, int nlvl,
                   DualArray4D<Real> anorm, DualArray2D<Real> apnorm,
                   Real ang[2]);

KOKKOS_INLINE_FUNCTION
void RotateGrid(int nlvl, Real zeta, Real psi,
                DualArray4D<Real> anorm, DualArray2D<Real> apnorm);

KOKKOS_INLINE_FUNCTION
Real ArcLength(int ic1, int ic2, int nlvl,
               DualArray4D<Real> anorm, DualArray2D<Real> apnorm);

KOKKOS_INLINE_FUNCTION
void CircumcenterNormalized(Real x1, Real x2, Real x3,
                            Real y1, Real y2, Real y3,
                            Real z1, Real z2, Real z3,
                            Real *x, Real *y, Real *z);
KOKKOS_INLINE_FUNCTION
void GreatCircleParam(Real zeta1, Real zeta2, Real psi1, Real psi2,
                      Real *apar, Real *psi0);

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::InitMesh()
//! \brief Initialize angular mesh

void Radiation::InitAngularMesh() {
  // extract nlevel for angular mesh
  int lev_ = nlevel;

  if (lev_ > 0) {  // construct geodesic mesh
    Real sin_ang = 2.0/sqrt(5.0);
    Real cos_ang = 1.0/sqrt(5.0);
    Real p1[3] = {0.0, 0.0, 1.0};
    Real p2[3] = {sin_ang, 0.0, cos_ang};
    Real p3[3] = {sin_ang*cos( 0.2*M_PI), sin_ang*sin( 0.2*M_PI), -cos_ang};
    Real p4[3] = {sin_ang*cos(-0.4*M_PI), sin_ang*sin(-0.4*M_PI),  cos_ang};
    Real p5[3] = {sin_ang*cos(-0.2*M_PI), sin_ang*sin(-0.2*M_PI), -cos_ang};
    Real p6[3] = {0.0, 0.0, -1.0};

    // get coordinates of each face center, i.e., the normal component
    auto anorm_ = amesh_normals;
    auto apnorm_ = ameshp_normals;

    // start with poles, which we can set explicitly
    apnorm_.h_view(0,0) = 0.0;
    apnorm_.h_view(0,1) = 0.0;
    apnorm_.h_view(0,2) = 1.0;
    apnorm_.h_view(1,0) = 0.0;
    apnorm_.h_view(1,1) = 0.0;
    apnorm_.h_view(1,2) = -1.0;

    // now move on and start by filling in one of the five patches
    // we only fill in the center (ignoring ghost values)
    int row_index = 1;
    for (int l=0; l<lev_; ++l) {
      int col_index = 1;
      for (int m=l; m<lev_; ++m) {
        Real x = ((m-l+1)*p2[0] + (lev_-m-1)*p1[0] + l*p4[0])/(Real)(lev_);
        Real y = ((m-l+1)*p2[1] + (lev_-m-1)*p1[1] + l*p4[1])/(Real)(lev_);
        Real z = ((m-l+1)*p2[2] + (lev_-m-1)*p1[2] + l*p4[2])/(Real)(lev_);
        Real norm = sqrt(SQR(x) + SQR(y) + SQR(z));
        anorm_.h_view(0,row_index,col_index,0) = x/norm;
        anorm_.h_view(0,row_index,col_index,1) = y/norm;
        anorm_.h_view(0,row_index,col_index,2) = z/norm;
        col_index += 1;
      }
      for (int m=lev_-l; m<lev_; ++m) {
        Real x = ((lev_-l)*p2[0] + (m-lev_+l+1)*p5[0] + (lev_-m-1)*p4[0])/(Real)(lev_);
        Real y = ((lev_-l)*p2[1] + (m-lev_+l+1)*p5[1] + (lev_-m-1)*p4[1])/(Real)(lev_);
        Real z = ((lev_-l)*p2[2] + (m-lev_+l+1)*p5[2] + (lev_-m-1)*p4[2])/(Real)(lev_);
        Real norm = sqrt(SQR(x) + SQR(y) + SQR(z));
        anorm_.h_view(0,row_index,col_index,0) = x/norm;
        anorm_.h_view(0,row_index,col_index,1) = y/norm;
        anorm_.h_view(0,row_index,col_index,2) = z/norm;
        col_index += 1;
      }
      for (int m=l; m<lev_; ++m) {
        Real x = ((m-l+1)*p3[0] + (lev_-m-1)*p2[0] + l*p5[0])/(Real)(lev_);
        Real y = ((m-l+1)*p3[1] + (lev_-m-1)*p2[1] + l*p5[1])/(Real)(lev_);
        Real z = ((m-l+1)*p3[2] + (lev_-m-1)*p2[2] + l*p5[2])/(Real)(lev_);
        Real norm = sqrt(SQR(x) + SQR(y) + SQR(z));
        anorm_.h_view(0,row_index,col_index,0) = x/norm;
        anorm_.h_view(0,row_index,col_index,1) = y/norm;
        anorm_.h_view(0,row_index,col_index,2) = z/norm;
        col_index += 1;
      }
      for (int m=lev_-l; m<lev_; ++m) {
        Real x = ((lev_-l)*p3[0] + (m-lev_+l+1)*p6[0] + (lev_-m-1)*p5[0])/(Real)(lev_);
        Real y = ((lev_-l)*p3[1] + (m-lev_+l+1)*p6[1] + (lev_-m-1)*p5[1])/(Real)(lev_);
        Real z = ((lev_-l)*p3[2] + (m-lev_+l+1)*p6[2] + (lev_-m-1)*p5[2])/(Real)(lev_);
        Real norm = sqrt(SQR(x) + SQR(y) + SQR(z));
        anorm_.h_view(0,row_index,col_index,0) = x/norm;
        anorm_.h_view(0,row_index,col_index,1) = y/norm;
        anorm_.h_view(0,row_index,col_index,2) = z/norm;
        col_index += 1;
      }
      row_index += 1;
    }

    // now fill the other four patches by rotating the first one. only set
    // the internal (non-ghost) values
    for (int ptch=1; ptch<5; ++ptch) {
      for (int l=1; l<1+lev_; ++l) {
        for (int m=1; m<1+2*lev_; ++m) {
          Real x0 = anorm_.h_view(0,l,m,0);
          Real y0 = anorm_.h_view(0,l,m,1);
          Real z0 = anorm_.h_view(0,l,m,2);
          anorm_.h_view(ptch,l,m,0) = (x0*cos(ptch*0.4*M_PI)+y0*sin(ptch*0.4*M_PI));
          anorm_.h_view(ptch,l,m,1) = (y0*cos(ptch*0.4*M_PI)-x0*sin(ptch*0.4*M_PI));
          anorm_.h_view(ptch,l,m,2) = z0;
        }
      }
    }
    for (int i=0; i<3; ++i) {
      for (int bl=0; bl<5; ++bl) {
        for (int k=0; k<lev_; ++k) {
          anorm_.h_view(bl,0,     k+1,     i)=anorm_.h_view((bl+4)%5,k+1, 1,       i);
          anorm_.h_view(bl,0,     k+lev_+1,i)=anorm_.h_view((bl+4)%5,lev_,k+1,     i);
          anorm_.h_view(bl,k+1,   2*lev_+1,i)=anorm_.h_view((bl+4)%5,lev_,k+lev_+1,i);
          anorm_.h_view(bl,k+2,   0,       i)=anorm_.h_view((bl+1)%5,1,   k+1,     i);
          anorm_.h_view(bl,lev_+1,k+1,     i)=anorm_.h_view((bl+1)%5,1,   k+lev_+1,i);
          anorm_.h_view(bl,lev_+1,k+lev_+1,i)=anorm_.h_view((bl+1)%5,k+2, 2*lev_,  i);
        }
        anorm_.h_view(bl,1,     0,       i) = apnorm_.h_view(0,i);
        anorm_.h_view(bl,lev_+1,2*lev_,  i) = apnorm_.h_view(1,i);
        anorm_.h_view(bl,0,     2*lev_+1,i) = anorm_.h_view(bl,0,2*lev_,i);
      }
    }

    // generate 2d -> 1d map, seting center (non-ghost) values
    auto aind_ = amesh_indices;
    auto apind_ = ameshp_indices;
    for (int ptch=0; ptch<5; ++ptch) {
      for (int l=0; l<lev_; ++l) {
        for (int m=0; m<2*lev_; ++m) {
          aind_.h_view(ptch,l+1,m+1) = ptch*2*SQR(lev_) + l*2*lev_ + m;
        }
      }
    }

    // start with poles
    apind_.h_view(0) = 5*2*SQR(lev_);
    apind_.h_view(1) = 5*2*SQR(lev_) + 1;

    // move on to regular faces
    for (int bl=0; bl<5; ++bl) {
      for (int k=0; k<lev_; ++k) {
        aind_.h_view(bl,0,     k+1     ) = aind_.h_view((bl+4)%5,k+1, 1       );
        aind_.h_view(bl,0,     k+lev_+1) = aind_.h_view((bl+4)%5,lev_,k+1     );
        aind_.h_view(bl,k+1,   2*lev_+1) = aind_.h_view((bl+4)%5,lev_,k+lev_+1);
        aind_.h_view(bl,k+2,   0       ) = aind_.h_view((bl+1)%5,1,   k+1     );
        aind_.h_view(bl,lev_+1,k+1     ) = aind_.h_view((bl+1)%5,1,   k+lev_+1);
        aind_.h_view(bl,lev_+1,k+lev_+1) = aind_.h_view((bl+1)%5,k+2, 2*lev_  );
      }
      aind_.h_view(bl,1,     0       ) = apind_.h_view(0);
      aind_.h_view(bl,lev_+1,2*lev_  ) = apind_.h_view(1);
      aind_.h_view(bl,0,     2*lev_+1) = aind_.h_view(bl,0,2*lev_);
    }

    // set up geometric factors and neighbor information arrays
    auto num_neighbors_ = num_neighbors;
    auto ind_neighbors_ = ind_neighbors;
    auto solid_angle_ = solid_angle;
    auto arc_lengths_ = arc_lengths;
    for (int n=0; n<nangles; ++n) {
      Real dual_edge[6];
      int neighbors[6];
      num_neighbors_.h_view(n) = Neighbors(n,lev_,aind_,neighbors);
      solid_angle_.h_view(n) = ComputeWeightAndDualEdges(n,lev_,anorm_,apnorm_,
                                                         aind_,dual_edge);
      for (int nb=0; nb<6; ++nb) {
        ind_neighbors_.h_view(n,nb) = neighbors[nb];
        arc_lengths_.h_view(n,nb) = dual_edge[nb];
      }
    }

    // rotate geodesic mesh
    if (rotate_geo || angular_fluxes) {
      Real rotangles[2];
      OptimalAngles(nangles,lev_,anorm_,apnorm_,rotangles);
      RotateGrid(lev_,rotangles[0],rotangles[1],anorm_,apnorm_);
    }

    // set tetrad frame unit normal components
    auto nh_c_ = nh_c;
    auto nh_f_ = nh_f;
    for (int n=0; n<nangles; ++n) {
      Real x, y, z;
      GridCartPosition(n,lev_,anorm_,apnorm_,&x,&y,&z);
      nh_c_.h_view(n,0) = 1.0;
      nh_c_.h_view(n,1) = x;
      nh_c_.h_view(n,2) = y;
      nh_c_.h_view(n,3) = z;
      int nn = num_neighbors_.h_view(n);
      for (int nb=0; nb<nn; ++nb) {
        Real xm, ym, zm;
        GridCartPositionMid(n,ind_neighbors_.h_view(n,nb),lev_,anorm_,apnorm_,
                            &xm,&ym,&zm);
        nh_f_.h_view(n,nb,0) = 1.0;
        nh_f_.h_view(n,nb,1) = xm;
        nh_f_.h_view(n,nb,2) = ym;
        nh_f_.h_view(n,nb,3) = zm;
      }
      if (nn==5) {
        nh_f_.h_view(n,5,0) = (FLT_MAX);
        nh_f_.h_view(n,5,1) = (FLT_MAX);
        nh_f_.h_view(n,5,2) = (FLT_MAX);
        nh_f_.h_view(n,5,3) = (FLT_MAX);
      }
    }

    // guarantee that arc lengths at shared faces are identical
    for (int n=0; n<nangles; ++n) {
      for (int nb=0; nb<num_neighbors_.h_view(n); ++nb) {
        bool match_not_found = true;
        Real this_arc = arc_lengths_.h_view(n,nb);
        for (int nnb=0; nnb<num_neighbors_.h_view(ind_neighbors_.h_view(n,nb)); ++nnb) {
          Real neigh_arc = arc_lengths_.h_view(ind_neighbors_.h_view(n,nb),nnb);
          if (nh_f_.h_view(n,nb,1) == nh_f_.h_view(ind_neighbors_.h_view(n,nb),nnb,1) &&
              nh_f_.h_view(n,nb,2) == nh_f_.h_view(ind_neighbors_.h_view(n,nb),nnb,2) &&
              nh_f_.h_view(n,nb,3) == nh_f_.h_view(ind_neighbors_.h_view(n,nb),nnb,3)) {
            match_not_found = false;
            Real arc_avg = 0.5*(this_arc+neigh_arc);
            arc_lengths_.h_view(n,nb) = arc_avg;
            arc_lengths_.h_view(ind_neighbors_.h_view(n,nb),nnb) = arc_avg;
          }
        }
        if (match_not_found) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "Error in geodesic grid initialization" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
    }

    anorm_.template modify<HostMemSpace>();
    anorm_.template sync<DevExeSpace>();
    apnorm_.template modify<HostMemSpace>();
    apnorm_.template sync<DevExeSpace>();
    aind_.template modify<HostMemSpace>();
    aind_.template sync<DevExeSpace>();
    apind_.template modify<HostMemSpace>();
    apind_.template sync<DevExeSpace>();
    num_neighbors_.template modify<HostMemSpace>();
    num_neighbors_.template sync<DevExeSpace>();
    ind_neighbors_.template modify<HostMemSpace>();
    ind_neighbors_.template sync<DevExeSpace>();
    arc_lengths_.template modify<HostMemSpace>();
    arc_lengths_.template sync<DevExeSpace>();
    solid_angle_.template modify<HostMemSpace>();
    solid_angle_.template sync<DevExeSpace>();
    nh_c_.template modify<HostMemSpace>();
    nh_c_.template sync<DevExeSpace>();
    nh_f_.template modify<HostMemSpace>();
    nh_f_.template sync<DevExeSpace>();

  } else if (lev_==0) {  // one angle per octant mesh (only for testing)
    auto nh_c_ = nh_c;
    auto nh_f_ = nh_f;
    auto solid_angle_ = solid_angle;
    Real zetav[2] = {M_PI/4.0, 3.0*M_PI/4.0};
    Real psiv[4] = {M_PI/4.0, 3.0*M_PI/4.0, 5.0*M_PI/4.0, 7.0*M_PI/4.0};
    for (int z=0, n=0; z<2; ++z) {
      for (int p=0; p<4; ++p, ++n) {
        nh_c_.h_view(n,0) = 1.0;
        nh_c_.h_view(n,1) = sin(zetav[z])*cos(psiv[p])*sqrt(4.0/3.0);
        nh_c_.h_view(n,2) = sin(zetav[z])*sin(psiv[p])*sqrt(4.0/3.0);
        nh_c_.h_view(n,3) = cos(zetav[z])*sqrt(2.0/3.0);
        solid_angle_.h_view(n) = 4.0*M_PI/nangles;
      }
    }

    nh_c_.template modify<HostMemSpace>();
    nh_c_.template sync<DevExeSpace>();
    nh_f_.template modify<HostMemSpace>();
    nh_f_.template sync<DevExeSpace>();
    solid_angle_.template modify<HostMemSpace>();
    solid_angle_.template sync<DevExeSpace>();

  } else {  // invalid selection for <radiation>/nlevel
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "nlevel must be >= 0, "
        << "but <radiation>/nlevel=" << lev_ << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::SetOrthonormalTetrad()
//! \brief Set orthonormal tetrad data

void Radiation::SetOrthonormalTetrad() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  int &nmb = pmy_pack->nmb_thispack;
  auto &coord = pmy_pack->pcoord->coord_data;

  int nangles_ = nangles;
  int nlev_ = nlevel;

  auto nh_c_ = nh_c;
  auto nh_f_ = nh_f;
  auto num_neighbors_ = num_neighbors;
  auto ind_neighbors_ = ind_neighbors;

  // Calculate n^mu and n_mu
  auto nmu_ = nmu;
  auto n_mu_ = n_mu;
  par_for("rad_nl_nu",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
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

    Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
    ComputeTetrad(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin, e, e_cov, omega);
    for (int n=0; n<nangles_; ++n) {
      Real n0 = 0.0;
      Real n1 = 0.0;
      Real n2 = 0.0;
      Real n3 = 0.0;
      Real n_0 = 0.0;
      Real n_1 = 0.0;
      Real n_2 = 0.0;
      Real n_3 = 0.0;
      for (int d=0; d<4; ++d) {
        n0 += e[d][0]*nh_c_.d_view(n,d);
        n1 += e[d][1]*nh_c_.d_view(n,d);
        n2 += e[d][2]*nh_c_.d_view(n,d);
        n3 += e[d][3]*nh_c_.d_view(n,d);
        n_0 += e_cov[d][0]*nh_c_.d_view(n,d);
        n_1 += e_cov[d][1]*nh_c_.d_view(n,d);
        n_2 += e_cov[d][2]*nh_c_.d_view(n,d);
        n_3 += e_cov[d][3]*nh_c_.d_view(n,d);
      }
      nmu_(m,n,k,j,i,0) = n0;
      nmu_(m,n,k,j,i,1) = n1;
      nmu_(m,n,k,j,i,2) = n2;
      nmu_(m,n,k,j,i,3) = n3;
      n_mu_(m,n,k,j,i,0) = n_0;
      n_mu_(m,n,k,j,i,1) = n_1;
      n_mu_(m,n,k,j,i,2) = n_2;
      n_mu_(m,n,k,j,i,3) = n_3;
    }
  });

  // Calculate n^1 n_mu
  auto n1_n_0_ = n1_n_0;
  par_for("rad_n1_n_0",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,n1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1f = LeftEdgeX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
    ComputeTetrad(x1f, x2v, x3v, coord.is_minkowski, coord.bh_spin, e, e_cov, omega);
    for (int n=0; n<nangles_; ++n) {
      Real n1 = 0.0;
      Real n_0 = 0.0;
      for (int d=0; d<4; ++d) {
        n1 += e[d][1]*nh_c_.d_view(n,d);
        n_0 += e_cov[d][0]*nh_c_.d_view(n,d);
      }
      n1_n_0_(m,n,k,j,i) = n1*n_0;
    }
  });

  // Calculate n^2 n_mu
  auto n2_n_0_ = n2_n_0;
  par_for("rad_n2_n_0", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, n2, 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2f = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
    ComputeTetrad(x1v, x2f, x3v, coord.is_minkowski, coord.bh_spin, e, e_cov, omega);
    for (int n=0; n<nangles_; ++n) {
      Real n2 = 0.0;
      Real n_0 = 0.0;
      for (int d=0; d<4; ++d) {
        n2 += e[d][2]*nh_c_.d_view(n,d);
        n_0 += e_cov[d][0]*nh_c_.d_view(n,d);
      }
      n2_n_0_(m,n,k,j,i) = n2*n_0;
    }
  });

  // Calculate n^3 n_mu
  auto n3_n_0_ = n3_n_0;
  par_for("rad_n3_n_0", DevExeSpace(), 0, (nmb-1), 0, n3, 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3f = LeftEdgeX(k-ks, indcs.nx3, x3min, x3max);

    Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
    ComputeTetrad(x1v, x2v, x3f, coord.is_minkowski, coord.bh_spin, e, e_cov, omega);
    for (int n=0; n<nangles_; ++n) {
      Real n3 = 0.0;
      Real n_0 = 0.0;
      for (int d = 0; d < 4; ++d) {
        n3 += e[d][3]*nh_c_.d_view(n,d);
        n_0 += e_cov[d][0]*nh_c_.d_view(n,d);
      }
      n3_n_0_(m,n,k,j,i) = n3*n_0;
    }
  });

  // Calculate n^angle n_0
  auto na_n_0_ = na_n_0;
  if (nlev_ != 0) {  // do not compute na if nlevel=0
    par_for("rad_na_n_0", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
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

      Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
      ComputeTetrad(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin, e, e_cov, omega);
      for (int n=0; n<nangles_; ++n) {
        Real zetav = acos(nh_c_.d_view(n,3));
        Real psiv  = atan2(nh_c_.d_view(n,2), nh_c_.d_view(n,1));
        for (int nb=0; nb<num_neighbors_.d_view(n); ++nb) {
          Real zetaf = acos(nh_f_.d_view(n,nb,3));
          Real psif  = atan2(nh_f_.d_view(n,nb,2), nh_f_.d_view(n,nb,1));
          Real na1 = 0.0;
          Real na2 = 0.0;
          for (int q=0; q<4; ++q) {
            for (int p=0; p<4; ++p) {
              na1 += (1.0/sin(zetaf)*nh_f_.d_view(n,nb,q)*nh_f_.d_view(n,nb,p)
                      * (nh_f_.d_view(n,nb,0)*omega[3][q][p]
                      -  nh_f_.d_view(n,nb,3)*omega[0][q][p]));
              na2 += (1.0/SQR(sin(zetaf))*nh_f_.d_view(n,nb,q)*nh_f_.d_view(n,nb,p)
                      * (nh_f_.d_view(n,nb,2)*omega[1][q][p]
                      -  nh_f_.d_view(n,nb,1)*omega[2][q][p]));
            }
          }
          Real unit_zeta, unit_psi;
          UnitFluxDir(zetav,psiv,zetaf,psif,&unit_zeta,&unit_psi);
          Real na = na1*unit_zeta + SQR(sin(zetaf))*na2*unit_psi;
          Real n_0 = 0.0;
          for (int q=0; q<4; ++q) {
            n_0 += e_cov[q][0]*nh_f_.d_view(n,nb,q);
          }
          na_n_0_(m,n,k,j,i,nb) = na*n_0;
        }
      }

      // Guarantee that na_n_0 at shared faces are identical
      for (int n=0; n<nangles_; ++n) {
        for (int nb=0; nb<num_neighbors_.d_view(n); ++nb) {
          Real this_na_n_0 = na_n_0_(m,n,k,j,i,nb);
          for (int nnb=0; nnb<num_neighbors_.d_view(ind_neighbors_.d_view(n,nb)); ++nnb) {
            Real neigh_na_n_0 = na_n_0_(m,ind_neighbors_.d_view(n,nb),k,j,i,nnb);
            if (nh_f_.d_view(n,nb,1) == nh_f_.d_view(ind_neighbors_.d_view(n,nb),nnb,1) &&
                nh_f_.d_view(n,nb,2) == nh_f_.d_view(ind_neighbors_.d_view(n,nb),nnb,2) &&
                nh_f_.d_view(n,nb,3) == nh_f_.d_view(ind_neighbors_.d_view(n,nb),nnb,3)) {
              Real na_n_0_avg = 0.5*(fabs(this_na_n_0)+fabs(neigh_na_n_0));
              na_n_0_(m,n,k,j,i,nb) = copysign(na_n_0_avg, this_na_n_0);
              na_n_0_(m,ind_neighbors_.d_view(n,nb),k,j,i,nnb) = copysign(na_n_0_avg,
                                                                          neigh_na_n_0);
            }
          }
        }
      }
    });
  }

  // Calculate norm_to_tet
  auto norm_to_tet_ = norm_to_tet;
  par_for("rad_norm_to_tet", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
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

    Real g_[NMETRIC], gi_[NMETRIC];
    ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin, g_, gi_);
    Real e[4][4]; Real e_cov[4][4]; Real omega[4][4][4];
    ComputeTetrad(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin, e, e_cov, omega);

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
  });

  return;
}

// find position at face center
KOKKOS_INLINE_FUNCTION
void GridCartPosition(int n, int nlvl,
                      DualArray4D<Real> anorm, DualArray2D<Real> apnorm,
                      Real *x, Real *y, Real *z) {
  int ibl0 = (n / (2*nlvl*nlvl));
  int ibl1 = (n % (2*nlvl*nlvl)) / (2*nlvl);
  int ibl2 = (n % (2*nlvl*nlvl)) % (2*nlvl);
  if (ibl0 == 5) {
    *x = apnorm.h_view(ibl2, 0);
    *y = apnorm.h_view(ibl2, 1);
    *z = apnorm.h_view(ibl2, 2);
  } else {
    *x = anorm.h_view(ibl0,ibl1+1,ibl2+1,0);
    *y = anorm.h_view(ibl0,ibl1+1,ibl2+1,1);
    *z = anorm.h_view(ibl0,ibl1+1,ibl2+1,2);
  }
}

// find mid position between two face centers
KOKKOS_INLINE_FUNCTION
void GridCartPositionMid(int n, int nb, int nlvl,
                         DualArray4D<Real> anorm, DualArray2D<Real> apnorm,
                         Real *x, Real *y, Real *z) {
  Real x1, y1, z1, x2, y2, z2;
  GridCartPosition(n, nlvl,anorm,apnorm,&x1,&y1,&z1);
  GridCartPosition(nb,nlvl,anorm,apnorm,&x2,&y2,&z2);
  Real xm = 0.5*(x1+x2);
  Real ym = 0.5*(y1+y2);
  Real zm = 0.5*(z1+z2);
  Real norm = sqrt(SQR(xm)+SQR(ym)+SQR(zm));
  *x = xm/norm;
  *y = ym/norm;
  *z = zm/norm;
}

// inline function to retrieve weights (solid angles) and edge lengths
KOKKOS_INLINE_FUNCTION
Real ComputeWeightAndDualEdges(int n, int nlvl, DualArray4D<Real> anorm,
                               DualArray2D<Real> apnorm, DualArray3D<Real> aind,
                               Real length[6]) {
  int nvec[6];
  int nnum = Neighbors(n, nlvl, aind, nvec);
  Real x0, y0, z0;
  GridCartPosition(n,nlvl,anorm,apnorm,&x0,&y0,&z0);
  Real weight = 0.0;
  for (int nb=0; nb<nnum; ++nb) {
    Real xn1, yn1, zn1;
    Real xn2, yn2, zn2;
    Real xn3, yn3, zn3;
    GridCartPosition(nvec[(nb + nnum - 1)%nnum],nlvl,anorm,apnorm,&xn1,&yn1,&zn1);
    GridCartPosition(nvec[nb],                  nlvl,anorm,apnorm,&xn2,&yn2,&zn2);
    GridCartPosition(nvec[(nb + 1)%nnum],       nlvl,anorm,apnorm,&xn3,&yn3,&zn3);
    Real xc1, yc1, zc1;
    Real xc2, yc2, zc2;
    CircumcenterNormalized(x0,xn1,xn2,y0,yn1,yn2,z0,zn1,zn2,&xc1,&yc1,&zc1);
    CircumcenterNormalized(x0,xn2,xn3,y0,yn2,yn3,z0,zn2,zn3,&xc2,&yc2,&zc2);
    Real scalprod_c1 = x0 *xc1 + y0 *yc1 + z0 *zc1;
    Real scalprod_c2 = x0 *xc2 + y0 *yc2 + z0 *zc2;
    Real scalprod_12 = xc1*xc2 + yc1*yc2 + zc1*zc2;
    Real numerator = fabs(x0*(yc1*zc2-yc2*zc1) +
                          y0*(xc2*zc1-xc1*zc2) +
                          z0*(xc1*yc2-yc1*xc2));
    Real denominator = 1.0+scalprod_c1+scalprod_c2+scalprod_12;
    weight += 2.0*atan(numerator/denominator);
    length[nb] = acos(scalprod_12);
  }
  if (nnum == 5) {
    length[5] = (FLT_MAX);
  }

  return weight;
}

// inline function to find unit vector at face edge
KOKKOS_INLINE_FUNCTION
void UnitFluxDir(Real zetav, Real psiv, Real zetaf, Real psif,
                 Real *dtheta, Real *dphi) {
  if (fabs(psif-psiv) < 1.0e-10 ||
      fabs(fabs(cos(zetav))-1) < 1.0e-10) {
    *dtheta = (FLT_MAX);
    *dphi = (FLT_MAX);
  } else {
    Real a_par, p_par;
    GreatCircleParam(zetav,zetaf,psiv,psif,&a_par,&p_par);
    Real zeta_deriv = (a_par*sin(psif-p_par)
                       / (1.0+a_par*a_par*cos(psif-p_par)*cos(psif-p_par)));
    Real denom = 1.0/sqrt(zeta_deriv*zeta_deriv+sin(zetaf)*sin(zetaf));
    Real signfactor = copysign(1.0,psif-psiv)*copysign(1.0,M_PI-fabs(psif-psiv));
    *dtheta = signfactor*zeta_deriv*denom;
    *dphi   = signfactor*denom;
  }
}

// inline function to retrieve neighbors (and number of neighbors)
KOKKOS_INLINE_FUNCTION
int Neighbors(int n, int nlvl, DualArray3D<Real> aind, int neighbors[6]) {
  int num_neighbors;
  // handle north pole
  if (n==10*nlvl*nlvl) {
    for (int bl=0; bl<5; ++bl) {
      neighbors[bl] = aind.h_view(bl,1,1);
    }
    neighbors[5] = (INT_MAX);
    num_neighbors = 5;
  } else if (n==10*nlvl*nlvl + 1) {  // handle south pole
    for (int bl=0; bl<5; ++bl) {
      neighbors[bl] = aind.h_view(bl,nlvl,2*nlvl);
    }
    neighbors[5] = (INT_MAX);
    num_neighbors = 5;
  } else {
    int ibl0 = (n / (2*nlvl*nlvl));
    int ibl1 = (n % (2*nlvl*nlvl)) / (2*nlvl);
    int ibl2 = (n % (2*nlvl*nlvl)) % (2*nlvl);
    neighbors[0] = aind.h_view(ibl0, ibl1+1, ibl2+2);
    neighbors[1] = aind.h_view(ibl0, ibl1+2, ibl2+1);
    neighbors[2] = aind.h_view(ibl0, ibl1+2, ibl2);
    neighbors[3] = aind.h_view(ibl0, ibl1+1, ibl2);
    neighbors[4] = aind.h_view(ibl0, ibl1  , ibl2+1);

    // TODO(@gnwong, @pdmullen) check carefully, see if it can be inline optimized
    if (n % (2*nlvl*nlvl) == nlvl-1 || n % (2*nlvl*nlvl) == 2*nlvl-1) {
      neighbors[5] = (INT_MAX);
      num_neighbors = 5;
    } else {
      neighbors[5] = aind.h_view(ibl0, ibl1, ibl2+2);
      num_neighbors = 6;
    }
  }
  return num_neighbors;
}

// inline function to find an optimal angle by which to rotate the geodesic mesh
KOKKOS_INLINE_FUNCTION
void OptimalAngles(int nangles, int nlvl,
                   DualArray4D<Real> anorm, DualArray2D<Real> apnorm,
                   Real ang[2]) {
  int nzeta = 200;  // nzeta val inherited from Viktoriya Giryanskaya
  int npsi  = 200;  // npsi  val inherited from Viktoriya Giryanskaya
  Real maxangle = ArcLength(0,1,nlvl,anorm,apnorm);
  Real deltazeta = maxangle/nzeta;
  Real deltapsi = M_PI/npsi;
  Real zeta;
  Real psi;
  Real vx, vy, vz;
  Real vrx, vry, vrz;
  Real vmax = 0.0;
  for (int l=0; l<nzeta; ++l) {
    zeta = (l+1)*deltazeta;
    for (int k=0; k<npsi; ++k) {
      psi = (k+1)*deltapsi;
      Real kx = -sin(psi);
      Real ky =  cos(psi);
      Real vmin_curr = 1.0;
      for (int n=0; n<nangles; ++n) {
        GridCartPosition(n,nlvl,anorm,apnorm,&vx,&vy,&vz);
        vrx = vx*cos(zeta)+ky*vz*sin(zeta)+kx*(kx*vx+ky*vy)*(1.0-cos(zeta));
        vry = vy*cos(zeta)-kx*vz*sin(zeta)+ky*(kx*vx+ky*vy)*(1.0-cos(zeta));
        vrz = vz*cos(zeta)+(kx*vy-ky*vx)*sin(zeta);
        if (fabs(vrx) < vmin_curr) {vmin_curr = fabs(vrx);}
        if (fabs(vry) < vmin_curr) {vmin_curr = fabs(vry);}
        if (fabs(vrz) < vmin_curr) {vmin_curr = fabs(vrz);}
      }
      if (vmin_curr > vmax) {
        vmax = vmin_curr;
        ang[0] = zeta;
        ang[1] = psi;
      }
    }
  }
}

// inline function to rotate the geodesic mesh
KOKKOS_INLINE_FUNCTION
void RotateGrid(int nlvl, Real zeta, Real psi,
                DualArray4D<Real> anorm, DualArray2D<Real> apnorm) {
  Real kx = -sin(psi);
  Real ky =  cos(psi);
  Real vx, vy, vz, vrx, vry, vrz;
  for (int bl=0; bl<5; ++bl) {
    for (int l=0; l<nlvl; ++l) {
      for (int m=0; m<2*nlvl; ++m) {
        vx = anorm.h_view(bl, l+1, m+1, 0);
        vy = anorm.h_view(bl, l+1, m+1, 1);
        vz = anorm.h_view(bl, l+1, m+1, 2);
        vrx = (vx*cos(zeta)+ky*vz*sin(zeta) + kx*(kx*vx+ky*vy)*(1.0-cos(zeta)));
        vry = (vy*cos(zeta)-kx*vz*sin(zeta) + ky*(kx*vx+ky*vy)*(1.0-cos(zeta)));
        vrz = (vz*cos(zeta)+(kx*vy-ky*vx)*sin(zeta));
        anorm.h_view(bl,l+1,m+1,0) = vrx;
        anorm.h_view(bl,l+1,m+1,1) = vry;
        anorm.h_view(bl,l+1,m+1,2) = vrz;
      }
    }
  }
  for (int pl=0; pl<2; ++pl) {
    vx = apnorm.h_view(pl,0);
    vy = apnorm.h_view(pl,1);
    vz = apnorm.h_view(pl,2);
    vrx = vx*cos(zeta)+ky*vz*sin(zeta)+kx*(kx*vx+ky*vy)*(1.0-cos(zeta));
    vry = vy*cos(zeta)-kx*vz*sin(zeta)+ky*(kx*vx+ky*vy)*(1.0-cos(zeta));
    vrz = vz*cos(zeta)+(kx*vy-ky*vx)*sin(zeta);
    apnorm.h_view(pl,0) = vrx;
    apnorm.h_view(pl,1) = vry;
    apnorm.h_view(pl,2) = vrz;
  }
  for (int i=0; i<3; ++i) {
    for (int bl=0; bl<5; ++bl) {
      for (int k=0; k<nlvl; ++k) {
        anorm.h_view(bl,0,     k+1,     i) = anorm.h_view((bl+4)%5,k+1, 1,       i);
        anorm.h_view(bl,0,     k+nlvl+1,i) = anorm.h_view((bl+4)%5,nlvl,k+1,     i);
        anorm.h_view(bl,k+1,   2*nlvl+1,i) = anorm.h_view((bl+4)%5,nlvl,k+nlvl+1,i);
        anorm.h_view(bl,k+2,   0,       i) = anorm.h_view((bl+1)%5,1,   k+1,     i);
        anorm.h_view(bl,nlvl+1,k+1,     i) = anorm.h_view((bl+1)%5,1,   k+nlvl+1,i);
        anorm.h_view(bl,nlvl+1,k+nlvl+1,i) = anorm.h_view((bl+1)%5,k+2, 2*nlvl,  i);
      }
      anorm.h_view(bl,1,     0,       i) = apnorm.h_view(0,i);
      anorm.h_view(bl,nlvl+1,2*nlvl,  i) = apnorm.h_view(1,i);
      anorm.h_view(bl,0,     2*nlvl+1,i) = anorm.h_view(bl,0,2*nlvl,i);
    }
  }
}

// inline function to find arc length between two face centers
KOKKOS_INLINE_FUNCTION
Real ArcLength(int ic1, int ic2, int nlvl,
               DualArray4D<Real> anorm, DualArray2D<Real> apnorm) {
  Real x1, y1, z1, x2, y2, z2;
  GridCartPosition(ic1,nlvl,anorm,apnorm,&x1,&y1,&z1);
  GridCartPosition(ic2,nlvl,anorm,apnorm,&x2,&y2,&z2);
  return acos(x1*x2+y1*y2+z1*z2);
}

// inline function to find circumcenter of face
KOKKOS_INLINE_FUNCTION
void CircumcenterNormalized(Real x1, Real x2, Real x3,
                            Real y1, Real y2, Real y3,
                            Real z1, Real z2, Real z3,
                            Real *x, Real *y, Real *z) {
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

// inline function to find the parameters describing a great circle connecting two angles
KOKKOS_INLINE_FUNCTION
void GreatCircleParam(Real zeta1, Real zeta2, Real psi1, Real psi2,
                      Real *apar, Real *psi0) {
  Real atilde = (sin(psi2)/tan(zeta1)-sin(psi1)/tan(zeta2))/sin(psi2-psi1);
  Real btilde = (cos(psi2)/tan(zeta1)-cos(psi1)/tan(zeta2))/sin(psi1-psi2);
  *psi0 = atan2(btilde, atilde);
  *apar = sqrt(atilde*atilde+btilde*btilde);
}

} // namespace radiation
