//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_geom.cpp
//  \brief Initializes angular mesh and coordinate frame data.

#include <cmath>

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

#define HUGE_NUMBER 1.0e+36

namespace radiation {

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::InitMesh()
//! \brief Initialize angular mesh

void Radiation::InitAngularMesh() {
  // construct polar angles, equally spaced in cosine
  auto zetaf_ = zetaf;
  auto zetav_ = zetav;
  auto dzetaf_ = dzetaf;
  int &nzeta = amesh_indcs.nzeta;
  int &zs = amesh_indcs.zs;
  int &ze = amesh_indcs.ze;
  int &ng = amesh_indcs.ng;

  Real dczeta = -2.0 / nzeta;
  zetaf_.h_view(zs) = 0.0;     // set N pole exactly
  zetaf_.h_view(ze+1) = M_PI;  // set S pole exactly
  for (int z = zs+1; z <= (nzeta-1)/2+ng; ++z) {
    Real czeta = 1.0 + (z - ng) * dczeta;
    Real zeta = acos(czeta);
    zetaf_.h_view(z) = zeta;                 // set N active faces
    zetaf_.h_view(ze+ng+1-z) = M_PI - zeta;  // set S active faces
  }
  if (nzeta%2 == 0) {
    zetaf_.h_view(nzeta/2+ng) = M_PI/2.0;  // set equator exactly if present
  }
  for (int z = zs-ng; z <= zs-1; ++z) {
    zetaf_.h_view(z) = -zetaf_.h_view(2*ng - z);                   // set N ghost faces
    zetaf_.h_view(ze+ng+1-z) = 2.0*M_PI - zetaf_.h_view(nzeta+z);  // set S ghost faces
  }
  for (int z = zs-ng; z <= ze+ng; ++z) {
    zetav_.h_view(z) = (zetaf_.h_view(z+1)  * cos(zetaf_.h_view(z+1))
                  - sin(zetaf_.h_view(z+1)) - zetaf_.h_view(z)
                  * cos(zetaf_.h_view(z  )) + sin(zetaf_.h_view(z)))
                 / (cos(zetaf_.h_view(z+1)) - cos(zetaf_.h_view(z)));
    dzetaf_.h_view(z) = zetaf_.h_view(z+1)  - zetaf_.h_view(z);
  }

  zetaf_.template modify<HostMemSpace>();
  zetav_.template modify<HostMemSpace>();
  dzetaf_.template modify<HostMemSpace>();

  zetaf_.template sync<DevExeSpace>();
  zetav_.template sync<DevExeSpace>();
  dzetaf_.template sync<DevExeSpace>();

  // construct azimuthal angles, equally spaced
  auto psif_ = psif;
  auto psiv_ = psiv;
  auto dpsif_ = dpsif;
  int &npsi = amesh_indcs.npsi;
  int &ps = amesh_indcs.ps;
  int &pe = amesh_indcs.pe;

  Real dpsi = 2.0*M_PI / npsi;
  psif_.h_view(ps) = 0.0;       // set origin exactly
  psif_.h_view(pe+1) = 2.0*M_PI;  // set origin exactly
  for (int p = ps+1; p <= pe; ++p) {
    psif_.h_view(p) = (p - ng) * dpsi;  // set active faces
  }
  for (int p = ps-ng; p <= ps-1; ++p) {
    psif_.h_view(p) = psif_.h_view(npsi+p) - 2.0*M_PI;          // first ghost faces
    psif_.h_view(pe+ng+1-p) = psif_.h_view(2*ng-p) + 2.0*M_PI;  // last ghost faces
  }
  for (int p = ps-ng; p <= pe+ng; ++p) {
    psiv_.h_view(p) = 0.5 * (psif_.h_view(p) + psif_.h_view(p+1));
    dpsif_.h_view(p) = psif_.h_view(p+1) - psif_.h_view(p);
  }

  psif_.template modify<HostMemSpace>();
  psiv_.template modify<HostMemSpace>();
  dpsif_.template modify<HostMemSpace>();

  psif_.template sync<DevExeSpace>();
  psiv_.template sync<DevExeSpace>();
  dpsif_.template sync<DevExeSpace>();

  // Calculate angular lengths and areas
  auto zeta_length_ = zeta_length;
  auto psi_length_ = psi_length;
  auto solid_angle_ = solid_angle;

  for (int z = zs-ng; z <= ze+ng; ++z) {
    for (int p = ps-ng; p <= pe+ng+1; ++p) {
      zeta_length_.h_view(z,p) = cos(zetaf_.h_view(z)) - cos(zetaf_.h_view(z+1));
    }
  }
  for (int z = zs-ng; z <= ze+ng+1; ++z) {
    for (int p = ps-ng; p <= pe+ng; ++p) {
      psi_length_.h_view(z,p) = sin(zetaf_.h_view(z)) * dpsif_.h_view(p);
    }
  }
  for (int z = zs-ng; z <= ze+ng; ++z) {
    for (int p = ps-ng; p <= pe+ng; ++p) {
      solid_angle_.h_view(z,p) = (cos(zetaf_.h_view(z))
                                  - cos(zetaf_.h_view(z+1))) * dpsif_.h_view(p);
    }
  }

  zeta_length_.template modify<HostMemSpace>();
  psi_length_.template modify<HostMemSpace>();
  solid_angle_.template modify<HostMemSpace>();

  zeta_length_.template sync<DevExeSpace>();
  psi_length_.template sync<DevExeSpace>();
  solid_angle_.template sync<DevExeSpace>();


  int ncellsa1 = amesh_indcs.nzeta + 2*(amesh_indcs.ng);
  int ncellsa2 = amesh_indcs.npsi + 2*(amesh_indcs.ng);
  auto nh_c_ = nh_c;
  auto nh_zf_ = nh_zf;
  auto nh_pf_ = nh_pf;

  // Calculate unit normal components at angle centers
  for (int z = 0; z <= (ncellsa1-1); ++z) {
    for (int p = 0; p <= (ncellsa2-1); ++p) {
      nh_c_.h_view(z,p,0) = 1.0;
      nh_c_.h_view(z,p,1) = sin(zetav_.h_view(z)) * cos(psiv_.h_view(p));
      nh_c_.h_view(z,p,2) = sin(zetav_.h_view(z)) * sin(psiv_.h_view(p));
      nh_c_.h_view(z,p,3) = cos(zetav_.h_view(z));
    }
  }

  // Calculate unit normal components in orthonormal frame at zeta-faces
  for (int z = 0; z <= ncellsa1; ++z) {
    for (int p = 0; p <= (ncellsa2-1); ++p) {
      nh_zf_.h_view(z,p,0) = 1.0;
      nh_zf_.h_view(z,p,1) = sin(zetaf_.h_view(z)) * cos(psiv_.h_view(p));
      nh_zf_.h_view(z,p,2) = sin(zetaf_.h_view(z)) * sin(psiv_.h_view(p));
      nh_zf_.h_view(z,p,3) = cos(zetaf_.h_view(z));
    }
  }

  // Calculate unit normal components in orthonormal frame at psi-faces
  for (int z = 0; z <= (ncellsa1-1); ++z) {
    for (int p = 0; p <= ncellsa2; ++p) {
      nh_pf_.h_view(z,p,0) = 1.0;
      nh_pf_.h_view(z,p,1) = sin(zetav_.h_view(z)) * cos(psif_.h_view(p));
      nh_pf_.h_view(z,p,2) = sin(zetav_.h_view(z)) * sin(psif_.h_view(p));
      nh_pf_.h_view(z,p,3) = cos(zetav_.h_view(z));
    }
  }

  nh_c_.template modify<HostMemSpace>();
  nh_zf_.template modify<HostMemSpace>();
  nh_pf_.template modify<HostMemSpace>();

  nh_c_.template sync<DevExeSpace>();
  nh_zf_.template sync<DevExeSpace>();
  nh_pf_.template sync<DevExeSpace>();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::InitCoordinateFrame()
//! \brief Initialize frame related quantities.

void Radiation::InitRadiationFrame() {
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

  int ncellsa1 = amesh_indcs.nzeta + 2*(amesh_indcs.ng);
  int ncellsa2 = amesh_indcs.npsi + 2*(amesh_indcs.ng);

  auto nh_c_ = nh_c;
  auto nh_zf_ = nh_zf;
  auto nh_pf_ = nh_pf;

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
    for (int z=0; z<=(ncellsa1-1); ++z) {
      for (int p=0; p<=(ncellsa2-1); ++p) {
        Real n0 = 0.0;
        Real n1 = 0.0;
        Real n2 = 0.0;
        Real n3 = 0.0;
        Real n_0 = 0.0;
        Real n_1 = 0.0;
        Real n_2 = 0.0;
        Real n_3 = 0.0;
        for (int d=0; d<4; ++d) {
          n0 += e[d][0]*nh_c_.d_view(z,p,d);
          n1 += e[d][1]*nh_c_.d_view(z,p,d);
          n2 += e[d][2]*nh_c_.d_view(z,p,d);
          n3 += e[d][3]*nh_c_.d_view(z,p,d);
          n_0 += e_cov[d][0]*nh_c_.d_view(z,p,d);
          n_1 += e_cov[d][1]*nh_c_.d_view(z,p,d);
          n_2 += e_cov[d][2]*nh_c_.d_view(z,p,d);
          n_3 += e_cov[d][3]*nh_c_.d_view(z,p,d);
        }
        nmu_(m,z,p,k,j,i,0) = n0;
        nmu_(m,z,p,k,j,i,1) = n1;
        nmu_(m,z,p,k,j,i,2) = n2;
        nmu_(m,z,p,k,j,i,3) = n3;
        n_mu_(m,z,p,k,j,i,0) = n_0;
        n_mu_(m,z,p,k,j,i,1) = n_1;
        n_mu_(m,z,p,k,j,i,2) = n_2;
        n_mu_(m,z,p,k,j,i,3) = n_3;
      }
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
    for (int z=0; z<=(ncellsa1-1); ++z) {
      for (int p=0; p<=(ncellsa2-1); ++p) {
        Real n1 = 0.0;
        Real n_0 = 0.0;
        for (int d=0; d<4; ++d) {
          n1 += e[d][1]*nh_c_.d_view(z,p,d);
          n_0 += e_cov[d][0]*nh_c_.d_view(z,p,d);
        }
        n1_n_0_(m,z,p,k,j,i) = n1*n_0;
      }
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
    for (int z=0; z<=(ncellsa1-1); ++z) {
      for (int p=0; p<=(ncellsa2-1); ++p) {
        Real n2 = 0.0;
        Real n_0 = 0.0;
        for (int d=0; d<4; ++d) {
          n2 += e[d][2]*nh_c_.d_view(z,p,d);
          n_0 += e_cov[d][0]*nh_c_.d_view(z,p,d);
        }
        n2_n_0_(m,z,p,k,j,i) = n2*n_0;
      }
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
    for (int z=0; z<=(ncellsa1-1); ++z) {
      for (int p=0; p<=(ncellsa2-1); ++p) {
        Real n3 = 0.0;
        Real n_0 = 0.0;
        for (int d=0; d<4; ++d) {
          n3 += e[d][3]*nh_c_.d_view(z,p,d);
          n_0 += e_cov[d][0]*nh_c_.d_view(z,p,d);
        }
        n3_n_0_(m,z,p,k,j,i) = n3*n_0;
      }
    }
  });

  // Calculate n^angle1 n_0
  auto zetaf_ = zetaf;
  auto zetav_ = zetav;
  auto na1_n_0_ = na1_n_0;
  auto na2_n_0_ = na2_n_0;
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
    for (int z=0; z<=(ncellsa1); ++z) {
      for (int p=0; p<=(ncellsa2-1); ++p) {
        Real na1 = 0.0;
        for (int d1=0; d1<4; ++d1) {
          for (int d2=0; d2<4; ++d2) {
            na1 += (1.0/sin(zetaf_.d_view(z))*nh_zf_.d_view(z,p,d1)*nh_zf_.d_view(z,p,d2)
                    * (nh_zf_.d_view(z,p,0)*omega[3][d1][d2]
                    -  nh_zf_.d_view(z,p,3)*omega[0][d1][d2]));
          }
        }
        Real n_0 = 0.0;
        for (int q=0; q<4; ++q) {
          n_0 += e_cov[q][0]*nh_zf_.d_view(z,p,q);
        }
        na1_n_0_(m,z,p,k,j,i) = na1*n_0;
      }
    }

    for (int z=0; z<=(ncellsa1-1); ++z) {
      for (int p=0; p<=(ncellsa2); ++p) {
        Real na2 = 0.0;
        for (int d1=0; d1<4; ++d1) {
          for (int d2=0; d2<4; ++d2) {
            na2 += (1.0/SQR(sin(zetav_.d_view(z)))*nh_pf_.d_view(z,p,d1)*nh_pf_.d_view(z,p,d2)
                    * (nh_pf_.d_view(z,p,2)*omega[1][d1][d2]
                    -  nh_pf_.d_view(z,p,1)*omega[2][d1][d2]));
          }
        }
        Real n_0 = 0.0;
        for (int q=0; q<4; ++q) {
          n_0 += e_cov[q][0]*nh_pf_.d_view(z,p,q);
        }
        na2_n_0_(m,z,p,k,j,i) = na2*n_0;
      }
    }
  });

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

} // namespace radiation
