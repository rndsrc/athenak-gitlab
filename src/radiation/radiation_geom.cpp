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
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"
#include "radiation.hpp"

#include <cmath>
  
namespace radiation {

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::InitMesh
//! \brief Initialize angular mesh

void Radiation::InitMesh() {
  // construct polar angles, equally spaced in cosine
  auto zetaf_ = zetaf;
  auto zetav_ = zetav;
  auto dzetaf_ = dzetaf;
  int &nzeta = amesh_indcs.nzeta;
  int &zs = amesh_indcs.zs;
  int &ze = amesh_indcs.ze;
  int &ng = amesh_indcs.ng;

  Real dczeta = -2.0 / nzeta;
  zetaf_.h_view(zs) = 0.0;   // set N pole exactly
  zetaf_.h_view(ze+1) = M_PI;  // set south pole exactly
  for (int z = zs+1; z <= (nzeta-1)/2+ng; ++z) {
    Real czeta = 1.0 + (z - ng) * dczeta;
    Real zeta = acos(czeta);
    zetaf_.h_view(z) = zeta;                // set N active faces
    zetaf_.h_view(ze+ng+1-z) = M_PI - zeta;  // set S active faces
  }
  if (nzeta%2 == 0) {
    zetaf_.h_view(nzeta/2+ng) = M_PI/2.0;  // set equator exactly if present
  }
  for (int z = zs-ng; z <= zs-1; ++z) {
    zetaf_.h_view(z) = -zetaf_.h_view(2*ng - z);                 // set N ghost faces
    zetaf_.h_view(ze+ng+1-z) = 2.0*M_PI - zetaf_.h_view(nzeta+z);  // set S ghost faces
  }
  for (int z = zs-ng; z <= ze+ng; ++z) {
    zetav_.h_view(z) = (zetaf_.h_view(z+1) * cos(zetaf_.h_view(z+1))
                - sin(zetaf_.h_view(z+1)) - zetaf_.h_view(z)
                * cos(zetaf_.h_view(z)) + sin(zetaf_.h_view(z)))
                / (cos(zetaf_.h_view(z+1)) - cos(zetaf_.h_view(z)));
    dzetaf_.h_view(z) = zetaf_.h_view(z+1) - zetaf_.h_view(z);
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

  // Calculate unit normal components in orthonormal frame at angle centers
  auto nh_cc_ = nh_cc;
  auto nh_fc_ = nh_fc;
  auto nh_cf_ = nh_cf;

  for (int z = zs; z <= ze; ++z) {
    for (int p = ps; p <= pe; ++p) {
      nh_cc_.h_view(0,z,p) = 1.0;
      nh_cc_.h_view(1,z,p) = sin(zetav_.h_view(z)) * cos(psiv_.h_view(p));
      nh_cc_.h_view(2,z,p) = sin(zetav_.h_view(z)) * sin(psiv_.h_view(p));
      nh_cc_.h_view(3,z,p) = cos(zetav_.h_view(z));
    }
  }

  // Calculate unit normal components in orthonormal frame at zeta-faces
  for (int z = zs; z <= ze+1; ++z) {
    for (int p = ps; p <= pe; ++p) {
      nh_fc_.h_view(0,z,p) = 1.0;
      nh_fc_.h_view(1,z,p) = sin(zetaf_.h_view(z)) * cos(psiv_.h_view(p));
      nh_fc_.h_view(2,z,p) = sin(zetaf_.h_view(z)) * sin(psiv_.h_view(p));
      nh_fc_.h_view(3,z,p) = cos(zetaf_.h_view(z));
    }
  }

  // Calculate unit normal components in orthonormal frame at psi-faces
  for (int z = zs; z <= ze; ++z) {
    for (int p = ps; p <= pe+1; ++p) {
      nh_cf_.h_view(0,z,p) = 1.0;
      nh_cf_.h_view(1,z,p) = sin(zetav_.h_view(z)) * cos(psif_.h_view(p));
      nh_cf_.h_view(2,z,p) = sin(zetav_.h_view(z)) * sin(psif_.h_view(p));
      nh_cf_.h_view(3,z,p) = cos(zetav_.h_view(z));
    }
  }

  nh_cc_.template modify<HostMemSpace>();
  nh_fc_.template modify<HostMemSpace>();
  nh_cf_.template modify<HostMemSpace>();

  nh_cc_.template sync<DevExeSpace>();
  nh_fc_.template sync<DevExeSpace>();
  nh_cf_.template sync<DevExeSpace>();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::InitCoordinateFrame
//! \brief Initialize frame related quantities.
// (TODO: @pdmullen): this is really gross...Does this simplify to something much
// simpler for just CartesianKS.  Lugging around these huge arrays may be crippling on
// GPUs.

void Radiation::InitCoordinateFrame() {
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;

  int &zs = amesh_indcs.zs;  int &ze = amesh_indcs.ze;
  int &ps = amesh_indcs.ps;  int &pe = amesh_indcs.pe;

  int &nmb = pmy_pack->nmb_thispack;
  auto coord = pmy_pack->coord.coord_data;

  auto nh_cc_ = nh_cc;
  auto nh_fc_ = nh_fc;
  auto nh_cf_ = nh_cf;

  auto zetaf_ = zetaf;
  auto zetav_ = zetav;

  Real e[4][4] = {};
  Real e_cov[4][4] = {};
  Real omega[4][4][4] = {};
  auto e_ = e;
  auto e_cov_ = e_cov;
  auto omega_ = omega;

  // Calculate n^mu and n^0 n_mu
  auto nmu_ = nmu;
  auto n0_n_mu_ = n0_n_mu;
  par_for("rad_nmu_n0_n_mu", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
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

      ComputeTetrad(x1v, x2v, x3v, true, e_, e_cov_, omega_);
      for (int z = zs; z <= ze; ++z) {
        for (int p = ps; p <= pe; ++p) {
          Real n0 = 0.0;
          Real n1 = 0.0;
          Real n2 = 0.0;
          Real n3 = 0.0;
          Real n_0 = 0.0;
          Real n_1 = 0.0;
          Real n_2 = 0.0;
          Real n_3 = 0.0;
          for (int d = 0; d < 4; ++d) {
            n0 += e_[d][0] * nh_cc_.d_view(d,z,p);
            n1 += e_[d][1] * nh_cc_.d_view(d,z,p);
            n2 += e_[d][2] * nh_cc_.d_view(d,z,p);
            n3 += e_[d][3] * nh_cc_.d_view(d,z,p);
            n_0 += e_cov_[d][0] * nh_cc_.d_view(d,z,p);
            n_1 += e_cov_[d][1] * nh_cc_.d_view(d,z,p);
            n_2 += e_cov_[d][2] * nh_cc_.d_view(d,z,p);
            n_3 += e_cov_[d][3] * nh_cc_.d_view(d,z,p);
          }
          nmu_(m,0,z,p,k,j,i) = n0;
          nmu_(m,1,z,p,k,j,i) = n1;
          nmu_(m,2,z,p,k,j,i) = n2;
          nmu_(m,3,z,p,k,j,i) = n3;
          n0_n_mu_(m,0,z,p,k,j,i) = n0 * n_0;
          n0_n_mu_(m,1,z,p,k,j,i) = n0 * n_1;
          n0_n_mu_(m,2,z,p,k,j,i) = n0 * n_2;
          n0_n_mu_(m,3,z,p,k,j,i) = n0 * n_3;
        }
      }
    }
  );

  // Calculate n^1 n_mu
  auto n1_n_mu_ = n1_n_mu;
  par_for("rad_n1_n_mu", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
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

      ComputeTetrad(x1f, x2v, x3v, true, e_, e_cov_, omega_);
      for (int z = zs; z <= ze; ++z) {
        for (int p = ps; p <= pe; ++p) {
          Real n1 = 0.0;
          Real n_0 = 0.0;
          Real n_1 = 0.0;
          Real n_2 = 0.0;
          Real n_3 = 0.0;
          for (int d = 0; d < 4; ++d) {
            n1 += e_[d][1] * nh_cc_.d_view(d,z,p);
            n_0 += e_cov_[d][0] * nh_cc_.d_view(d,z,p);
            n_1 += e_cov_[d][1] * nh_cc_.d_view(d,z,p);
            n_2 += e_cov_[d][2] * nh_cc_.d_view(d,z,p);
            n_3 += e_cov_[d][3] * nh_cc_.d_view(d,z,p);
          }
          n1_n_mu_(m,0,z,p,k,j,i) = n1 * n_0;
          n1_n_mu_(m,1,z,p,k,j,i) = n1 * n_1;
          n1_n_mu_(m,2,z,p,k,j,i) = n1 * n_2;
          n1_n_mu_(m,3,z,p,k,j,i) = n1 * n_3;
        }
      }
    }
  );

  // Calculate n^2 n_mu
  auto n2_n_mu_ = n2_n_mu;
  par_for("rad_n2_n_mu", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
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

      ComputeTetrad(x1v, x2f, x3v, true, e_, e_cov_, omega_);
      for (int z = zs; z <= ze; ++z) {
        for (int p = ps; p <= pe; ++p) {
          Real n2 = 0.0;
          Real n_0 = 0.0;
          Real n_1 = 0.0;
          Real n_2 = 0.0;
          Real n_3 = 0.0;
          for (int d = 0; d < 4; ++d) {
            n2 += e_[d][2] * nh_cc_.d_view(d,z,p);
            n_0 += e_cov_[d][0] * nh_cc_.d_view(d,z,p);
            n_1 += e_cov_[d][1] * nh_cc_.d_view(d,z,p);
            n_2 += e_cov_[d][2] * nh_cc_.d_view(d,z,p);
            n_3 += e_cov_[d][3] * nh_cc_.d_view(d,z,p);
          }
          n2_n_mu_(m,0,z,p,k,j,i) = n2 * n_0;
          n2_n_mu_(m,1,z,p,k,j,i) = n2 * n_1;
          n2_n_mu_(m,2,z,p,k,j,i) = n2 * n_2;
          n2_n_mu_(m,3,z,p,k,j,i) = n2 * n_3;
        }
      }
    }
  );

  // Calculate n^3 n_mu
  auto n3_n_mu_ = n3_n_mu;
  par_for("rad_n3_n_mu", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
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

      ComputeTetrad(x1v, x2v, x3f, true, e_, e_cov_, omega_);
      for (int z = zs; z <= ze; ++z) {
        for (int p = ps; p <= pe; ++p) {
          Real n3 = 0.0;
          Real n_0 = 0.0;
          Real n_1 = 0.0;
          Real n_2 = 0.0;
          Real n_3 = 0.0;
          for (int d = 0; d < 4; ++d) {
            n3 += e_[d][3] * nh_cc_.d_view(d,z,p);
            n_0 += e_cov_[d][0] * nh_cc_.d_view(d,z,p);
            n_1 += e_cov_[d][1] * nh_cc_.d_view(d,z,p);
            n_2 += e_cov_[d][2] * nh_cc_.d_view(d,z,p);
            n_3 += e_cov_[d][3] * nh_cc_.d_view(d,z,p);
          }
          n3_n_mu_(m,0,z,p,k,j,i) = n3 * n_0;
          n3_n_mu_(m,1,z,p,k,j,i) = n3 * n_1;
          n3_n_mu_(m,2,z,p,k,j,i) = n3 * n_2;
          n3_n_mu_(m,3,z,p,k,j,i) = n3 * n_3;
        }
      }
    }
  );

  // Calculate n^zeta n_0
  auto na1_n_0_ = na1_n_0;
  par_for("rad_na1_n_0", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
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

      ComputeTetrad(x1v, x2v, x3v, true, e_, e_cov_, omega_);
      for (int z = zs+1; z <= ze; ++z) {
        for (int p = ps; p <= pe; ++p) {
          Real na1 = 0.0;
          for (int d1 = 0; d1 < 4; ++d1) {
            for (int d2 = 0; d2 < 4; ++d2) {
              na1 += (1.0 / sin(zetaf_.d_view(z))
                      * nh_fc_.d_view(d1,z,p) * nh_fc_.d_view(d2,z,p)
                      * (nh_fc_.d_view(0,z,p) * omega_[3][d1][d2]
                         - nh_fc_.d_view(3,z,p) * omega_[0][d1][d2]));
            }
          }
          Real n_0 = 0.0;
          for (int d1 = 0; d1 < 4; ++d1) {
            n_0 += e_cov_[d1][0] * nh_fc_.d_view(d1,z,p);
          }
          na1_n_0_(m,z,p,k,j,i) = na1 * n_0;
        }
      }
    }
  );

  // Calculate n^psi n_0
  auto na2_n_0_ = na1_n_0;
  par_for("rad_na2_n_0", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
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

      ComputeTetrad(x1v, x2v, x3v, true, e_, e_cov_, omega_);
      for (int z = zs; z <= ze; ++z) {
        for (int p = ps; p <= pe+1; ++p) {
          Real na2 = 0.0;
          for (int d1 = 0; d1 < 4; ++d1) {
            for (int d2 = 0; d2 < 4; ++d2) {
              na2 += (1.0 / SQR(sin(zetav_.d_view(z))) * nh_cf_.d_view(d1,z,p)
                      * nh_cf_.d_view(d2,z,p)
                      * (nh_cf_.d_view(2,z,p) * omega_[1][d1][d2]
                         - nh_cf_.d_view(1,z,p) * omega_[2][d1][d2]));
            }
          }
          Real n_0 = 0.0;
          for (int d1 = 0; d1 < 4; ++d1) {
            n_0 += e_cov_[d1][0] * nh_cf_.d_view(d1,z,p);
          }
          na2_n_0_(m,z,p,k,j,i) = na2 * n_0;
        }
      }
    }
  );

  return;
}

} // namespace radiation
