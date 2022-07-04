//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file derived_variables.cpp
//! \brief Calculates various derived variables for outputs, storing them into the
//! "derived_vars" device array located in BaseTypeOutput class.  Variables are only
//! calculated over active zones (ghost zones excluded). Currently implemented are:
//!   - z-component of vorticity Curl(v)_z  [non-relativistic]
//!   - magnitude of vorticity Curl(v)^2  [non-relativistic]
//!   - z-component of current density Jz  [non-relativistic]
//!   - magnitude of current density J^2  [non-relativistic]

#include <sstream>
#include <string>   // std::string, to_string()

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "outputs.hpp"

#include "coordinates/cartesian_ks.hpp"
#include "radiation/radiation_tetrad.hpp"

//----------------------------------------------------------------------------------------
// BaseTypeOutput::ComputeDerivedVariable()

void BaseTypeOutput::ComputeDerivedVariable(std::string name, Mesh *pm) {
  int nmb = pm->pmb_pack->nmb_thispack;
  auto &indcs = pm->mb_indcs;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &multi_d = pm->multi_d;
  auto &three_d = pm->three_d;

  // z-component of vorticity.
  // Not computed in ghost zones since requires derivative
  if (name.compare("hydro_wz") == 0 ||
      name.compare("mhd_wz") == 0) {
    auto dv = derived_var;
    auto w0_ = (name.compare("hydro_wz") == 0)?
      pm->pmb_pack->phydro->w0 : pm->pmb_pack->pmhd->w0;
    par_for("jz", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      dv(m,0,k,j,i) = (w0_(m,IVY,k,j,i+1) - w0_(m,IVY,k,j,i-1))/size.d_view(m).dx1;
      if (multi_d) {
        dv(m,0,k,j,i) -=(w0_(m,IVX,k,j+1,i) - w0_(m,IVX,k,j-1,i))/size.d_view(m).dx2;
      }
    });
  }

  // magnitude of vorticity.
  // Not computed in ghost zones since requires derivative
  if (name.compare("hydro_w2") == 0 ||
      name.compare("mhd_w2") == 0) {
    auto dv = derived_var;
    auto w0_ = (name.compare("hydro_w2") == 0)?
      pm->pmb_pack->phydro->w0 : pm->pmb_pack->pmhd->w0;
    par_for("jz", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real w1 = 0.0;
      Real w2 = -(w0_(m,IVZ,k,j,i+1) - w0_(m,IVZ,k,j,i-1))/size.d_view(m).dx1;
      Real w3 =  (w0_(m,IVY,k,j,i+1) - w0_(m,IVY,k,j,i-1))/size.d_view(m).dx1;
      if (multi_d) {
        w1 += (w0_(m,IVZ,k,j+1,i) - w0_(m,IVZ,k,j-1,i))/size.d_view(m).dx2;
        w3 -= (w0_(m,IVX,k,j+1,i) - w0_(m,IVX,k,j-1,i))/size.d_view(m).dx2;
      }
      if (three_d) {
        w1 -= (w0_(m,IVY,k+1,j,i) - w0_(m,IVY,k-1,j,i))/size.d_view(m).dx3;
        w2 += (w0_(m,IVX,k+1,j,i) - w0_(m,IVX,k-1,j,i))/size.d_view(m).dx3;
      }
      dv(m,0,k,j,i) = w1*w1 + w2*w2 + w3*w3;
    });
  }

  // z-component of current density.  Calculated from cell-centered fields.
  // This makes for a large stencil, but approximates volume-averaged value within cell.
  // Not computed in ghost zones since requires derivative
  if (name.compare("mhd_jz") == 0) {
    auto dv = derived_var;
    auto bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("jz", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      dv(m,0,k,j,i) = (bcc(m,IBY,k,j,i+1) - bcc(m,IBY,k,j,i-1))/size.d_view(m).dx1;
      if (multi_d) {
        dv(m,0,k,j,i) -=(bcc(m,IBX,k,j+1,i) - bcc(m,IBX,k,j-1,i))/size.d_view(m).dx2;
      }
    });
  }

  // magnitude of current density.  Calculated from cell-centered fields.
  // Not computed in ghost zones since requires derivative
  if (name.compare("mhd_j2") == 0) {
    auto dv = derived_var;
    auto bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("jz", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real j1 = 0.0;
      Real j2 = -(bcc(m,IBZ,k,j,i+1) - bcc(m,IBZ,k,j,i-1))/size.d_view(m).dx1;
      Real j3 =  (bcc(m,IBY,k,j,i+1) - bcc(m,IBY,k,j,i-1))/size.d_view(m).dx1;
      if (multi_d) {
        j1 += (bcc(m,IBZ,k,j+1,i) - bcc(m,IBZ,k,j-1,i))/size.d_view(m).dx2;
        j3 -= (bcc(m,IBX,k,j+1,i) - bcc(m,IBX,k,j-1,i))/size.d_view(m).dx2;
      }
      if (three_d) {
        j1 -= (bcc(m,IBY,k+1,j,i) - bcc(m,IBY,k-1,j,i))/size.d_view(m).dx3;
        j2 += (bcc(m,IBX,k+1,j,i) - bcc(m,IBX,k-1,j,i))/size.d_view(m).dx3;
      }
      dv(m,0,k,j,i) = j1*j1 + j2*j2 + j3*j3;
    });
  }

  // divergence of B, including ghost zones
  if (name.compare("mhd_divb") == 0) {
    Kokkos::realloc(derived_var, nmb, 1, n3, n2, n1);

    // set the loop limits for 1D/2D/3D problems
    int jl = js, ju = je, kl = ks, ku = ke;
    if (multi_d) {
      jl = js-ng, ju = je+ng;
    } else if (three_d) {
      jl = js-ng, ju = je+ng, kl = ks-ng, ku = ke+ng;
    }

    auto dv = derived_var;
    auto b0 = pm->pmb_pack->pmhd->b0;
    par_for("divb", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, (is-ng), (ie+ng),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real divb = (b0.x1f(m,k,j,i+1) - b0.x1f(m,k,j,i))/size.d_view(m).dx1;
      if (multi_d) {
        divb += (b0.x2f(m,k,j+1,i) - b0.x2f(m,k,j,i))/size.d_view(m).dx2;
      }
      if (three_d) {
        divb += (b0.x3f(m,k+1,j,i) - b0.x3f(m,k,j,i))/size.d_view(m).dx3;
      }
      dv(m,0,k,j,i) = divb;
    });
  }

  // radiation moments evaluated in the coordinate frame
  if (name.compare("rad_coord") == 0) {
    Kokkos::realloc(derived_var, nmb, 10, n3, n2, n1);
    auto dv = derived_var;
    int nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
    auto nh_c_ = pm->pmb_pack->prad->nh_c;
    auto tet_c_ = pm->pmb_pack->prad->tet_c;
    auto tetcov_c_ = pm->pmb_pack->prad->tetcov_c;
    auto i0_ = pm->pmb_pack->prad->i0;
    auto solid_angles_ = pm->pmb_pack->prad->prgeo->solid_angles;
    par_for("moments_coord",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      for (int n1=0, n12=0; n1<4; ++n1) {
        for (int n2=n1; n2<4; ++n2, ++n12) {
          dv(m,n12,k,j,i) = 0.0;
          for (int n=0; n<=nang1; ++n) {
            Real nmun1 = 0.0; Real nmun2 = 0.0; Real n_0 = 0.0;
            for (int d=0; d<4; ++d) {
              nmun1  += tet_c_   (m,d,n1,k,j,i)*nh_c_.d_view(n,d);
              nmun2  += tet_c_   (m,d,n2,k,j,i)*nh_c_.d_view(n,d);
              n_0     += tetcov_c_(m,d,0, k,j,i)*nh_c_.d_view(n,d);
            }
            dv(m,n12,k,j,i) += (nmun1*nmun2*(i0_(m,n,k,j,i)/n_0)*solid_angles_.d_view(n));
          }
        }
      }
    });
  }

  // radiation moments evaluated in the coordinate frame
  if (name.compare("rad_fluid") == 0) {
    Kokkos::realloc(derived_var, nmb, 10, n3, n2, n1);
    auto dv = derived_var;
    int nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
    auto nh_c_ = pm->pmb_pack->prad->nh_c;
    auto tet_c_ = pm->pmb_pack->prad->tet_c;
    auto tetcov_c_ = pm->pmb_pack->prad->tetcov_c;
    auto solid_angles_ = pm->pmb_pack->prad->prgeo->solid_angles;

    auto &coord = pm->pmb_pack->pcoord->coord_data;
    bool &flat = coord.is_minkowski;
    Real &spin = coord.bh_spin;

    auto i0_ = pm->pmb_pack->prad->i0;
    auto w0_ = pm->pmb_pack->phydro->w0;
    auto norm_to_tet_ = pm->pmb_pack->prad->norm_to_tet;
    par_for("moments_fluid",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Set tetrad-frame components
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      // Extract components of metric
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
      Real dgx[4][4], dgy[4][4], dgz[4][4];
      ComputeMetricDerivatives(x1v,x2v,x3v,flat,spin,dgx,dgy,dgz);
      Real e[4][4], e_cov[4][4], omega[4][4][4];
      ComputeTetrad(x1v,x2v,x3v,flat,spin,glower,gupper,dgx,dgy,dgz,e,e_cov,omega);
      // fluid velocity
      Real uu1 = w0_(m,IVX,k,j,i);
      Real uu2 = w0_(m,IVY,k,j,i);
      Real uu3 = w0_(m,IVZ,k,j,i);
      Real q = glower[1][1]*uu1*uu1 + 2.0*glower[1][2]*uu1*uu2 + 2.0*glower[1][3]*uu1*uu3
             + glower[2][2]*uu2*uu2 + 2.0*glower[2][3]*uu2*uu3
             + glower[3][3]*uu3*uu3;
      Real uu0 = sqrt(1.0 + q);

      // fluid velocity in tetrad frame
      Real u_tet_[4];
      u_tet_[0] = (norm_to_tet_(m,0,0,k,j,i)*uu0 + norm_to_tet_(m,0,1,k,j,i)*uu1 +
                   norm_to_tet_(m,0,2,k,j,i)*uu2 + norm_to_tet_(m,0,3,k,j,i)*uu3);
      u_tet_[1] = (norm_to_tet_(m,1,0,k,j,i)*uu0 + norm_to_tet_(m,1,1,k,j,i)*uu1 +
                   norm_to_tet_(m,1,2,k,j,i)*uu2 + norm_to_tet_(m,1,3,k,j,i)*uu3);
      u_tet_[2] = (norm_to_tet_(m,2,0,k,j,i)*uu0 + norm_to_tet_(m,2,1,k,j,i)*uu1 +
                   norm_to_tet_(m,2,2,k,j,i)*uu2 + norm_to_tet_(m,2,3,k,j,i)*uu3);
      u_tet_[3] = (norm_to_tet_(m,3,0,k,j,i)*uu0 + norm_to_tet_(m,3,1,k,j,i)*uu1 +
                   norm_to_tet_(m,3,2,k,j,i)*uu2 + norm_to_tet_(m,3,3,k,j,i)*uu3);

      // Construct Lorentz boost from tetrad frame to orthonormal fluid frame
      Real tet_to_fluid[4][4] = {0.0};
      tet_to_fluid[0][0] = u_tet_[0];
      tet_to_fluid[0][1] = tet_to_fluid[1][0] = -u_tet_[1];
      tet_to_fluid[0][2] = tet_to_fluid[2][0] = -u_tet_[2];
      tet_to_fluid[0][3] = tet_to_fluid[3][0] = -u_tet_[3];
      tet_to_fluid[1][1] = SQR(u_tet_[1])/(1.0 + u_tet_[0]) + 1.0;
      tet_to_fluid[1][2] = tet_to_fluid[2][1] = u_tet_[1] * u_tet_[2]/(1.0 + u_tet_[0]);
      tet_to_fluid[1][3] = tet_to_fluid[3][1] = u_tet_[1] * u_tet_[3]/(1.0 + u_tet_[0]);
      tet_to_fluid[2][2] = SQR(u_tet_[2])/(1.0 + u_tet_[0]) + 1.0;
      tet_to_fluid[2][3] = tet_to_fluid[3][2] = u_tet_[2] * u_tet_[3]/(1.0 + u_tet_[0]);
      tet_to_fluid[3][3] = SQR(u_tet_[3])/(1.0 + u_tet_[0]) + 1.0;

      // set coordinate frame components
      for (int n1=0, n12=0; n1<4; ++n1) {
        for (int n2=n1; n2<4; ++n2, ++n12) {
          dv(m,n12,k,j,i) = 0.0;
          for (int n=0; n<=nang1; ++n) {
            Real nmun1 = 0.0; Real nmun2 = 0.0; Real n_0 = 0.0;
            for (int d=0; d<4; ++d) {
              nmun1  += tet_c_   (m,d,n1,k,j,i)*nh_c_.d_view(n,d);
              nmun2  += tet_c_   (m,d,n2,k,j,i)*nh_c_.d_view(n,d);
              n_0     += tetcov_c_(m,d,0, k,j,i)*nh_c_.d_view(n,d);
            }
            dv(m,n12,k,j,i) += (nmun1*nmun2*(i0_(m,n,k,j,i)/n_0)*solid_angles_.d_view(n));
          }
        }
      }

      // stash coordinate frame moments
      Real moments_coord_full[4][4] = {0.0};
      moments_coord_full[0][0] = dv(m,0,k,j,i);
      moments_coord_full[0][1] = dv(m,1,k,j,i);
      moments_coord_full[0][2] = dv(m,2,k,j,i);
      moments_coord_full[0][3] = dv(m,3,k,j,i);
      moments_coord_full[1][1] = dv(m,4,k,j,i);
      moments_coord_full[1][2] = dv(m,5,k,j,i);
      moments_coord_full[1][3] = dv(m,6,k,j,i);
      moments_coord_full[2][2] = dv(m,7,k,j,i);
      moments_coord_full[2][3] = dv(m,8,k,j,i);
      moments_coord_full[3][3] = dv(m,9,k,j,i);
      moments_coord_full[1][0] = moments_coord_full[0][1];
      moments_coord_full[2][0] = moments_coord_full[0][2];
      moments_coord_full[3][0] = moments_coord_full[0][3];
      moments_coord_full[2][1] = moments_coord_full[1][2];
      moments_coord_full[3][1] = moments_coord_full[1][3];
      moments_coord_full[3][2] = moments_coord_full[2][3];

      // set tetrad frame moments
      for (int n1=0, n12=0; n1<4; ++n1) {
        for (int n2=n1; n2<4; ++n2, ++n12) {
          dv(m,n12,k,j,i) = 0.0;
          for (int m1=0; m1<4; ++m1) {
            for (int m2=0; m2<4; ++m2) {
              dv(m,n12,k,j,i) += (e_cov[n1][m1]*e_cov[n2][m2]
                                  *moments_coord_full[m1][m2]);
            }
          }
        }
      }
      dv(m,1,k,j,i) *= -1.0;
      dv(m,2,k,j,i) *= -1.0;
      dv(m,3,k,j,i) *= -1.0;

      // stash tetrad frame moments
      Real moments_tetrad_full[4][4] = {0.0};
      moments_tetrad_full[0][0] = dv(m,0,k,j,i);
      moments_tetrad_full[0][1] = dv(m,1,k,j,i);
      moments_tetrad_full[0][2] = dv(m,2,k,j,i);
      moments_tetrad_full[0][3] = dv(m,3,k,j,i);
      moments_tetrad_full[1][1] = dv(m,4,k,j,i);
      moments_tetrad_full[1][2] = dv(m,5,k,j,i);
      moments_tetrad_full[1][3] = dv(m,6,k,j,i);
      moments_tetrad_full[2][2] = dv(m,7,k,j,i);
      moments_tetrad_full[2][3] = dv(m,8,k,j,i);
      moments_tetrad_full[3][3] = dv(m,9,k,j,i);
      moments_tetrad_full[1][0] = moments_tetrad_full[0][1];
      moments_tetrad_full[2][0] = moments_tetrad_full[0][2];
      moments_tetrad_full[3][0] = moments_tetrad_full[0][3];
      moments_tetrad_full[2][1] = moments_tetrad_full[1][2];
      moments_tetrad_full[3][1] = moments_tetrad_full[1][3];
      moments_tetrad_full[3][2] = moments_tetrad_full[2][3];

      for (int n1=0, n12=0; n1<4; ++n1) {
        for (int n2=n1; n2<4; ++n2, ++n12) {
          dv(m,n12,k,j,i) = 0.0;
          for (int m1 = 0; m1 < 4; ++m1) {
            for (int m2 = 0; m2 < 4; ++m2) {
              dv(m,n12,k,j,i) += (tet_to_fluid[n1][m1]*tet_to_fluid[n2][m2]
                                  *moments_tetrad_full[m1][m2]);
            }
          }
        }
      }
    });
  }
}
