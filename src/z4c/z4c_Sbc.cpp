//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_Sbc.cpp
//! \brief placeholder for Sommerfeld boundary condition

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/cell_locations.hpp"

namespace z4c {
/*
//---------------------------------------------------------------------------------------
//! \fn TaskStatus Z4c::Z4cBoundaryRHS
//! \brief placeholder for the Sommerfield Boundary conditions for z4c
TaskStatus Z4c::Z4cBoundaryRHS(Driver *pdrive, int stage) {
  if (stage == pdrive->nexp_stages) {
    
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    auto &size = pmy_pack->pmb->mb_size;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    int isg = is-indcs.ng; int ieg = ie+indcs.ng;
    int jsg = js-indcs.ng; int jeg = je+indcs.ng;
    int ksg = ks-indcs.ng; int keg = ke+indcs.ng;
    auto &z4c = pmy_pack->pz4c->z4c;
    auto &rhs = pmy_pack->pz4c->rhs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng); // Align scratch buffers with variables
    int nmb = pmy_pack->nmb_thispack;
    auto &mb_bcs = pmy_pack->pmb->mb_bcs;
 
    int scr_level = 0;
    size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);

    // Note that the derivative Ds in Athena++ corresponds to Dx<2> here
    par_for_outer("impose SBC",DevExeSpace(),
    scr_size,scr_level,0,nmb-1,ksg,keg,jsg,jeg,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
      // Theta 1st drvts
      AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dKhat_d;
      dKhat_d.NewAthenaScratchTensor(member, scr_level, ncells1);
      AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dDtheta_d;
      dDtheta_d.NewAthenaScratchTensor(member, scr_level, ncells1);
      AthenaScratchTensor<Real, TensorSymm::NONE, 3, 2> dGam_du;
      dGam_du.NewAthenaScratchTensor(member, scr_level, ncells1);
      AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> dA_ddd;      // A 1st drvts
      dGam_du.NewAthenaScratchTensor(member, scr_level, ncells1);
      AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> s_u;
      s_u.NewAthenaScratchTensor(member, scr_level, ncells1);
      AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dTheta_d;    // Theta 1st drvts
      dTheta_d.NewAthenaScratchTensor(member, scr_level, ncells1);

      Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};

      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> r;
      r.NewAthenaScratchTensor(member, scr_level, ncells1);
     
      par_for_inner(member, is, ie, [&](const int i) {
        if (((mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::outflow) && j == js) ||
            ((mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::outflow) && j == je) ||
            ((mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::outflow) && k == ks) ||
            ((mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::outflow) && k == ke) ||
            ((mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::outflow) && i == is) ||
            ((mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::outflow) && i == ie)) {
          for(int a = 0; a < 3; ++a) {
            dKhat_d   (a,i) = Dx<2>(a, idx, z4c.vKhat,  m,k,j,i); 
            dDtheta_d (a,i) = Dx<2>(a, idx, z4c.vTheta, m,k,j,i); 
            for(int b = 0; b < 3; ++b) {
              dGam_du (b,a,i) = Dx<2>(b, idx, z4c.vGam_u,  m,a,k,j,i);
              if (b >= a) {
                for(int c = 0; c < 3; ++c) {
                  dA_ddd(c,a,b,i) = Dx<2>(c, idx, z4c.vA_dd, m,a,b,k,j,i);
                }
              }
            }
          }
          // -----------------------------------------------------------------------------------
          // Compute pseudo-radial vector
          // 
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
          r(i) = std::sqrt(std::pow(x3v,2) + std::pow(x2v,2) + std::pow(x1v,2));
          s_u(0,i) = x1v/r(i);
          s_u(1,i) = x2v/r(i);
          s_u(2,i) = x3v/r(i);
          
          // -----------------------------------------------------------------------------------
          // Boundary RHS for scalars
          //
          rhs.vTheta(m,k,j,i) = - z4c.vTheta(m,k,j,i)/r(i);
          rhs.vKhat(m,k,j,i) = - std::sqrt(2.) * z4c.vKhat(m,k,j,i)/r(i);
          for(int a = 0; a < 3; ++a) {
            rhs.vTheta(m,k,j,i) -= s_u(a,i) * dTheta_d(a,i);
            rhs.vKhat(m,k,j,i) -= std::sqrt(2.) * s_u(a,i) * dKhat_d(a,i);
          }
          // -----------------------------------------------------------------------------------
          // Boundary RHS for the Gamma's
          //
          for(int a = 0; a < 3; ++a) {
            rhs.vGam_u(m,a,k,j,i) = - z4c.vGam_u(m,a,k,j,i)/r(i);
            for(int b = 0; b < 3; ++b) {
              rhs.vGam_u(m,a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
            }
          }
          // -----------------------------------------------------------------------------------
          // Boundary RHS for the A_ab
          //
          for(int a = 0; a < 3; ++a)
          for(int b = a; b < 3; ++b) {
            rhs.vA_dd(m,a,b,k,j,i) = - z4c.vA_dd(m,a,b,k,j,i)/r(i);
            for(int c = 0; c < 3; ++c) {
              rhs.vA_dd(m,a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
            }
          }
        }
      });
    });
  }
  return TaskStatus::complete;
}
*/

/*
//---------------------------------------------------------------------------------------
//! \fn TaskStatus Z4c::Z4cBoundaryRHS
//! \brief placeholder for the Sommerfield Boundary conditions for z4c
TaskStatus Z4c::Z4cBoundaryRHS(Driver *pdrive, int stage) {
  if (stage == pdrive->nexp_stages) {
    
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    auto &size = pmy_pack->pmb->mb_size;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    auto &z4c = pmy_pack->pz4c->z4c;
    auto &rhs = pmy_pack->pz4c->rhs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng); // Align scratch buffers with variables
    int nmb = pmy_pack->nmb_thispack;
    auto &mb_bcs = pmy_pack->pmb->mb_bcs;
 
    int scr_level = 0;
    size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)+ 
                      ScrArray2D<Real>::shmem_size(3,ncells1)*4+
                      ScrArray2D<Real>::shmem_size(9,ncells1)+
                      ScrArray2D<Real>::shmem_size(18,ncells1);

    // Note that the derivative Ds in Athena++ corresponds to Dx<2> here
    par_for_outer("Impose SBC",DevExeSpace(),
    scr_size,scr_level,0,nmb-1,ks,ke,js,je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {

      AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> r;
      r.NewAthenaScratchTensor(member, scr_level, ncells1);

      AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> s_u;
      s_u.NewAthenaScratchTensor(member, scr_level, ncells1);
      AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dTheta_d;    // Theta 1st drvts
      dTheta_d.NewAthenaScratchTensor(member, scr_level, ncells1);
      AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dKhat_d;
      dKhat_d.NewAthenaScratchTensor(member, scr_level, ncells1);
      AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dDtheta_d;
      dDtheta_d.NewAthenaScratchTensor(member, scr_level, ncells1);

      AthenaScratchTensor<Real, TensorSymm::NONE, 3, 2> dGam_du;
      dGam_du.NewAthenaScratchTensor(member, scr_level, ncells1);

      AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> dA_ddd;      // A 1st drvts
      dGam_du.NewAthenaScratchTensor(member, scr_level, ncells1);

      Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};

      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      
      // This implementation has a high content of repeated code. In particular 
      // the code inside every if condition is repeated, for a total of 6 times.
      // This is necessary because the use of functions inside a kernel is
      // not possible. 
      // Also, it is not possible to use a single if condition because otherwise 
      // the conditions would not be properly applied on MBs at the corner and at
      // the edges of the physical domain.
      par_for_inner(member, is, ie, [&](const int i) {

        // Boundary conditions on the y inner physical face
        if ((mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::outflow ||
             mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::extrapolate_outflow)
            && j == js) {
          for(int a = 0; a < 3; ++a) {
            dKhat_d   (a,i) = Dx<2>(a, idx, z4c.vKhat,  m,k,j,i); 
            dDtheta_d (a,i) = Dx<2>(a, idx, z4c.vTheta, m,k,j,i); 
            for(int b = 0; b < 3; ++b) {
              dGam_du (b,a,i) = Dx<2>(b, idx, z4c.vGam_u,  m,a,k,j,i);
              if (b >= a) {
                for(int c = 0; c < 3; ++c) {
                  dA_ddd(c,a,b,i) = Dx<2>(c, idx, z4c.vA_dd, m,a,b,k,j,i);
                }
              }
            }
          }
          // -----------------------------------------------------------------------------------
          // Compute pseudo-radial vector
          // 
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
          r(i) = std::sqrt(std::pow(x3v,2) + std::pow(x2v,2) + std::pow(x1v,2));
          s_u(0,i) = x1v/r(i);
          s_u(1,i) = x2v/r(i);
          s_u(2,i) = x3v/r(i);
          
          // -----------------------------------------------------------------------------------
          // Boundary RHS for scalars
          //
          rhs.vTheta(m,k,j,i) = - z4c.vTheta(m,k,j,i)/r(i);
          rhs.vKhat(m,k,j,i) = - std::sqrt(2.) * z4c.vKhat(m,k,j,i)/r(i);
          for(int a = 0; a < 3; ++a) {
            rhs.vTheta(m,k,j,i) -= s_u(a,i) * dTheta_d(a,i);
            rhs.vKhat(m,k,j,i) -= std::sqrt(2.) * s_u(a,i) * dKhat_d(a,i);
          }
          // -----------------------------------------------------------------------------------
          // Boundary RHS for the Gamma's
          //
          for(int a = 0; a < 3; ++a) {
            rhs.vGam_u(m,a,k,j,i) = - z4c.vGam_u(m,a,k,j,i)/r(i);
            for(int b = 0; b < 3; ++b) {
              rhs.vGam_u(m,a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
            }
          }
          // -----------------------------------------------------------------------------------
          // Boundary RHS for the A_ab
          //
          for(int a = 0; a < 3; ++a)
          for(int b = a; b < 3; ++b) {
            rhs.vA_dd(m,a,b,k,j,i) = - z4c.vA_dd(m,a,b,k,j,i)/r(i);
            for(int c = 0; c < 3; ++c) {
              rhs.vA_dd(m,a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
            }
          }
        }

        // Boundary conditions on the y outer physical face
        if ((mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::outflow ||
             mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::extrapolate_outflow)
            && j == je) {
          for(int a = 0; a < 3; ++a) {
            dKhat_d   (a,i) = Dx<2>(a, idx, z4c.vKhat,  m,k,j,i);
            dDtheta_d (a,i) = Dx<2>(a, idx, z4c.vTheta, m,k,j,i); 
            for(int b = 0; b < 3; ++b) {
              dGam_du (b,a,i) = Dx<2>(b, idx, z4c.vGam_u,  m,a,k,j,i);
              if (b >= a) {
                for(int c = 0; c < 3; ++c) {
                  dA_ddd(c,a,b,i) = Dx<2>(c, idx, z4c.vA_dd, m,a,b,k,j,i);
                }
              }
            }
          }
          // -----------------------------------------------------------------------------------
          // Compute pseudo-radial vector
          // 
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
          r(i) = std::sqrt(std::pow(x3v,2) + std::pow(x2v,2) + std::pow(x1v,2));
          s_u(0,i) = x1v/r(i);
          s_u(1,i) = x2v/r(i);
          s_u(2,i) = x3v/r(i);
          
          // -----------------------------------------------------------------------------------
          // Boundary RHS for scalars
          //
          rhs.vTheta(m,k,j,i) = - z4c.vTheta(m,k,j,i)/r(i);
          rhs.vKhat(m,k,j,i) = - std::sqrt(2.) * z4c.vKhat(m,k,j,i)/r(i);
          for(int a = 0; a < 3; ++a) {
            rhs.vTheta(m,k,j,i) -= s_u(a,i) * dTheta_d(a,i);
            rhs.vKhat(m,k,j,i) -= std::sqrt(2.) * s_u(a,i) * dKhat_d(a,i);
          }
          // -----------------------------------------------------------------------------------
          // Boundary RHS for the Gamma's
          //
          for(int a = 0; a < 3; ++a) {
            rhs.vGam_u(m,a,k,j,i) = - z4c.vGam_u(m,a,k,j,i)/r(i);
            for(int b = 0; b < 3; ++b) {
              rhs.vGam_u(m,a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
            }
          }
          // -----------------------------------------------------------------------------------
          // Boundary RHS for the A_ab
          //
          for(int a = 0; a < 3; ++a)
          for(int b = a; b < 3; ++b) {
            rhs.vA_dd(m,a,b,k,j,i) = - z4c.vA_dd(m,a,b,k,j,i)/r(i);
            for(int c = 0; c < 3; ++c) {
              rhs.vA_dd(m,a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
            }
          }
        }

        // Boundary conditions on the z inner physical face
        if ((mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::outflow ||
             mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::extrapolate_outflow)
            && k == ks) {
          for(int a = 0; a < 3; ++a) {
            dKhat_d   (a,i) = Dx<2>(a, idx, z4c.vKhat,  m,k,j,i); 
            dDtheta_d (a,i) = Dx<2>(a, idx, z4c.vTheta, m,k,j,i); 
            for(int b = 0; b < 3; ++b) {
              dGam_du (b,a,i) = Dx<2>(b, idx, z4c.vGam_u,  m,a,k,j,i);
              if (b >= a) {
                for(int c = 0; c < 3; ++c) {
                  dA_ddd(c,a,b,i) = Dx<2>(c, idx, z4c.vA_dd, m,a,b,k,j,i);
                }
              }
            }
          }
          // -----------------------------------------------------------------------------------
          // Compute pseudo-radial vector
          // 
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
          r(i) = std::sqrt(std::pow(x3v,2) + std::pow(x2v,2) + std::pow(x1v,2));
          s_u(0,i) = x1v/r(i);
          s_u(1,i) = x2v/r(i);
          s_u(2,i) = x3v/r(i);
          
          // -----------------------------------------------------------------------------------
          // Boundary RHS for scalars
          //
          rhs.vTheta(m,k,j,i) = - z4c.vTheta(m,k,j,i)/r(i);
          rhs.vKhat(m,k,j,i) = - std::sqrt(2.) * z4c.vKhat(m,k,j,i)/r(i);
          for(int a = 0; a < 3; ++a) {
            rhs.vTheta(m,k,j,i) -= s_u(a,i) * dTheta_d(a,i);
            rhs.vKhat(m,k,j,i) -= std::sqrt(2.) * s_u(a,i) * dKhat_d(a,i);
          }
          // -----------------------------------------------------------------------------------
          // Boundary RHS for the Gamma's
          //
          for(int a = 0; a < 3; ++a) {
            rhs.vGam_u(m,a,k,j,i) = - z4c.vGam_u(m,a,k,j,i)/r(i);
            for(int b = 0; b < 3; ++b) {
              rhs.vGam_u(m,a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
            }
          }
          // -----------------------------------------------------------------------------------
          // Boundary RHS for the A_ab
          //
          for(int a = 0; a < 3; ++a)
          for(int b = a; b < 3; ++b) {
            rhs.vA_dd(m,a,b,k,j,i) = - z4c.vA_dd(m,a,b,k,j,i)/r(i);
            for(int c = 0; c < 3; ++c) {
              rhs.vA_dd(m,a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
            }
          }
        }

        // Boundary conditions on the z outer physical face
        if ((mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::outflow ||
             mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::extrapolate_outflow)
            && k == ke) {
          for(int a = 0; a < 3; ++a) {
            dKhat_d   (a,i) = Dx<2>(a, idx, z4c.vKhat,  m,k,j,i); 
            dDtheta_d (a,i) = Dx<2>(a, idx, z4c.vTheta, m,k,j,i); 
            for(int b = 0; b < 3; ++b) {
              dGam_du (b,a,i) = Dx<2>(b, idx, z4c.vGam_u,  m,a,k,j,i);
              if (b >= a) {
                for(int c = 0; c < 3; ++c) {
                  dA_ddd(c,a,b,i) = Dx<2>(c, idx, z4c.vA_dd, m,a,b,k,j,i);
                }
              }
            }
          }
          // -----------------------------------------------------------------------------------
          // Compute pseudo-radial vector
          // 
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
          r(i) = std::sqrt(std::pow(x3v,2) + std::pow(x2v,2) + std::pow(x1v,2));
          s_u(0,i) = x1v/r(i);
          s_u(1,i) = x2v/r(i);
          s_u(2,i) = x3v/r(i);
          
          // -----------------------------------------------------------------------------------
          // Boundary RHS for scalars
          //
          rhs.vTheta(m,k,j,i) = - z4c.vTheta(m,k,j,i)/r(i);
          rhs.vKhat(m,k,j,i) = - std::sqrt(2.) * z4c.vKhat(m,k,j,i)/r(i);
          for(int a = 0; a < 3; ++a) {
            rhs.vTheta(m,k,j,i) -= s_u(a,i) * dTheta_d(a,i);
            rhs.vKhat(m,k,j,i) -= std::sqrt(2.) * s_u(a,i) * dKhat_d(a,i);
          }
          // -----------------------------------------------------------------------------------
          // Boundary RHS for the Gamma's
          //
          for(int a = 0; a < 3; ++a) {
            rhs.vGam_u(m,a,k,j,i) = - z4c.vGam_u(m,a,k,j,i)/r(i);
            for(int b = 0; b < 3; ++b) {
              rhs.vGam_u(m,a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
            }
          }
          // -----------------------------------------------------------------------------------
          // Boundary RHS for the A_ab
          //
          for(int a = 0; a < 3; ++a)
          for(int b = a; b < 3; ++b) {
            rhs.vA_dd(m,a,b,k,j,i) = - z4c.vA_dd(m,a,b,k,j,i)/r(i);
            for(int c = 0; c < 3; ++c) {
              rhs.vA_dd(m,a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
            }
          }
        }
      });
      // Boundary conditions on the x inner physical face; note: this is outside inner loop because i is
      // fixed at i = is
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::outflow ||
          mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::extrapolate_outflow) {
        int i = is;
        for(int a = 0; a < 3; ++a) {
          dKhat_d   (a,i) = Dx<2>(a, idx, z4c.vKhat,  m,k,j,i); 
          dDtheta_d (a,i) = Dx<2>(a, idx, z4c.vTheta, m,k,j,i); 
          for(int b = 0; b < 3; ++b) {
            dGam_du (b,a,i) = Dx<2>(b, idx, z4c.vGam_u,  m,a,k,j,i);
            if (b >= a) {
              for(int c = 0; c < 3; ++c) {
                dA_ddd(c,a,b,i) = Dx<2>(c, idx, z4c.vA_dd, m,a,b,k,j,i);
              }
            }
          }
        }
        // -----------------------------------------------------------------------------------
        // Compute pseudo-radial vector
        // 
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
        r(i) = std::sqrt(std::pow(x3v,2) + std::pow(x2v,2) + std::pow(x1v,2));
        s_u(0,i) = x1v/r(i);
        s_u(1,i) = x2v/r(i);
        s_u(2,i) = x3v/r(i);
        
        // -----------------------------------------------------------------------------------
        // Boundary RHS for scalars
        //
        rhs.vTheta(m,k,j,i) = - z4c.vTheta(m,k,j,i)/r(i);
        rhs.vKhat(m,k,j,i) = - std::sqrt(2.) * z4c.vKhat(m,k,j,i)/r(i);
        for(int a = 0; a < 3; ++a) {
          rhs.vTheta(m,k,j,i) -= s_u(a,i) * dTheta_d(a,i);
          rhs.vKhat(m,k,j,i) -= std::sqrt(2.) * s_u(a,i) * dKhat_d(a,i);
        }
        // -----------------------------------------------------------------------------------
        // Boundary RHS for the Gamma's
        //
        for(int a = 0; a < 3; ++a) {
          rhs.vGam_u(m,a,k,j,i) = - z4c.vGam_u(m,a,k,j,i)/r(i);
          for(int b = 0; b < 3; ++b) {
            rhs.vGam_u(m,a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
          }
        }
        // -----------------------------------------------------------------------------------
        // Boundary RHS for the A_ab
        //
        for(int a = 0; a < 3; ++a)
        for(int b = a; b < 3; ++b) {
          rhs.vA_dd(m,a,b,k,j,i) = - z4c.vA_dd(m,a,b,k,j,i)/r(i);
          for(int c = 0; c < 3; ++c) {
            rhs.vA_dd(m,a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
          }
        }
      }

      // Boundary conditions on the x outer physical face; note: this is outside inner loop because i is
      // fixed at i = ie
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::outflow ||
          mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::extrapolate_outflow) {
        int i = ie;
        for(int a = 0; a < 3; ++a) {
          dKhat_d   (a,i) = Dx<2>(a, idx, z4c.vKhat,  m,k,j,i); 
          dDtheta_d (a,i) = Dx<2>(a, idx, z4c.vTheta, m,k,j,i); 
          for(int b = 0; b < 3; ++b) {
            dGam_du (b,a,i) = Dx<2>(b, idx, z4c.vGam_u,  m,a,k,j,i);
            if (b >= a) {
              for(int c = 0; c < 3; ++c) {
                dA_ddd(c,a,b,i) = Dx<2>(c, idx, z4c.vA_dd, m,a,b,k,j,i);
              }
            }
          }
        }
        // -----------------------------------------------------------------------------------
        // Compute pseudo-radial vector
        // 
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
        r(i) = std::sqrt(std::pow(x3v,2) + std::pow(x2v,2) + std::pow(x1v,2));
        s_u(0,i) = x1v/r(i);
        s_u(1,i) = x2v/r(i);
        s_u(2,i) = x3v/r(i);
        
        // -----------------------------------------------------------------------------------
        // Boundary RHS for scalars
        //
        rhs.vTheta(m,k,j,i) = - z4c.vTheta(m,k,j,i)/r(i);
        rhs.vKhat(m,k,j,i) = - std::sqrt(2.) * z4c.vKhat(m,k,j,i)/r(i);
        for(int a = 0; a < 3; ++a) {
          rhs.vTheta(m,k,j,i) -= s_u(a,i) * dTheta_d(a,i);
          rhs.vKhat(m,k,j,i) -= std::sqrt(2.) * s_u(a,i) * dKhat_d(a,i);
        }
        // -----------------------------------------------------------------------------------
        // Boundary RHS for the Gamma's
        //
        for(int a = 0; a < 3; ++a) {
          rhs.vGam_u(m,a,k,j,i) = - z4c.vGam_u(m,a,k,j,i)/r(i);
          for(int b = 0; b < 3; ++b) {
            rhs.vGam_u(m,a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
          }
        }
        // -----------------------------------------------------------------------------------
        // Boundary RHS for the A_ab
        //
        for(int a = 0; a < 3; ++a)
        for(int b = a; b < 3; ++b) {
          rhs.vA_dd(m,a,b,k,j,i) = - z4c.vA_dd(m,a,b,k,j,i)/r(i);
          for(int c = 0; c < 3; ++c) {
            rhs.vA_dd(m,a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
          }
        }
      }
    });
  }
  return TaskStatus::complete;
}
*/

//---------------------------------------------------------------------------------------
//! \fn TaskStatus Z4c::Z4cBoundaryRHS
//! \brief placeholder for the Sommerfield Boundary conditions for z4c
TaskStatus Z4c::Z4cBoundaryRHS(Driver *pdrive, int stage) {
  if (stage == pdrive->nexp_stages) {
    
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    Z4cSommerfeld(is,is,js,je,ks,ke,BoundaryFace::inner_x1);
    Z4cSommerfeld(ie,ie,js,je,ks,ke,BoundaryFace::outer_x1);
    Z4cSommerfeld(is,ie,js,js,ks,ke,BoundaryFace::inner_x2);
    Z4cSommerfeld(is,ie,je,je,ks,ke,BoundaryFace::outer_x2);
    Z4cSommerfeld(is,ie,js,je,ks,ks,BoundaryFace::inner_x3);
    Z4cSommerfeld(is,ie,js,je,ke,ke,BoundaryFace::outer_x3);
 
  }
  return TaskStatus::complete;
}

void Z4c::Z4cSommerfeld(int const is, int const ie,
                        int const js, int const je,
                        int const ks, int const ke, 
                        int const bound) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  auto &z4c = pmy_pack->pz4c->z4c;
  auto &rhs = pmy_pack->pz4c->rhs;
  int nmb = pmy_pack->nmb_thispack;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  
  // Check this!
  int ncells1 = ie-is;
  ncells1 = ncells1 == 0 ? 1 : ncells1;
 
  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)+ 
                    ScrArray2D<Real>::shmem_size(3,ncells1)*4+
                    ScrArray2D<Real>::shmem_size(9,ncells1)+
                    ScrArray2D<Real>::shmem_size(18,ncells1);

  // Note that the derivative Ds in Athena++ corresponds to Dx<2> here
  par_for_outer("Impose SBC",DevExeSpace(),
  scr_size,scr_level,0,nmb-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {

    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> r;
    r.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> s_u;
    s_u.NewAthenaScratchTensor(member, scr_level, ncells1);
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dTheta_d;    // Theta 1st drvts
    dTheta_d.NewAthenaScratchTensor(member, scr_level, ncells1);
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dKhat_d;
    dKhat_d.NewAthenaScratchTensor(member, scr_level, ncells1);
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dDtheta_d;
    dDtheta_d.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 2> dGam_du;
    dGam_du.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> dA_ddd;      // A 1st drvts
    dGam_du.NewAthenaScratchTensor(member, scr_level, ncells1);

    Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
   

    
    par_for_inner(member, is, ie, [&](const int i) {
      // Boundary conditions on the y inner physical face
      if (mb_bcs.d_view(m,bound) == BoundaryFlag::outflow ||
          mb_bcs.d_view(m,bound) == BoundaryFlag::extrapolate_outflow) {
        for(int a = 0; a < 3; ++a) {
          dKhat_d   (a,i) = Dx<2>(a, idx, z4c.vKhat,  m,k,j,i); 
          dDtheta_d (a,i) = Dx<2>(a, idx, z4c.vTheta, m,k,j,i); 
          for(int b = 0; b < 3; ++b) {
            dGam_du (b,a,i) = Dx<2>(b, idx, z4c.vGam_u,  m,a,k,j,i);
            if (b >= a) {
              for(int c = 0; c < 3; ++c) {
                dA_ddd(c,a,b,i) = Dx<2>(c, idx, z4c.vA_dd, m,a,b,k,j,i);
              }
            }
          }
        }
        // -----------------------------------------------------------------------------------
        // Compute pseudo-radial vector
        // 
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
        r(i) = std::sqrt(std::pow(x3v,2) + std::pow(x2v,2) + std::pow(x1v,2));
        s_u(0,i) = x1v/r(i);
        s_u(1,i) = x2v/r(i);
        s_u(2,i) = x3v/r(i);
        
        // -----------------------------------------------------------------------------------
        // Boundary RHS for scalars
        //
        rhs.vTheta(m,k,j,i) = - z4c.vTheta(m,k,j,i)/r(i);
        rhs.vKhat(m,k,j,i) = - std::sqrt(2.) * z4c.vKhat(m,k,j,i)/r(i);
        for(int a = 0; a < 3; ++a) {
          rhs.vTheta(m,k,j,i) -= s_u(a,i) * dTheta_d(a,i);
          rhs.vKhat(m,k,j,i) -= std::sqrt(2.) * s_u(a,i) * dKhat_d(a,i);
        }
        // -----------------------------------------------------------------------------------
        // Boundary RHS for the Gamma's
        //
        for(int a = 0; a < 3; ++a) {
          rhs.vGam_u(m,a,k,j,i) = - z4c.vGam_u(m,a,k,j,i)/r(i);
          for(int b = 0; b < 3; ++b) {
            rhs.vGam_u(m,a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
          }
        }
        // -----------------------------------------------------------------------------------
        // Boundary RHS for the A_ab
        //
        for(int a = 0; a < 3; ++a)
        for(int b = a; b < 3; ++b) {
          rhs.vA_dd(m,a,b,k,j,i) = - z4c.vA_dd(m,a,b,k,j,i)/r(i);
          for(int c = 0; c < 3; ++c) {
            rhs.vA_dd(m,a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
          }
        }
      }
    });
  }); 
}

} // end namespace z4c

