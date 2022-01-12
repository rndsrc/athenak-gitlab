//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_fc.cpp
//! \brief functions to pack/send and recv/unpack/prolongate boundary values for
//! face-centered variables, implemented as part of the BValFC class.

#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
// BValFC constructor:

BoundaryValuesFC::BoundaryValuesFC(MeshBlockPack *pp,
                                   ParameterInput *pin) : BoundaryValues(pp, pin) {
}

//----------------------------------------------------------------------------------------
//! \!fn void BoundaryValuesFC::PackAndSendFC()
//! \brief Pack face-centered variables into boundary buffers and send to neighbors.
//!
//! As for cell-centered data, this routine packs ALL the buffers on ALL the faces, edges,
//! and corners simultaneously for all three components of face-fields on ALL the
//! MeshBlocks.
//!
//! Input array must be DvceFaceFld4D dimensioned (nmb, nx3, nx2, nx1)
//! DvceFaceFld4D of coarsened (restricted) fields also required with SMR/AMR

TaskStatus BoundaryValuesFC::PackAndSendFC(DvceFaceFld4D<Real> &b,
                                           DvceFaceFld4D<Real> &cb) {
  // create local references for variables in kernel
  int nmb = pmy_pack->pmb->nmb;
  int nnghbr = pmy_pack->pmb->nnghbr;

  {int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = send_buf;
  auto &rbuf = recv_buf;

  // load buffers, using 3 levels of hierarchical parallelism
  // Outer loop over (# of MeshBlocks)*(# of buffers)*(three field components)
  int nmnv = 3*nmb*nnghbr;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(3*nnghbr);
    const int n = (tmember.league_rank() - m*(3*nnghbr))/3;
    const int v = (tmember.league_rank() - m*(3*nnghbr) - 3*n);

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      // if neighbor is at coarser level, use cindices to pack buffer
      // Note indices can be different for each component of face-centered field.
      int il, iu, jl, ju, kl, ku, nv[3];
      nv[0] = 0;
      if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
        il = sbuf[n].icoar[v].bis;
        iu = sbuf[n].icoar[v].bie;
        jl = sbuf[n].icoar[v].bjs;
        ju = sbuf[n].icoar[v].bje;
        kl = sbuf[n].icoar[v].bks;
        ku = sbuf[n].icoar[v].bke;
        nv[1] = (sbuf[n].icoar[0].bie - sbuf[n].icoar[0].bis + 1)*
                (sbuf[n].icoar[0].bje - sbuf[n].icoar[0].bjs + 1)*
                (sbuf[n].icoar[0].bke - sbuf[n].icoar[0].bks + 1);
        nv[2] = nv[1] + (sbuf[n].icoar[1].bie - sbuf[n].icoar[1].bis + 1)*
                        (sbuf[n].icoar[1].bje - sbuf[n].icoar[1].bjs + 1)*
                        (sbuf[n].icoar[1].bke - sbuf[n].icoar[1].bks + 1);
      // if neighbor is at same level, use sindices to pack buffer
      } else if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
        il = sbuf[n].isame[v].bis;
        iu = sbuf[n].isame[v].bie;
        jl = sbuf[n].isame[v].bjs;
        ju = sbuf[n].isame[v].bje;
        kl = sbuf[n].isame[v].bks;
        ku = sbuf[n].isame[v].bke;
        nv[1] = (sbuf[n].isame[0].bie - sbuf[n].isame[0].bis + 1)*
                (sbuf[n].isame[0].bje - sbuf[n].isame[0].bjs + 1)*
                (sbuf[n].isame[0].bke - sbuf[n].isame[0].bks + 1);
        nv[2] = nv[1] + (sbuf[n].isame[1].bie - sbuf[n].isame[1].bis + 1)*
                        (sbuf[n].isame[1].bje - sbuf[n].isame[1].bjs + 1)*
                        (sbuf[n].isame[1].bke - sbuf[n].isame[1].bks + 1);
      // if neighbor is at finer level, use findices to pack buffer
      } else {
        il = sbuf[n].ifine[v].bis;
        iu = sbuf[n].ifine[v].bie;
        jl = sbuf[n].ifine[v].bjs;
        ju = sbuf[n].ifine[v].bje;
        kl = sbuf[n].ifine[v].bks;
        ku = sbuf[n].ifine[v].bke;
        nv[1] = (sbuf[n].ifine[0].bie - sbuf[n].ifine[0].bis + 1)*
                (sbuf[n].ifine[0].bje - sbuf[n].ifine[0].bjs + 1)*
                (sbuf[n].ifine[0].bke - sbuf[n].ifine[0].bks + 1);
        nv[2] = nv[1] + (sbuf[n].ifine[1].bie - sbuf[n].ifine[1].bis + 1)*
                        (sbuf[n].ifine[1].bje - sbuf[n].ifine[1].bjs + 1)*
                        (sbuf[n].ifine[1].bke - sbuf[n].ifine[1].bks + 1);
      }
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkj  = nk*nj;

      // indices of recv'ing MB and buffer: assumes MB IDs are stored sequentially
      int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
      int dn = nghbr.d_view(m,n).dest;

      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        // Inner (vector) loop over i
        // copy field components directly into recv buffer if MeshBlocks on same rank

        if (nghbr.d_view(m,n).rank == my_rank) {
          // if neighbor is at same or finer level, load data from b0
          if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
            if (v==0) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                rbuf[dn].vars(dm,i-il + ni*(j-jl + nj*(k-kl))) = b.x1f(m,k,j,i);
              });
            } else if (v==1) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                rbuf[dn].vars(dm,nv[1] + i-il + ni*(j-jl + nj*(k-kl))) = b.x2f(m,k,j,i);
              });
            } else {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                rbuf[dn].vars(dm,nv[2] + i-il + ni*(j-jl + nj*(k-kl))) = b.x3f(m,k,j,i);
              });
            }
          // if neighbor is at coarser level, load data from coarse_b0
          } else {
            if (v==0) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                rbuf[dn].vars(dm,i-il + ni*(j-jl + nj*(k-kl))) = cb.x1f(m,k,j,i);
              });
            } else if (v==1) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                rbuf[dn].vars(dm,nv[1] + i-il + ni*(j-jl + nj*(k-kl))) = cb.x2f(m,k,j,i);
              });
            } else {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                rbuf[dn].vars(dm,nv[2] + i-il + ni*(j-jl + nj*(k-kl))) = cb.x3f(m,k,j,i);
              });
            }
          }

        // else copy field components into send buffer for MPI communication below

        } else {
          // if neighbor is at same or finer level, load data from b0
          if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
            if (v==0) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                sbuf[n].vars(m,i-il + ni*(j-jl + nj*(k-kl))) = b.x1f(m,k,j,i);
              });
            } else if (v==1) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                sbuf[n].vars(m,nv[1] + i-il + ni*(j-jl + nj*(k-kl))) = b.x2f(m,k,j,i);
              });
            } else {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                sbuf[n].vars(m,nv[2] + i-il + ni*(j-jl + nj*(k-kl))) = b.x3f(m,k,j,i);
              });
            }
          // if neighbor is at coarser level, load data from coarse_b0
          } else {
            if (v==0) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                sbuf[n].vars(m,i-il + ni*(j-jl + nj*(k-kl))) = cb.x1f(m,k,j,i);
              });
            } else if (v==1) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                sbuf[n].vars(m,nv[1] + i-il + ni*(j-jl + nj*(k-kl))) = cb.x2f(m,k,j,i);
              });
            } else {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                sbuf[n].vars(m,nv[2] + i-il + ni*(j-jl + nj*(k-kl))) = cb.x3f(m,k,j,i);
              });
            }
          }
        }
      });
    } // end if-neighbor-exists block
  }); // end par_for_outer
  }

  // Send boundary buffer to neighboring MeshBlocks using MPI

  int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recv_buf;
  auto &mblev = pmy_pack->pmb->mb_lev;

  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {  // neighbor exists and not a physical boundary
        // index and rank of destination Neighbor
        int dn = nghbr.h_view(m,n).dest;
        int drank = nghbr.h_view(m,n).rank;

        // if MeshBlocks are on same rank, data already copied into receive buffer above
        // So simply set communication status tag as received.
        if (drank == my_rank) {
          // index of destination MeshBlock in this MBPack
          int dm = nghbr.h_view(m,n).gid - pmy_pack->gids;
          rbuf[dn].vars_stat[dm] = BoundaryCommStatus::received;
#if MPI_PARALLEL_ENABLED
        } else {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gidslist[drank];
          int tag = CreateMPITag(lid, dn);

          // get ptr to send buffer when neighbor is at coarser/same/fine level
          int data_size = 3;
          if (nghbr.h_view(m,n).lev < mblev.h_view(m)) {
            data_size *= send_buf[n].icoar_ndat;
          } else if (nghbr.h_view(m,n).lev == mblev.h_view(m)) {
            data_size *= send_buf[n].isame_ndat;
          } else {
            data_size *= send_buf[n].ifine_ndat;
          }
          void* send_ptr = &(send_buf[n].vars(m,0));

          int ierr = MPI_Isend(send_ptr, data_size, MPI_ATHENA_REAL, drank, tag,
                               vars_comm, &(send_buf[n].vars_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
#endif
        }
      }
    }
  }
  if (no_errors) return TaskStatus::complete;

  return TaskStatus::fail;
}

//----------------------------------------------------------------------------------------
// \!fn void RecvBuffers()
// \brief Unpack boundary buffers

TaskStatus BoundaryValuesFC::RecvAndUnpackFC(DvceFaceFld4D<Real> &b,
                                             DvceFaceFld4D<Real> &cb) {
  // create local references for variables in kernel
  int nmb = pmy_pack->pmb->nmb;
  int nnghbr = pmy_pack->pmb->nnghbr;

  bool bflag = false;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recv_buf;

#if MPI_PARALLEL_ENABLED
  // probe MPI communications.  This is a bit of black magic that seems to promote
  // communications to top of stack and gets them to complete more quickly
  int test;
  int ierr = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, vars_comm, &test, MPI_STATUS_IGNORE);
  if (ierr != MPI_SUCCESS) {return TaskStatus::incomplete;}
#endif

  //----- STEP 1: check that recv boundary buffer communications have all completed

  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) { // ID != -1, so not a physical boundary
        if (nghbr.h_view(m,n).rank == global_variable::my_rank) {
          if (rbuf[n].vars_stat[m] == BoundaryCommStatus::waiting) {bflag = true;}
#if MPI_PARALLEL_ENABLED
        } else {
          MPI_Test(&(rbuf[n].vars_req[m]), &test, MPI_STATUS_IGNORE);
          if (static_cast<bool>(test)) {
            rbuf[n].vars_stat[m] = BoundaryCommStatus::received;
          } else {
            bflag = true;
          }
#endif
        }
      }
    }
  }

  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}

  //----- STEP 2: buffers have all completed, so unpack 3-components of field

  auto &mblev = pmy_pack->pmb->mb_lev;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(three field components)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (3*nmb*nnghbr), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(3*nnghbr);
    const int n = (tmember.league_rank() - m*(3*nnghbr))/3;
    const int v = (tmember.league_rank() - m*(3*nnghbr) - 3*n);

    // only unpack buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      // if neighbor is at coarser level, use cindices to unpack buffer
      int il, iu, jl, ju, kl, ku, nv[3];
      nv[0] = 0;
      if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
        il = rbuf[n].icoar[v].bis;
        iu = rbuf[n].icoar[v].bie;
        jl = rbuf[n].icoar[v].bjs;
        ju = rbuf[n].icoar[v].bje;
        kl = rbuf[n].icoar[v].bks;
        ku = rbuf[n].icoar[v].bke;
        nv[1] = (rbuf[n].icoar[0].bie - rbuf[n].icoar[0].bis + 1)*
                (rbuf[n].icoar[0].bje - rbuf[n].icoar[0].bjs + 1)*
                (rbuf[n].icoar[0].bke - rbuf[n].icoar[0].bks + 1);
        nv[2] = nv[1] + (rbuf[n].icoar[1].bie - rbuf[n].icoar[1].bis + 1)*
                        (rbuf[n].icoar[1].bje - rbuf[n].icoar[1].bjs + 1)*
                        (rbuf[n].icoar[1].bke - rbuf[n].icoar[1].bks + 1);
      // if neighbor is at same level, use sindices to unpack buffer
      } else if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
        il = rbuf[n].isame[v].bis;
        iu = rbuf[n].isame[v].bie;
        jl = rbuf[n].isame[v].bjs;
        ju = rbuf[n].isame[v].bje;
        kl = rbuf[n].isame[v].bks;
        ku = rbuf[n].isame[v].bke;
        nv[1] = (rbuf[n].isame[0].bie - rbuf[n].isame[0].bis + 1)*
                (rbuf[n].isame[0].bje - rbuf[n].isame[0].bjs + 1)*
                (rbuf[n].isame[0].bke - rbuf[n].isame[0].bks + 1);
        nv[2] = nv[1] + (rbuf[n].isame[1].bie - rbuf[n].isame[1].bis + 1)*
                        (rbuf[n].isame[1].bje - rbuf[n].isame[1].bjs + 1)*
                        (rbuf[n].isame[1].bke - rbuf[n].isame[1].bks + 1);
      // if neighbor is at finer level, use findices to unpack buffer
      } else {
        il = rbuf[n].ifine[v].bis;
        iu = rbuf[n].ifine[v].bie;
        jl = rbuf[n].ifine[v].bjs;
        ju = rbuf[n].ifine[v].bje;
        kl = rbuf[n].ifine[v].bks;
        ku = rbuf[n].ifine[v].bke;
        nv[1] = (rbuf[n].ifine[0].bie - rbuf[n].ifine[0].bis + 1)*
                (rbuf[n].ifine[0].bje - rbuf[n].ifine[0].bjs + 1)*
                (rbuf[n].ifine[0].bke - rbuf[n].ifine[0].bks + 1);
        nv[2] = nv[1] + (rbuf[n].ifine[1].bie - rbuf[n].ifine[1].bis + 1)*
                        (rbuf[n].ifine[1].bje - rbuf[n].ifine[1].bjs + 1)*
                        (rbuf[n].ifine[1].bke - rbuf[n].ifine[1].bks + 1);
      }
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkj  = nk*nj;

      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        // Inner (vector) loop over i
        // copy contents of recv_buf into appropriate vector components

        // if neighbor is at same or finer level, load data directly into b0
        if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
          if (v==0) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              b.x1f(m,k,j,i) = rbuf[n].vars(m,i-il + ni*(j-jl + nj*(k-kl)));
            });
          } else if (v==1) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              b.x2f(m,k,j,i) = rbuf[n].vars(m,nv[1] + i-il + ni*(j-jl + nj*(k-kl)));
            });
          } else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              b.x3f(m,k,j,i) = rbuf[n].vars(m,nv[2] + i-il + ni*(j-jl + nj*(k-kl)));
            });
          }
        // if neighbor is at coarser level, load data into coarse_b0 (prolongate below)
        } else {
          if (v==0) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              cb.x1f(m,k,j,i) = rbuf[n].vars(m,i-il + ni*(j-jl + nj*(k-kl)));
            });
          } else if (v==1) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              cb.x2f(m,k,j,i) = rbuf[n].vars(m,nv[1] + i-il + ni*(j-jl + nj*(k-kl)));
            });
          } else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              cb.x3f(m,k,j,i) = rbuf[n].vars(m,nv[2] + i-il + ni*(j-jl + nj*(k-kl)));
            });
          }
        }
      });
    }  // end if-neighbor-exists block
  });  // end par_for_outer

  //----- STEP 3: Prolongate face-fields when neighbor at coarser level

  // Only perform prolongation with SMR/AMR
  if (pmy_pack->pmesh->multilevel) ProlongFC(b, cb);

  return TaskStatus::complete;
}
