//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_tasks.cpp
//  \brief implementation of functions that control Radiation tasks in the task list:

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "utils/create_mpitag.hpp"
#include "radiation/radiation.hpp"

namespace radiation {
//----------------------------------------------------------------------------------------
//! \fn  void Radiation::AssembleRadiationTasks
//  \brief Adds radiation tasks to stage start/run/end task lists
//  Called by MeshBlockPack::AddPhysicsModules() function after Radiation constrctr
//
//  Stage start tasks are those that must be cmpleted over all MeshBlocks before EACH
//  stage can be run (such as posting MPI receives, setting BoundaryCommStatus flags, etc)
//
//  Stage run tasks are those performed in EACH stage
//
//  Stage end tasks are those that can only be cmpleted after all the stage run tasks are
//  finished over all MeshBlocks for EACH stage, such as clearing all MPI non-blocking
//  sends, etc.

void Radiation::AssembleRadiationTasks(TaskList &start, TaskList &run, TaskList &end)
{
  TaskID none(0);

  // start task list
  id.irecv = start.AddTask(&Radiation::InitRecv, this, none);

  // run task list
  id.copyci = run.AddTask(&Radiation::CopyCons, this, none);
  id.flux = run.AddTask(&Radiation::CalcFluxes, this,id.copyci);
  id.expl  = run.AddTask(&Radiation::ExpRKUpdate, this, id.flux);
  id.sendci = run.AddTask(&Radiation::SendCI, this, id.expl);
  id.recvci = run.AddTask(&Radiation::RecvCI, this, id.sendci);
  id.bcs   = run.AddTask(&Radiation::ApplyPhysicalBCs, this, id.recvci);
  id.c2p   = run.AddTask(&Radiation::SetRadMoments, this, id.bcs);
  if (!(is_hydro_enabled || is_mhd_enabled)) {
    id.newdt = run.AddTask(&Radiation::NewTimeStep, this, id.c2p);
  }

  // end task list
  id.clear = end.AddTask(&Radiation::ClearSend, this, none);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::InitRecv
//  \brief function to post non-blocking receives (with MPI), and initialize all boundary
//  receive status flags to waiting (with or without MPI) for Radiation variables.

TaskStatus Radiation::InitRecv(Driver *pdrive, int stage)
{
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // Initialize communications for cell-centered conserved variables
  auto &rbufci = pbval_ci->recv_buf;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr[n].gid.h_view(m) >= 0) {
#if MPI_PARALLEL_ENABLED
        // post non-blocking receive if neighboring MeshBlock on a different rank 
        if (nghbr[n].rank.h_view(m) != global_variable::my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateMPITag(m, n, VariablesID::FluidCons_ID);
          auto recv_data = Kokkos::subview(rbufci[n].data, m, Kokkos::ALL, Kokkos::ALL);
          void* recv_ptr = recv_data.data();
          int ierr = MPI_Irecv(recv_ptr, recv_data.size(), MPI_ATHENA_REAL,
            nghbr[n].rank.h_view(m), tag, MPI_COMM_WORLD, &(rbufci[n].comm_req[m]));
        }
#endif
        // initialize boundary receive status flag
        rbufci[n].bcomm_stat(m) = BoundaryCommStatus::waiting;
      }
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::ClearRecv
//  \brief Waits for all MPI receives to complete before allowing execution to continue

TaskStatus Radiation::ClearRecv(Driver *pdrive, int stage)
{
#if MPI_PARALLEL_ENABLED
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking receives for U to finish before continuing 
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr[n].gid.h_view(m) >= 0) {
        if (nghbr[n].rank.h_view(m) != global_variable::my_rank) {
          MPI_Wait(&(pbval_ci->recv_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
        }
      }
    }
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::ClearSend
//  \brief Waits for all MPI sends to complete before allowing execution to continue

TaskStatus Radiation::ClearSend(Driver *pdrive, int stage)
{
#if MPI_PARALLEL_ENABLED
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking sends for U to finish before continuing 
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr[n].gid.h_view(m) >= 0) {
        if (nghbr[n].rank.h_view(m) != global_variable::my_rank) {
          MPI_Wait(&(pbval_ci->send_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
        }
      }
    }
  }
#endif
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  void Radiation::CopyCons
//  \brief  copy u0 --> u1 in first stage

TaskStatus Radiation::CopyCons(Driver *pdrive, int stage)
{
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), i1, i0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::SendU
//  \brief sends cell-centered conserved variables

TaskStatus Radiation::SendCI(Driver *pdrive, int stage)
{
  TaskStatus tstat = pbval_ci->SendBuffersCC(i0, VariablesID::FluidCons_ID);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::RecvU
//  \brief receives cell-centered conserved variables

TaskStatus Radiation::RecvCI(Driver *pdrive, int stage)
{
  TaskStatus tstat = pbval_ci->RecvBuffersCC(i0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::SetMoments
//  \brief compute radiation moments, only executed before outputs

TaskStatus Radiation::SetRadMoments(Driver *pdrive, int stage)
{
  if (stage == 0) {
    SetMoments(i0);
  }
  return TaskStatus::complete;
}

} // namespace radiation
