//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_tasks.cpp
//  \brief implementation of functions that control MHD tasks in the four task lists:
//  stagestart_tl, stagerun_tl, stageend_tl
//  operatorsplit_tl

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "bvals/bvals.hpp"
#include "utils/create_mpitag.hpp"
#include "srcterms/srcterms.hpp"
#include "mhd/mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn  void MHD::AssembleStageStartTasks
//  \brief adds MHD tasks to stage start TaskList
//  These are taks that must be cmpleted (such as posting MPI receives, setting 
//  BoundaryCommStatus flags, etc) over all MeshBlocks before stage can be run.
//  Called by MeshBlockPack::AddPhysicsModules() function directly after MHD constrctr

void MHD::AssembleStageStartTasks(TaskList &tl, TaskID start)
{
  auto mhd_init = tl.AddTask(&MHD::InitRecv, this, start);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::AssembleStageRunTasks
//  \brief adds MHD tasks to stage run TaskList
//  Called by MeshBlockPack::AddPhysicsModules() function directly after MHD constrctr

void MHD::AssembleStageRunTasks(TaskList &tl, TaskID start)
{
  auto mhd_copycons = tl.AddTask(&MHD::CopyCons, this, start);
  auto mhd_fluxes = tl.AddTask(&MHD::CalcFluxes, this, mhd_copycons);
  auto visc_fluxes = tl.AddTask(&MHD::ViscousFluxes, this, mhd_fluxes);
  auto mhd_update = tl.AddTask(&MHD::Update, this, visc_fluxes);
  auto mhd_src = tl.AddTask(&MHD::UpdateUnsplitSourceTerms, this, mhd_update);
  auto mhd_sendu = tl.AddTask(&MHD::SendU, this, mhd_src);
  auto mhd_recvu = tl.AddTask(&MHD::RecvU, this, mhd_sendu);
  auto mhd_emf = tl.AddTask(&MHD::CornerE, this, mhd_recvu);
  auto resist_emf = tl.AddTask(&MHD::ResistEMF, this, mhd_emf);
  auto mhd_ct = tl.AddTask(&MHD::CT, this, resist_emf);
  auto mhd_sendb = tl.AddTask(&MHD::SendB, this, mhd_ct);
  auto mhd_recvb = tl.AddTask(&MHD::RecvB, this, mhd_sendb);
  auto mhd_phybcs = tl.AddTask(&MHD::ApplyPhysicalBCs, this, mhd_recvb);
  auto mhd_con2prim = tl.AddTask(&MHD::ConToPrim, this, mhd_phybcs);
  auto mhd_newdt = tl.AddTask(&MHD::NewTimeStep, this, mhd_con2prim);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::AssembleStageEndTasks
//  \brief adds MHD tasks to stage end TaskList
//  These are tasks that can only be cmpleted after all the stage run tasks are finished
//  over all MeshBlocks, such as clearing all MPI non-blocking sends, etc.
//  Called by MeshBlockPack::AddPhysicsModules() function directly after MHD constrctr

void MHD::AssembleStageEndTasks(TaskList &tl, TaskID start)
{
  auto mhd_clear = tl.AddTask(&MHD::ClearSend, this, start);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::AssmebleOperatorSplitTasks
//  \brief adds MHD tasks to operator split TaskList
//  Called by MeshBlockPack::AddPhysicsModules() function directly after MHD constrctr

void MHD::AssembleOperatorSplitTasks(TaskList &tl, TaskID start)
{
  if (not (psrc->operatorsplit_terms)) {return;}
  auto split_srcterms = tl.AddTask(&MHD::UpdateOperatorSplitSourceTerms, this, start);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::InitRecv
//  \brief function to post non-blocking receives (with MPI), and initialize all boundary
//  receive status flags to waiting (with or without MPI).  Note this must be done for
//  communication of BOTH conserved (cell-centered) and face-centered fields

TaskStatus MHD::InitRecv(Driver *pdrive, int stage)
{
  int &nmb = pmy_pack->pmb->nmb;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // Initialize communications for both cell-centered conserved variables and 
  // face-centered magnetic fields
  auto &rbufu = pbval_u->recv_buf;
  auto &rbufb = pbval_b->recv_buf;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr[n].gid.h_view(m) >= 0) {
#if MPI_PARALLEL_ENABLED
        // post non-blocking receive if neighboring MeshBlock on a different rank 
        if (nghbr[n].rank.h_view(m) != global_variable::my_rank) {
          {
          // Receive requests for U
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateMPITag(m, n, VariablesID::FluidCons_ID);
          auto recv_data = Kokkos::subview(rbufu[n].data, m, Kokkos::ALL, Kokkos::ALL);
          void* recv_ptr = recv_data.data();
          int ierr = MPI_Irecv(recv_ptr, recv_data.size(), MPI_ATHENA_REAL,
            nghbr[n].rank.h_view(m), tag, MPI_COMM_WORLD, &(rbufu[n].comm_req[m]));
          }

          {
          // Receive requests for B
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateMPITag(m, n, VariablesID::BField_ID);
          auto recv_data = Kokkos::subview(rbufb[n].data, m, Kokkos::ALL, Kokkos::ALL);
          void* recv_ptr = recv_data.data();
          int ierr = MPI_Irecv(recv_ptr, recv_data.size(), MPI_ATHENA_REAL,
            nghbr[n].rank.h_view(m), tag, MPI_COMM_WORLD, &(rbufb[n].comm_req[m]));
          }
        }
#endif
        // initialize boundary receive status flag
        rbufu[n].bcomm_stat(m) = BoundaryCommStatus::waiting;
        rbufb[n].bcomm_stat(m) = BoundaryCommStatus::waiting;
      }
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::ClearRecv
//  \brief Waits for all MPI receives to complete before allowing execution to continue
//  With MHD, clears both receives of U and B

TaskStatus MHD::ClearRecv(Driver *pdrive, int stage)
{
#if MPI_PARALLEL_ENABLED
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking receives for U and B to finish before continuing 
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr[n].gid.h_view(m) >= 0) {
        if (nghbr[n].rank.h_view(m) != global_variable::my_rank) {
          MPI_Wait(&(pbval_u->recv_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
          MPI_Wait(&(pbval_b->recv_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
        }
      }
    }
  }
#endif
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  void MHD::ClearSend
//  \brief Waits for all MPI sends to complete before allowing execution to continue
//  With MHD, clears both sends of U and B

TaskStatus MHD::ClearSend(Driver *pdrive, int stage)
{
#if MPI_PARALLEL_ENABLED
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking sends for U and B to finish before continuing 
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr[n].gid.h_view(m) >= 0) {
        if (nghbr[n].rank.h_view(m) != global_variable::my_rank) {
          MPI_Wait(&(pbval_u->send_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
          MPI_Wait(&(pbval_b->send_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
        }
      }
    }
  }
#endif
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  void MHD::CopyCons
//  \brief  copy u0 --> u1, and b0 --> b1 in first stage

TaskStatus MHD::CopyCons(Driver *pdrive, int stage)
{
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
    Kokkos::deep_copy(DevExeSpace(), b1.x1f, b0.x1f);
    Kokkos::deep_copy(DevExeSpace(), b1.x2f, b0.x2f);
    Kokkos::deep_copy(DevExeSpace(), b1.x3f, b0.x3f);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::SendU
//  \brief sends cell-centered conserved variables

TaskStatus MHD::SendU(Driver *pdrive, int stage) 
{
  TaskStatus tstat = pbval_u->SendBuffersCC(u0, VariablesID::FluidCons_ID);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::SendB
//  \brief sends face-centered magnetic fields

TaskStatus MHD::SendB(Driver *pdrive, int stage)
{
  TaskStatus tstat = pbval_b->SendBuffersFC(b0, VariablesID::BField_ID);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::RecvU
//  \brief receives cell-centered conserved variables

TaskStatus MHD::RecvU(Driver *pdrive, int stage)
{
  TaskStatus tstat = pbval_u->RecvBuffersCC(u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::RecvB
//  \brief receives face-centered magnetic fields

TaskStatus MHD::RecvB(Driver *pdrive, int stage)
{
  TaskStatus tstat = pbval_b->RecvBuffersFC(b0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::ConToPrim
//  \brief

TaskStatus MHD::ConToPrim(Driver *pdrive, int stage)
{
  peos->ConsToPrim(u0, b0, w0, bcc0);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::ViscousFluxes
//  \brief

TaskStatus MHD::ViscousFluxes(Driver *pdrive, int stage)
{
  if (pvisc != nullptr) pvisc->AddViscousFlux(u0, uflx);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::ResistEMF
//  \brief

TaskStatus MHD::ResistEMF(Driver *pdrive, int stage)
{
  if (presist != nullptr) presist->AddResistiveEMF(b0, efld);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::UpdateUnsplitSourceTerms
//  \brief adds source terms to MHD variables in EACH stage of stage run task list.
//  These are source terms that will be included as an unsplit algorithm.
//  This task is always included in the StageRun tasklist (see AssembleStageRunTasks()
//  function above), so a return test is needed in the case of no source terms.

TaskStatus MHD::UpdateUnsplitSourceTerms(Driver *pdrive, int stage)
{
  // return if no source terms included
  if (not (psrc->stagerun_terms)) {return TaskStatus::complete;}

  // apply source terms update to conserved variables
  psrc->ApplySrcTermsStageRunTL(u0,w0,stage);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::UpdateOperatorSplitSourceTerms
//  \brief adds source terms to MHD variables in operator split task list.
//  These are source terms that will be included as an operator split algorithm.
//  This task is not included in the OperatorSplit tasklist if there are no operator split
//  source terms to be inlcuded (see AssembleOperatorSplitTasks() function above), so no
//  return test is needed in the case of no source terms.

TaskStatus MHD::UpdateOperatorSplitSourceTerms(Driver *pdrive, int stage)
{
  psrc->ApplySrcTermsOperatorSplitTL(u0);
  return TaskStatus::complete;
}

} // namespace mhd
