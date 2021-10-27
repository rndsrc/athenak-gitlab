#ifndef RADIATION_RADIATION_HPP_
#define RADIATION_RADIATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.hpp
//  \brief definitions for Hydro class

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class EquationOfState;
class Coordinates;
class SourceTerms;
class Driver;

//----------------------------------------------------------------------------------------
//! \struct RadiationTaskIDs
//  \brief container to hold TaskIDs of all radiation tasks
  
struct RadiationTaskIDs
{   
  TaskID irecv;
  TaskID copyci;
  TaskID flux;
  TaskID expl;
  TaskID sendci;
  TaskID recvci;
  TaskID bcs;
  TaskID c2p;
  TaskID newdt;
  TaskID clear;
};

//----------------------------------------------------------------------------------------
//! \struct RegionIndcs
//! \brief Cell indices and number of active and ghost cells in the angular mesh

struct AMeshIndcs
{
  int ng;                   // number of ghost cells
  int nzeta,npsi;          // number of active cells (not including ghost zones)
  int zs,ze,ps,pe;          // indices of ACTIVE cells
};

//----------------------------------------------------------------------------------------
//! int AngleInd(int z, int p, bool zeta_face, bool psi_face, struct AMeshIndcs amidcs)
//! \brief Inline function for indexing angles

KOKKOS_INLINE_FUNCTION
int AngleInd(int z, int p, bool zeta_face, bool psi_face, struct AMeshIndcs amidcs)
{
  if (psi_face) {
    return z * (amidcs.npsi + 2*amidcs.ng + 1) + p;
  }
  return z * (amidcs.npsi + 2*amidcs.ng) + p;
}

namespace radiation {

//----------------------------------------------------------------------------------------
//! \class Hydro

class Radiation
{
public:
  Radiation(MeshBlockPack *ppack, ParameterInput *pin);
  ~Radiation();

  // data
  // flags to denote hydro or mhd is enabled
  bool is_hydro_enabled = false;
  bool is_mhd_enabled = false;

  ReconstructionMethod recon_method;
  EquationOfState *peos;  // chosen EOS
  SourceTerms *psrc = nullptr;

  AMeshIndcs amesh_indcs;  // indices of cells in angular mesh
  int nangles;  // number of angles

  DvceArray5D<Real> i0;

  DualArray1D<Real> zetaf;
  DualArray1D<Real> zetav;
  DualArray1D<Real> dzetaf;
  DualArray1D<Real> psif;
  DualArray1D<Real> psiv;
  DualArray1D<Real> dpsif;
  DualArray2D<Real> zeta_length;
  DualArray2D<Real> psi_length;
  DualArray2D<Real> solid_angle;

  DualArray3D<Real> nh_cc;
  DualArray3D<Real> nh_fc;
  DualArray3D<Real> nh_cf;
  DvceArray6D<Real> n0;
  DvceArray6D<Real> n0_n_0;
  DvceArray6D<Real> n1_n_0;
  DvceArray6D<Real> n2_n_0;
  DvceArray6D<Real> n3_n_0;
  DvceArray6D<Real> na1_n_0;
  DvceArray6D<Real> na2_n_0;

  DvceArray5D<Real> moments_coord;

  // Object containing boundary communication buffers and routines for u
  BoundaryValueCC *pbval_ci;

  // following only used for time-evolving flow
  DvceArray5D<Real> i1;       // conserved variables at intermediate step
  DvceFaceFld5D<Real> iflx;   // fluxes of conserved quantities on cell faces
  DvceArray5D<Real> ia1flx;   // fluxes of conserved quantities in zeta
  DvceArray5D<Real> ia2flx;   // fluxes of conserved quantities in psi
  Real dtnew;

  // container to hold names of TaskIDs
  RadiationTaskIDs id;

  // functions
  void AssembleRadiationTasks(TaskList &start, TaskList &run, TaskList &end);
  TaskStatus InitRecv(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus ExpRKUpdate(Driver *d, int stage);
  TaskStatus SendCI(Driver *d, int stage);
  TaskStatus RecvCI(Driver *d, int stage);
  TaskStatus SetRadMoments(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);  // in radiation/bvals dir

  // CalculateFluxes function
  TaskStatus CalcFluxes(Driver *d, int stage);

  // Initialze angular mesh and coordinate frame system
  void InitAngularMesh();
  void InitCoordinateFrame();

  // Moments functions
  void SetMoments(DvceArray5D<Real> &prim);

  // functions to set physical BCs for Radiation conserved variables, applied to single MB
  // specified by argument 'm'.
  void OutflowInnerX1(int m);
  void OutflowOuterX1(int m);
  void OutflowInnerX2(int m);
  void OutflowOuterX2(int m);
  void OutflowInnerX3(int m);
  void OutflowOuterX3(int m);

  void AngularMeshBoundaries();

private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Radiation
};

} // namespace radiation

#endif // RADIATION_RADIATION_HPP_
