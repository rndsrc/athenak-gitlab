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
  int nzeta, npsi;          // number of active cells (not including ghost zones)
  int zs,ze,ps,pe;          // indices of ACTIVE cells
};

//----------------------------------------------------------------------------------------
//! \struct AngleInd
//! \brief Inline function for indexing angles

KOKKOS_INLINE_FUNCTION
int AngleInd(int z, int p, bool zeta_face, bool psi_face, struct AMeshIndcs amidcs)
{
  if (psi_face) {
    return z * (amidcs.npsi + 2*amidcs.ng + 1) + p;
  }
  return z * (amidcs.npsi + 2*amidcs.ng) + p;
}

//----------------------------------------------------------------------------------------
//! \struct InverseAngleInd
//! \brief Inline function for inversing the 1D indexing.
// (TODO: @pdmullen) will this always work? I think there are some limitations on
// what npsi can be given an nzeta.  Also, could round-off error differences
// lead to the incorrect indices being identified.

KOKKOS_INLINE_FUNCTION
void InverseAngleInd(int zm, int& z, int& p, struct AMeshIndcs amidcs)
{
  z = (int) (((Real) zm)/(2*amidcs.ng+amidcs.npsi));
  p = zm - z * (amidcs.npsi + 2*amidcs.ng);
  if ((z * (amidcs.npsi + 2*amidcs.ng) + p) != zm) {
    z = NAN;
    p = NAN;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ComputeMetricAndInverse
//! \brief computes covariant and contravariant components of Cartesian tetrad.  For now,
//! assume Minkowski space.  Taken from gr_rad branch

KOKKOS_INLINE_FUNCTION
void ComputeTetrad(Real x, Real y, Real z, bool minkowski,
                   Real e[][4], Real ecov[][4], Real omega[][4][4])
{
  // zero the passed arrays
  for (int i=0; i<4; ++i) {
    for (int j=0; j<4; ++j) {
      e[i][j] = 0.0; ecov[i][j] = 0.0;
      for (int k=0; k<4; ++k) {
        omega[i][j][k] = 0.0;
      }
    }
  }

  // allocate intermediate arrays
  Real eta[4][4] = {};
  Real g[4][4] = {};
  Real ei[4][4] = {};
  Real de[4][4][4] = {};

  // set Minkowski metric
  eta[0][0] = -1.0;
  eta[1][1] = 1.0;
  eta[2][2] = 1.0;
  eta[3][3] = 1.0;

  // set covariant metric
  // (TODO: @pdmullen): for now assume Minkowski; later will likely call
  // ComputeMetricAndInverse, however, note that we use g[4][4] rather than
  // g[NMETRIC] here.  The former is convienent for summations below.
  g[0][0] = -1.0;
  g[1][1] = 1.0;
  g[2][2] = 1.0;
  g[3][3] = 1.0;

  // define Cartesian tetrad
  // (TODO: @pdmullen):  again, for now assume Minkowski; need to derive a
  // Cartesian tetrad for CKS.
  e[0][0] = 1.0;
  e[1][1] = 1.0;
  e[2][2] = 1.0;
  e[3][3] = 1.0;

  // Calculate covariant tetrad and inverse of tetrad
  for (int i=0; i<4; ++i) {
    for (int j=0; j<4; ++j) {
      for (int k=0; k<4; ++k) {
        ecov[i][j] += g[j][k]*e[i][k];
        for (int l=0; l<4; ++l) {
          ei[i][j] += eta[i][k]*g[j][l]*e[k][l];
        }
      }
    }
  }

  // Calculate Ricci rotation coefficients
  for (int i=0; i<4; ++i) {
    for (int j=0; j<4; ++j) {
      for (int k=0; k<4; ++k) {
        for (int l=0; l<4; ++l) {
          for (int m=0; m<4; ++m) {
            omega[i][j][k] += ei[i][l]*e[k][m]*de[m][j][l];
          }
        }
      }
    }
  }

  return;
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

  DvceArray5D<Real> ci0;   // conserved variables
  DvceArray5D<Real> i0;    // primitive variables

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

  DvceArray7D<Real> nmu;
  DvceArray7D<Real> n0_n_mu;
  DvceArray7D<Real> n1_n_mu;
  DvceArray7D<Real> n2_n_mu;
  DvceArray7D<Real> n3_n_mu;
  DvceArray6D<Real> na1_n_0;
  DvceArray6D<Real> na2_n_0;

  DvceArray5D<Real> moments_coord;

  // Object containing boundary communication buffers and routines for u
  BoundaryValueCC *pbval_ci;

  // following only used for time-evolving flow
  DvceArray5D<Real> ci1;       // conserved variables at intermediate step
  DvceFaceFld5D<Real> ciflx;   // fluxes of conserved quantities on cell faces
  DvceArray5D<Real> cia1flx;   // fluxes of conserved quantities in zeta
  DvceArray5D<Real> cia2flx;   // fluxes of conserved quantities in psi
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
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);  // in radiation/bvals dir

  // CalculateFluxes function
  TaskStatus CalcFluxes(Driver *d, int stage);

  // Initialze angular mesh and coordinate frame system
  void InitMesh();
  void InitCoordinateFrame();

  // Moments functions
  void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim);
  void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons);

  // functions to set physical BCs for Radiation conserved variables, applied to single MB
  // specified by argument 'm'.
  void OutflowInnerX1(int m);
  void OutflowOuterX1(int m);
  void OutflowInnerX2(int m);
  void OutflowOuterX2(int m);
  void OutflowInnerX3(int m);
  void OutflowOuterX3(int m);

private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Radiation
};

} // namespace radiation
#endif // RADIATION_RADIATION_HPP_
