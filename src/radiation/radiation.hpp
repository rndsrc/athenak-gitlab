#ifndef RADIATION_RADIATION_HPP_
#define RADIATION_RADIATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.hpp
//  \brief definitions for Radiation class

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class EquationOfState;
class Coordinates;
class SourceTerms;
class Driver;


KOKKOS_INLINE_FUNCTION
int DeviceGetNeighbors(int lm, int nlvl,
                       DualArray3D<Real> a_indcs,
                       int neighbors[6])
{
  int num_neighbors;

  // handle north pole
  if (lm==10*nlvl*nlvl) {
    for (int bl = 0; bl < 5; ++bl) {
      neighbors[bl] = a_indcs.d_view(bl,1,1);
    }
    neighbors[5] = -1;
    num_neighbors = 5;
  } else if (lm == 10*nlvl*nlvl + 1) {  // handle south pole
    for (int bl = 0; bl < 5; ++bl) {
      neighbors[bl] = a_indcs.d_view(bl,nlvl,2*nlvl);
    }
    neighbors[5] = -1;
    num_neighbors = 5;
  } else {
    int ibl0 =  lm / (2*nlvl*nlvl);
    int ibl1 = (lm % (2*nlvl*nlvl)) / (2*nlvl);
    int ibl2 = (lm % (2*nlvl*nlvl)) % (2*nlvl);
    neighbors[0] = a_indcs.d_view(ibl0, ibl1+1, ibl2+2);
    neighbors[1] = a_indcs.d_view(ibl0, ibl1+2, ibl2+1);
    neighbors[2] = a_indcs.d_view(ibl0, ibl1+2, ibl2);
    neighbors[3] = a_indcs.d_view(ibl0, ibl1+1, ibl2);
    neighbors[4] = a_indcs.d_view(ibl0, ibl1  , ibl2+1);

    // TODO(@gnwong, @pdmullen) check carefully, see if it can be inline optimized
    if (lm % (2*nlvl*nlvl) == nlvl-1 || lm % (2*nlvl*nlvl) == 2*nlvl-1) {
      neighbors[5] = -1;
      num_neighbors = 5;
    } else {
      neighbors[5] = a_indcs.d_view(ibl0, ibl1, ibl2+2);
      num_neighbors = 6;
    }
  }
  return num_neighbors;
}


//----------------------------------------------------------------------------------------
//! \struct RadiationTaskIDs
//  \brief container to hold TaskIDs of all radiation tasks
  
struct RadiationTaskIDs
{   
  TaskID irecv;
  TaskID copyci;
  TaskID flux;
  TaskID expl;
  TaskID src;
  TaskID sendci;
  TaskID recvci;
  TaskID bcs;
  TaskID c2p;
  TaskID newdt;
  TaskID clear;
};

namespace radiation {
using OpacityFnPtr = void (*)(const Real rho, const Real temp,
                              Real& kappa_a, Real& kappa_s, Real& kappa_p);
}

namespace radiation {

//----------------------------------------------------------------------------------------
//! \class Hydro

class Radiation
{
public:
  static const int not_a_patch = -1;

public:
  Radiation(MeshBlockPack *ppack, ParameterInput *pin);
  ~Radiation();

  // data
  // flags to denote hydro or mhd is enabled
  bool is_hydro_enabled = false;
  bool is_mhd_enabled = false;
  bool is_rad_source_enabled = false;

  ReconstructionMethod recon_method;
  EquationOfState *peos;  // chosen EOS
  SourceTerms *psrc = nullptr;

  bool rad_source;
  bool coupling;
  bool arad;

  int nlevels;  // number of levels in geodesic grid
  int nangles;  // number of angles
  bool rotate_geo; // rotate geodesic grid (eliminating grid alignment)

  DvceArray5D<Real> i0;
  DvceArray5D<Real> moments_coord;

  // angular mesh quantities
  DualArray1D<Real> solid_angle;

  DualArray4D<Real> amesh_normals;  // includes ghost zones
  DualArray2D<Real> ameshp_normals;

  DvceArray6D<Real> nmu;
  DvceArray6D<Real> n_mu;
  DvceArray5D<Real> n1_n_0;
  DvceArray5D<Real> n2_n_0;
  DvceArray5D<Real> n3_n_0;
  DvceArray6D<Real> na_n_0;
  DvceArray6D<Real> norm_to_tet;

  // TODO(@gnwong) get rid of these arrays
  DualArray3D<Real> amesh_indices;  // includes ghost zones
  DualArray1D<Real> ameshp_indices;
  DualArray1D<int> num_neighbors;
  DualArray2D<int> ind_neighbors;
  DualArray2D<Real> arc_lengths;

  // TODO(@gnwong) almost certainly get rid of these arrays
  DualArray2D<Real> nh_c;
  DualArray3D<Real> nh_f;
  DualArray2D<Real> xi_mn;
  DualArray2D<Real> eta_mn;

  // Object containing boundary communication buffers and routines for i
  BoundaryValueCC *pbval_ci;

  // User-defined opacity function
  OpacityFnPtr OpacityFunc;

  // following only used for time-evolving flow
  DvceArray5D<Real> i1;       // conserved variables at intermediate step
  DvceFaceFld5D<Real> iflx;   // fluxes of conserved quantities on cell faces
  DvceArray6D<Real> iaflx;   // fluxes of conserved quantities in angle
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
  TaskStatus AddRadiationSourceTerm(Driver *d, int stage);
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

  // Helper geometry functions for geodesic mesh. Implemented in radiation_geom.cpp
  int GetNeighbors(int lm, int neighbors[6]) const;
  Real ComputeWeightAndDualEdges(int lm, Real length[6]) const;
  void GetGridCartPosition(int n, Real *x, Real *y, Real *z) const;
  void GetGridCartPositionMid(int n, int nb, Real *x, Real *y, Real *z) const;

  void CircumcenterNormalized(Real x1, Real x2, Real x3,
                              Real y1, Real y2, Real y3,
                              Real z1, Real z2, Real z3,
                              Real *x, Real *y, Real *z) const;
 
  void GetGridPositionPolar(int ic, Real *theta, Real *phi) const;
  void OptimalAngles(Real ang[2]) const;
  void RotateGrid(Real zeta, Real psi);

  // TODO inline this
  Real ArcLength(int ic1, int ic2) const;
  void ComputeXiEta(int lm, Real xi[6], Real eta[6]) const;

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

  // enroll user-defined opacity function
  void EnrollOpacityFunction(OpacityFnPtr my_opacityfunc);

private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Radiation
};

} // namespace radiation

#endif // RADIATION_RADIATION_HPP_
