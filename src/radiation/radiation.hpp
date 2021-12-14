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
  double ComputeWeightAndDualEdges(int lm, double length[6]) const;
  void GetGridCartPosition(int n, double *x, double *y, double *z) const;
  void GetGridCartPositionMid(int n, int nb, double *x, double *y, double *z) const;

  void CircumcenterNormalized(double x1, double x2, double x3,
                              double y1, double y2, double y3,
                              double z1, double z2, double z3,
                              double *x, double *y, double *z) const;
 
  void GetGridPositionPolar(int ic, double *theta, double *phi) const;
  void GreatCircleParam(double zeta1, double zeta2, double psi1, double psi2, double *apar, double *psi0) const;
  void UnitFluxDir(int ic1, int ic2, double *dtheta, double *dphi) const;
  void OptimalAngles(double ang[2]) const;
  void RotateGrid(double zeta, double psi);

  // TODO inline this
  double ArcLength(int ic1, int ic2) const;
  void ComputeXiEta(int lm, double xi[6], double eta[6]) const;

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
