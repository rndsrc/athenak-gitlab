#ifndef RADIATION_RADIATION_HPP_
#define RADIATION_RADIATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation.hpp
//  \brief definitions for Radiation class

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

#define HUGE_NUMBER 1.0e+36

// forward declarations
class EquationOfState;
class Coordinates;
class SourceTerms;
class Driver;

//----------------------------------------------------------------------------------------
//! \struct RadiationTaskIDs
//  \brief container to hold TaskIDs of all radiation tasks

struct RadiationTaskIDs {
  TaskID rad_irecv;
  TaskID hyd_irecv;
  TaskID mhd_irecv;

  TaskID copycons;

  TaskID rad_flux;
  TaskID rad_sendf;
  TaskID rad_recvf;
  TaskID hyd_flux;
  TaskID hyd_sendf;
  TaskID hyd_recvf;
  TaskID mhd_flux;
  TaskID mhd_sendf;
  TaskID mhd_recvf;

  TaskID rad_expl;
  TaskID hyd_expl;
  TaskID mhd_expl;

  TaskID rad_src;

  TaskID rad_resti;
  TaskID rad_sendi;
  TaskID rad_recvi;
  TaskID hyd_restu;
  TaskID hyd_sendu;
  TaskID hyd_recvu;
  TaskID mhd_restu;
  TaskID mhd_sendu;
  TaskID mhd_recvu;

  TaskID mhd_efld;
  TaskID mhd_sende;
  TaskID mhd_recve;
  TaskID mhd_ct;
  TaskID mhd_restb;
  TaskID mhd_sendb;
  TaskID mhd_recvb;

  TaskID rad_bcs;
  TaskID hyd_bcs;
  TaskID mhd_bcs;

  TaskID hyd_c2p;
  TaskID mhd_c2p;

  TaskID rad_clear;
  TaskID hyd_clear;
  TaskID mhd_clear;
};

//----------------------------------------------------------------------------------------
//! \struct RegionIndcs
//! \brief Cell indices and number of active and ghost cells in the angular mesh

struct AMeshIndcs {
  int ng;                   // number of ghost cells
  int nzeta,npsi;           // number of active cells (not including ghost zones)
  int zs,ze,ps,pe;          // indices of ACTIVE cells
};

//----------------------------------------------------------------------------------------
//! int AngleInd(int z, int p, bool zeta_face, bool psi_face, struct AMeshIndcs amidcs)
//! \brief Inline function for indexing angles

KOKKOS_INLINE_FUNCTION
int AngleInd(int z, int p, bool zeta_face, bool psi_face, struct AMeshIndcs amidcs) {
  if (psi_face) {
    return z * (amidcs.npsi + 2*amidcs.ng + 1) + p;
  }
  return z * (amidcs.npsi + 2*amidcs.ng) + p;
}

namespace radiation {

//----------------------------------------------------------------------------------------
//! \class Radiation

class Radiation {
 public:
  Radiation(MeshBlockPack *ppack, ParameterInput *pin);
  ~Radiation();

  // flags to denote hydro or mhd is enabled
  bool is_hydro_enabled;
  bool is_mhd_enabled;

  // Radiation source term parameters
  bool rad_source;            // flag to enable/disable radiation source term
  bool fixed_fluid;           // flag to enable/disable fluid integration
  bool affect_fluid;          // flag to enable/disable feedback of rad field on fluid
  bool zero_radiation_force;  // flag to enable/disable radiation momentum force
  Real arad;                  // radiation constant
  Real kappa_a;               // Rosseland mean absoprtion coefficient
  Real kappa_s;               // scattering coefficient
  Real kappa_p;               // coefficient specifying diff b/w Rosseland and Planck
  bool constant_opacity;      // flag to enable opacity with fixed kappa
  bool power_opacity;         // flag to enable opacity that is powerlaw of rho and temp

  // Object(s) for extra physics (i.e., other srcterms)
  SourceTerms *psrc = nullptr;

  // Angular mesh parameters and functions
  AMeshIndcs amesh_indcs;             // indices of cells in angular mesh
  int nangles;                        // number of angles
  bool angular_fluxes;                // flag to enable/disable angular fluxes
  DualArray1D<Real> zetaf;
  DualArray1D<Real> zetav;
  DualArray1D<Real> dzetaf;
  DualArray1D<Real> psif;
  DualArray1D<Real> psiv;
  DualArray1D<Real> dpsif;
  DualArray2D<Real> zeta_length;
  DualArray2D<Real> psi_length;
  DualArray2D<Real> solid_angle;

  DualArray3D<Real> nh_c;             // normal vector computed at face center
  DualArray3D<Real> nh_zf;            // normal vector computed at zeta faces
  DualArray3D<Real> nh_pf;            // normal vector computed at psi  faces
  DvceArray6D<Real> tet_c;            // tetrad components at cell centers
  DvceArray6D<Real> tetcov_c;         // covariant tetrad components at cell centers
  DvceArray5D<Real> tet_d1_x1f;       // tetrad components (subset) at x1f
  DvceArray5D<Real> tetcov_d0_x1f;;   // covariant tetrad components (subset) at x1f
  DvceArray5D<Real> tet_d2_x2f;       // tetrad components (subset) at x2f
  DvceArray5D<Real> tetcov_d0_x2f;    // covariant tetrad components (subset) at x2f
  DvceArray5D<Real> tet_d3_x3f;       // tetrad components (subset) at x3f
  DvceArray5D<Real> tetcov_d0_x3f;    // covariant tetrad components (subset) at x3f
  DvceArray7D<Real> ricci;            // Ricci rotation coefficients
  DvceArray6D<Real> norm_to_tet;      // used in transform b/w normal frame and tet frame
  void InitAngularMesh();
  void SetOrthonormalTetrad();

  // intensity arrays
  DvceArray5D<Real> i0;         // intensities
  DvceArray5D<Real> coarse_i0;  // intensities on 2x coarser grid (for SMR/AMR)

  // Boundary communication buffers and functions for i
  BoundaryValuesCC *pbval_i;

  // following only used for time-evolving flow
  DvceArray5D<Real> i1;        // intensity at intermediate step
  DvceFaceFld5D<Real> iflx;    // spatial fluxes on zone faces
  DvceArray5D<Real> ia1flx;    // angular fluxes on face edges
  DvceArray5D<Real> ia2flx;    // angular fluxes on face edges
  Real dtnew;

  // reconstruction method
  ReconstructionMethod recon_method;

  // container to hold names of TaskIDs
  RadiationTaskIDs id;

  // TaskStatus functions
  void AssembleRadiationTasks(TaskList &start, TaskList &run, TaskList &end);
  TaskStatus InitRecv(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus ExpRKUpdate(Driver *d, int stage);
  TaskStatus AddRadiationSourceTerm(Driver *d, int stage);
  TaskStatus SendI(Driver *d, int stage);
  TaskStatus RecvI(Driver *d, int stage);
  TaskStatus CalcFluxes(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus RestrictI(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
};

} // namespace radiation
#endif // RADIATION_RADIATION_HPP_
