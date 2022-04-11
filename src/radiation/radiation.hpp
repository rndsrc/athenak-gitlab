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
  bool rad_source;         // flag to enable/disable radiation source term
  bool fixed_fluid;        // flag to enable/disable feedback of radiation field on fluid
  bool affect_fluid;       // flag to enable/disable feedback of radiation field on fluid
  Real arad;               // radiation constant
  Real kappa_a;            // Rosseland mean absoprtion coefficient
  Real kappa_s;            // scattering coefficient
  Real kappa_p;            // coefficient specifying difference b/w Rosseland and Planck
  bool constant_opacity;   // flag to enable opacity with fixed kappa
  bool power_opacity;      // flag to enable opacity that is powerlaw of density and temp

  // Object(s) for extra physics (i.e., other srcterms)
  SourceTerms *psrc = nullptr;

  // Angular mesh parameters and functions
  int nlevel;                         // geodesic nlevel
  int nangles;                        // number of angles
  bool rotate_geo;                    // rotate geodesic mesh
  bool angular_fluxes;                // flag to enable/disable angular fluxes
  bool moments_fluid;                 // flag to enable moment evaluation in fluid frame
  static const int not_a_patch = -1;  // set array elem that remain otherwise unaccessed
  DualArray4D<Real> amesh_normals;    // normal components for hexagonal faces
  DualArray2D<Real> ameshp_normals;   // normal components for pentagonal faces
  DualArray3D<Real> amesh_indices;    // indexing for hexagonal faces
  DualArray1D<Real> ameshp_indices;   // indexing for pentagonal faces
  DualArray1D<int>  num_neighbors;    // number of neighbors
  DualArray2D<int>  ind_neighbors;    // indicies of neighbors
  DualArray2D<Real> arc_lengths;      // arc lengths
  DualArray1D<Real> solid_angle;      // solid angles
  DualArray2D<Real> nh_c;             // normal vector computed at face center
  DualArray3D<Real> nh_f;             // normal vector computed at face edges
  DualArray2D<Real> xi_mn;            // xi angles
  DualArray2D<Real> eta_mn;           // eta angles
  DvceArray6D<Real> nmu;              // n^mu
  DvceArray6D<Real> n_mu;             // n_mu
  DvceArray5D<Real> n1_n_0;           // n^1*n_0
  DvceArray5D<Real> n2_n_0;           // n^2*n_0
  DvceArray5D<Real> n3_n_0;           // n^3*n_0
  DvceArray6D<Real> na_n_0;           // n^a*n_0
  DvceArray6D<Real> norm_to_tet;      // used in transform b/w fluid frame and coord frame
  DvceArray5D<Real> moments;          // moments of the radiation field
  int  GetNeighbors(int n, int neighbors[6]) const;
  Real ComputeWeightAndDualEdges(int n, Real length[6]) const;
  void GetGridCartPosition(int n, Real *x, Real *y, Real *z) const;
  void GetGridCartPositionMid(int n, int nb, Real *x, Real *y, Real *z) const;
  void CircumcenterNormalized(Real x1, Real x2, Real x3, Real y1, Real y2, Real y3,
                              Real z1, Real z2, Real z3, Real *x, Real *y, Real *z) const;
  void GetGridPositionPolar(int ic, Real *theta, Real *phi) const;
  void OptimalAngles(Real ang[2]) const;
  void RotateGrid(Real zeta, Real psi);
  Real ArcLength(int ic1, int ic2) const;
  void ComputeXiEta(int n, Real xi[6], Real eta[6]) const;
  void InitAngularMesh();
  void InitRadiationFrame();

  // intensity arrays
  DvceArray5D<Real> i0;         // intensities
  DvceArray5D<Real> coarse_i0;  // intensities on 2x coarser grid (for SMR/AMR)

  // Boundary communication buffers and functions for i
  BoundaryValuesCC *pbval_i;

  // following only used for time-evolving flow
  DvceArray5D<Real> i1;       // intensity at intermediate step
  DvceFaceFld5D<Real> iflx;   // spatial fluxes on zone faces
  DvceArray6D<Real> iaflx;    // angular fluxes on face edges
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

  // Functoin to set radiation moments
  void SetMoments(DvceArray5D<Real> &prim);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
};

// inline function to retrieve neighbors when in DevExeSpace()
KOKKOS_INLINE_FUNCTION
int DeviceGetNeighbors(int n, int nlvl, DualArray3D<Real> a_indcs, int neighbors[6]) {
  int num_neighbors;

  // handle north pole
  if (n==10*nlvl*nlvl) {
    for (int bl = 0; bl < 5; ++bl) {
      neighbors[bl] = a_indcs.d_view(bl,1,1);
    }
    neighbors[5] = -1;
    num_neighbors = 5;
  } else if (n == 10*nlvl*nlvl + 1) {  // handle south pole
    for (int bl = 0; bl < 5; ++bl) {
      neighbors[bl] = a_indcs.d_view(bl,nlvl,2*nlvl);
    }
    neighbors[5] = -1;
    num_neighbors = 5;
  } else {
    int ibl0 = (n / (2*nlvl*nlvl));
    int ibl1 = (n % (2*nlvl*nlvl)) / (2*nlvl);
    int ibl2 = (n % (2*nlvl*nlvl)) % (2*nlvl);
    neighbors[0] = a_indcs.d_view(ibl0, ibl1+1, ibl2+2);
    neighbors[1] = a_indcs.d_view(ibl0, ibl1+2, ibl2+1);
    neighbors[2] = a_indcs.d_view(ibl0, ibl1+2, ibl2);
    neighbors[3] = a_indcs.d_view(ibl0, ibl1+1, ibl2);
    neighbors[4] = a_indcs.d_view(ibl0, ibl1  , ibl2+1);

    // TODO(@gnwong, @pdmullen) check carefully, see if it can be inline optimized
    if (n % (2*nlvl*nlvl) == nlvl-1 || n % (2*nlvl*nlvl) == 2*nlvl-1) {
      neighbors[5] = -1;
      num_neighbors = 5;
    } else {
      neighbors[5] = a_indcs.d_view(ibl0, ibl1, ibl2+2);
      num_neighbors = 6;
    }
  }
  return num_neighbors;
}

} // namespace radiation
#endif // RADIATION_RADIATION_HPP_
