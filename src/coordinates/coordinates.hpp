#ifndef COORDINATES_COORDINATES_HPP_
#define COORDINATES_COORDINATES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coordinates.hpp
//! \brief implemention of light-weight coordinates class.  Provides data structure that
//! stores array of RegionSizes over (# of MeshBlocks), and inline functions for
//! computing positions.  In GR, also provides inline metric functions (currently only
//! Cartesian Kerr-Schild)

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"

// forward declarations
struct EOS_Data;

//----------------------------------------------------------------------------------------
//! \struct CoordData
//! \brief container for Coordinate variables and functions needed inside kernels. Storing
//! everything in a container makes them easier to capture, and pass to inline functions,
//! inside kernels.

struct CoordData {
  // following data is only used in GR calculations to compute metric
  bool is_minkowski;               // flag to specify Minkowski (flat) space
  Real bh_mass;                    // needed for GR metric
  Real bh_spin;                    // needed for GR metric
  bool bh_excise;                  // flag to specify excision
  Real dexcise;                    // rest-mass density inside excised region
  Real pexcise;                    // pressure inside excised region
  bool excise_rad=false;           // excise radiation field
};

//----------------------------------------------------------------------------------------
//! \class Coordinates
//! \brief data and functions for coordinates

class Coordinates {
 public:
  explicit Coordinates(ParameterInput *pin, MeshBlockPack *ppack);
  ~Coordinates() {}

  // flags to denote relativistic dynamics in these coordinates
  bool is_special_relativistic = false;
  bool is_general_relativistic = false;

  // data needed to compute metric in GR
  CoordData coord_data;

  // excision masks
  DvceArray4D<bool> cc_mask;          // cell-centered mask for C2P
  DvceArray4D<bool> cc_rad_mask;      // cell-centered mask for radiation
  DvceFaceFld4D<bool> fc_mask;        // face-centered mask to enable first-order fluxes

  // functions
  void AddCoordTerms(const DvceArray5D<Real> &w0, const EOS_Data &eos, const Real dt,
                     DvceArray5D<Real> &u0);
  void AddCoordTerms(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc,
                     const EOS_Data &eos, const Real dt, DvceArray5D<Real> &u0);
  void SetExcisionMasks();

 private:
  MeshBlockPack* pmy_pack;
};

#endif // COORDINATES_COORDINATES_HPP_
