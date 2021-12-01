#ifndef SRCTERMS_SRCTERMS_HPP_
#define SRCTERMS_SRCTERMS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.hpp
//! \brief Data, functions, and classes to implement various source terms in the hydro 
//! and/or MHD equations of motion.  Currently implemented:
//!  (1) constant (gravitational) acceleration - for RTI
//!  (2) shearing box in 2D (x-z), for both hydro and MHD
//!  (3) random forcing to drive turbulence - implemented in TurbulenceDriver class

#include <map>
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"

// forward declarations
class TurbulenceDriver;
class Driver;

//----------------------------------------------------------------------------------------
//! \class SourceTerms
//! \brief data and functions for physical source terms

class SourceTerms
{
 public:
  SourceTerms(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  ~SourceTerms();

  // data
  // flags for various source terms
  bool source_terms_enabled;   // true if any srcterm included
  bool const_accel;
  bool shearing_box;
  bool beam_source;
  bool rad_source;
  bool ism_cooling;

  // magnitude and direction of constant accel
  Real const_accel_val;
  int  const_accel_dir;

  // Orbital frequency and shear rate for shearing box
  Real omega0, qshear;

  // Beam source
  Real pos_1, pos_2, pos_3;
  Real width;
  Real dir_1, dir_2, dir_3;
  Real spread;
  Real dii_dt;

  // Radiation source
  Real arad;
  Real sigma_a;
  Real sigma_p;
  Real sigma_s;
  bool coupling;

  // physical constants and heating rate used with ISM cooling
  Real mbar, kboltz, hrate;

  // functions
  void AddConstantAccel(DvceArray5D<Real> &u0,const DvceArray5D<Real> &w0,const Real dt);
  void AddBeamSource(DvceArray5D<Real> &i0,const Real dt);
  void AddRadiationSourceTerm(DvceArray5D<Real> &i0, DvceArray5D<Real> &i1,
                              const Real dt);
  void AddShearingBox(DvceArray5D<Real> &u0,const DvceArray5D<Real> &w0,const Real dt);
  void AddShearingBox(DvceArray5D<Real> &u0, const DvceArray5D<Real> &w0,
                      const DvceArray5D<Real> &bcc, const Real dt);
  void AddSBoxEField(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);

 private:
  MeshBlockPack* pmy_pack;
};

#endif // SRCTERMS_SRCTERMS_HPP_
