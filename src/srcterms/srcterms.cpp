//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.cpp
//  \brief implementation of source terms in equations of motion

#include <iostream>

#include "athena.hpp"
#include "turb_driver.hpp"
#include "srcterms.hpp"

//----------------------------------------------------------------------------------------
// constructor, parses input file and initializes data structures and parameters

SourceTerms::SourceTerms(MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp)
{
  if (pp->pturb_driver != nullptr) {
    operatorsplit_terms = true;
    implicit_terms = true;
  }
}

//----------------------------------------------------------------------------------------
// destructor
  
SourceTerms::~SourceTerms()
{
}

//----------------------------------------------------------------------------------------
//! \fn ApplySrcTermsStageRunTL()
// apply unsplit source terms added in EACH stage of the stage run task list

void SourceTerms::ApplySrcTermsStageRunTL(DvceArray5D<Real> &u, DvceArray5D<Real> &w, int stage)
{
  if (pmy_pack->pturb_driver != nullptr) {
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ApplyImplicitSrcTermsStageRunTL()
// apply unsplit source terms added in EACH stage of the stage run task list

void SourceTerms::ApplyImplicitSrcTermsStageRunTL(DvceArray5D<Real> &u, DvceArray5D<Real> &w, int stage)
{
  if (pmy_pack->pturb_driver != nullptr) {
    static_cast<ImEx*>(pmy_pack->pturb_driver)->ApplySourceTermsImplicit(u,w,stage);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ApplySrcTermsStageRunTL()
// apply operator split source terms added in the operator split task list

void SourceTerms::ApplySrcTermsOperatorSplitTL(DvceArray5D<Real> &u, DvceArray5D<Real> &w)
{
  if (pmy_pack->pturb_driver != nullptr) {
    pmy_pack->pturb_driver->ApplyForcing(u);
    static_cast<ImEx*>(pmy_pack->pturb_driver)->ApplySourceTermsImplicitPreStage(u,w);
  }
  return;
}
