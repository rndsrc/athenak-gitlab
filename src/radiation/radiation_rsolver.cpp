//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_rsolver.cpp
//  \brief

#include "radiation.hpp"

namespace radiation {

//----------------------------------------------------------------------------------------
//! \fn void SpatialFlux
//  \brief Inlined flux calculation for radiation

KOKKOS_INLINE_FUNCTION
void SpatialFlux(TeamMember_t const &member, const EOS_Data eos, const CoordData &coord,
     const int m, const int k, const int j,  const int il, const int iu, const int ivx,
     const DvceArray7D<Real> nn, const int nangles, struct AMeshIndcs aindcs,
     const ScrArray2D<Real> &iil, const ScrArray2D<Real> &iir, DvceArray5D<Real> flx)
{
  par_for_inner(member, il, iu, [&](const int i)
  {
    for (int zp=0; zp<nangles; ++zp) {
      int z; int p;
      InverseAngleInd(zp, z, p, aindcs);
      flx(m,zp,k,j,i) = (nn(m,0,z,p,k,j,i)
                         * (nn(m,0,z,p,k,j,i) < 0.0 ? iil(zp,i) : iir(zp,i)));
    }
  });

  return;
}

} // namespace radiation
