//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_moments.cpp
//  \brief derived class that implements radiation moments and conversions

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "radiation/radiation.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

namespace radiation {

//----------------------------------------------------------------------------------------
// \!fn void SetMoments()
// \brief Sets radiation moments.

void Radiation::SetMoments(DvceArray5D<Real> &prim)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int nangles_ = nangles;
  int &nmb = pmy_pack->nmb_thispack;

  auto mcoord_ = moments_coord;
  auto nmu_ = nmu;
  auto solid_angle_ = solid_angle;

  par_for("set_moments",DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      for (int n=0; n<10; ++n) {
        mcoord_(m,n,k,j,i) = 0.0;
      }
      for (int n1=0, n12=0; n1<4; ++n1) {
        for (int n2=n1; n2<4; ++n2, ++n12) {
          for (int lm=0; lm<nangles_; ++lm) {
            mcoord_(m,n12,k,j,i) += (nmu_(m,lm,k,j,i,n1)*nmu_(m,lm,k,j,i,n2)
                                     *prim(m,lm,k,j,i)*solid_angle_.d_view(lm));
          }
        }
      }
    }
  );
  
  return;
}

} // namespace radiation
