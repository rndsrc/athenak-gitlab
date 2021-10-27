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

  auto &aindcs = amesh_indcs;
  int zs = aindcs.zs; int ze = aindcs.ze;
  int ps = aindcs.ps; int pe = aindcs.pe;

  int &nmb = pmy_pack->nmb_thispack;

  auto mcoord_ = moments_coord;
  auto n0_ =n0;
  auto solid_angle_ =solid_angle;

  par_for("set_moments",DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      mcoord_(m,0,k,j,i) = 0.0;
      for (int z=zs; z<=ze; ++z) {
        for (int p=ps; p<=pe; ++p) {
          int zp = AngleInd(z,p,false,false,aindcs);
          mcoord_(m,0,k,j,i) += (SQR(n0_(m,z,p,k,j,i))
                                 *prim(m,zp,k,j,i)*solid_angle_.d_view(z,p));
        }
      }
    }
  );

  return;
}

} // namespace radiation
