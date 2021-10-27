//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file angular_mesh_bcs.cpp
//  \brief

// Athena++ headers
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "radiation/radiation.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"


namespace radiation {

//----------------------------------------------------------------------------------------
//! \fn void Radiation::AngularMeshBoundaries()
//  \brief Populate angular ghost zones and zeros intensities inside excision radius

void Radiation::AngularMeshBoundaries()
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  auto &aindcs = pmy_pack->prad->amesh_indcs;
  int zs = aindcs.zs; int ze = aindcs.ze;
  int ps = aindcs.ps; int pe = aindcs.pe;

  int &nmb = pmy_pack->nmb_thispack;
  auto coord = pmy_pack->coord.coord_data;

  auto &i0_ = i0;

  par_for("rad_bvals",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real &x1min = coord.mb_size.d_view(m).x1min;
      Real &x1max = coord.mb_size.d_view(m).x1max;
      int nx1 = coord.mb_indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = coord.mb_size.d_view(m).x2min;
      Real &x2max = coord.mb_size.d_view(m).x2max;
      int nx2 = coord.mb_indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = coord.mb_size.d_view(m).x3min;
      Real &x3max = coord.mb_size.d_view(m).x3max;
      int nx3 = coord.mb_indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

      // Excusion radius
      if (rad < coord.bh_rmin) {
        for (int n=0; n<nangles; ++n) {
          i0_(m,n,k,j,i) = 0.0;
        }
      }

      // Populate angular ghost zones in azimuthal angle
      for (int z=zs; z<=ze; ++z) {
        for (int p=ps-aindcs.ng; p<=ps-1; ++p) {
          int p_src = pe - ps + 1 + p;
          int zp = AngleInd(z,p,false,false,aindcs);
          int zp_src = AngleInd(z,p_src,false,false,aindcs);
          i0_(m,zp,k,j,i) = i0_(m,zp_src,k,j,i);
        }
        for (int p=pe+1; p<=pe+aindcs.ng; ++p) {
          int p_src = ps - pe - 1 + p;
          int zp = AngleInd(z,p,false,false,aindcs);
          int zp_src = AngleInd(z,p_src,false,false,aindcs);
          i0_(m,zp,k,j,i) = i0_(m,zp_src,k,j,i);
        }
      }

      // Populate angular ghost zones in polar angle
      for (int z=zs-aindcs.ng; z<=zs-1; ++z) {
        for (int p=ps-aindcs.ng; p<=pe+aindcs.ng; ++p) {
          int z_src = 2*zs - 1 - z;
          int p_src = (p + aindcs.npsi/2) % (aindcs.npsi + 2*aindcs.ng);
          int zp = AngleInd(z,p,false,false,aindcs);
          int zp_src = AngleInd(z_src,p_src,false,false,aindcs);
          i0_(m,zp,k,j,i) = i0_(m,zp_src,k,j,i);
        }
      }
      for (int z=ze+1; z<=ze+aindcs.ng; ++z) {
        for (int p=ps-aindcs.ng; p<=pe+aindcs.ng; ++p) {
          int z_src = 2*ze + 1 - z;
          int p_src = (p + aindcs.npsi/2) % (aindcs.npsi + 2*aindcs.ng);
          int zp = AngleInd(z,p,false,false,aindcs);
          int zp_src = AngleInd(z_src,p_src,false,false,aindcs);
          i0_(m,zp,k,j,i) = i0_(m,zp_src,k,j,i);
        }
      }
    }
  );
}

} // namespace radiation
