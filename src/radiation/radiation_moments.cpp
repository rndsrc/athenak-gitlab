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
#include "eos/eos.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

RadiationMoments::RadiationMoments(MeshBlockPack *pp, ParameterInput *pin)
  : EquationOfState(pp, pin)
{
}

//----------------------------------------------------------------------------------------
// \!fn void ConsToPrim()
// \brief Converts conserved into primitive variables. Operates over entire MeshBlock,
//  including ghost cells.

void RadiationMoments::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;

  auto &aindcs = pmy_pack->prad->amesh_indcs;
  int zs = aindcs.zs; int ze = aindcs.ze;
  int ps = aindcs.ps; int pe = aindcs.pe;

  int &nmb = pmy_pack->nmb_thispack;
  auto n0_n_mu_ = pmy_pack->prad->n0_n_mu;

  par_for("rad_con2prim",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      for (int z=zs; z<=ze; ++z) {
        for (int p=ps; p<=pe; ++p) {
          int zp = AngleInd(z,p,false,false,aindcs);
            prim(m,zp,k,j,i) = cons(m,zp,k,j,i) / n0_n_mu_(m,0,z,p,k,j,i);
            if (prim(m,zp,k,j,i) < 0.0) {
              prim(m,zp,k,j,i) = 0.0;
              cons(m,zp,k,j,i) = 0.0;
          }
        }
      }

      // Populate angular ghost zones in azimuthal angle
      for (int z=zs; z<=ze; ++z) {
        for (int p=ps-ng; p<=ps-1; ++p) {
          int p_src = pe - ps + 1 + p;
          int zp = AngleInd(z,p,false,false,aindcs);
          int zp_src = AngleInd(z,p_src,false,false,aindcs);
          prim(m,zp,k,j,i) = prim(m,zp_src,k,j,i);
        }
        for (int p=pe+1; p<=pe+ng; ++p) {
          int p_src = ps - pe - 1 + p;
          int zp = AngleInd(z,p,false,false,aindcs);
          int zp_src = AngleInd(z,p_src,false,false,aindcs);
          prim(m,zp,k,j,i) = prim(m,zp_src,k,j,i);
        }
      }

      // Populate angular ghost zones in polar angle
      for (int z=zs-ng; z<=zs-1; ++z) {
        for (int p=ps-ng; p<=pe+ng; ++p) {
          int z_src = 2*zs - 1 - z;
          int p_src = (p + aindcs.npsi/2) % (aindcs.npsi + 2*ng);
          int zp = AngleInd(z,p,false,false,aindcs);
          int zp_src = AngleInd(z_src,p_src,false,false,aindcs);
          prim(m,zp,k,j,i) = prim(m,zp_src,k,j,i);
        }
      }
      for (int z=ze+1; z<=ze+ng; ++z) {
        for (int p=ps-ng; p<=pe+ng; ++p) {
          int z_src = 2*ze + 1 - z;
          int p_src = (p + aindcs.npsi/2) % (aindcs.npsi + 2*ng);
          int zp = AngleInd(z,p,false,false,aindcs);
          int zp_src = AngleInd(z_src,p_src,false,false,aindcs);
          prim(m,zp,k,j,i) = prim(m,zp_src,k,j,i);
        }
      }
    }
  );

  // Zero and set moments
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  auto mcoord_ = pmy_pack->prad->moments_coord;
  auto nmu_ = pmy_pack->prad->nmu;
  auto solid_angle_ = pmy_pack->prad->solid_angle;

  par_for("zero_moments",DevExeSpace(),0,(nmb-1),0,9,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
    {
      mcoord_(m,n,k,j,i) = 0.0;
    }
  );

  par_for("set_moments",DevExeSpace(),0,(nmb-1),zs,ze,ps,pe,
    KOKKOS_LAMBDA(int m, int z, int p)
    {
      int zp = AngleInd(z,p,false,false,aindcs);
      for (int n1 = 0, n12 = 0; n1 < 4; ++n1) {
        for (int n2 = n1; n2 < 4; ++n2, ++n12) {
          for (int k = ks; k <= ke; ++k) {
            for (int j = js; j <= je; ++j) {
              for (int i = is; i <= ie; ++i) {
                mcoord_(m,n12,k,j,i) += (nmu_(m,n1,z,p,k,j,i)*nmu_(m,n2,z,p,k,j,i)
                                         *prim(m,zp,k,j,i)*solid_angle_.d_view(z,p));
              }
            }
          }
        }
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void PrimToCons()
// \brief Converts conserved into primitive variables. Operates over only active cells.

void RadiationMoments::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  auto &aindcs = pmy_pack->prad->amesh_indcs;
  int &zs = aindcs.zs; int &ze = aindcs.ze;
  int &ps = aindcs.ps; int &pe = aindcs.pe;

  int &nmb = pmy_pack->nmb_thispack;
  auto n0_n_mu_ = pmy_pack->prad->n0_n_mu;

  par_for("rad_prim2con", DevExeSpace(),0,(nmb-1),zs,ze,ps,pe,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int z, int p, int k, int j, int i)
    {
      int zp = AngleInd(z,p,false,false,aindcs);
      cons(m,zp,k,j,i) = n0_n_mu_(m,0,z,p,k,j,i)*prim(m,zp,k,j,i);
    }
  );

  return;
}
