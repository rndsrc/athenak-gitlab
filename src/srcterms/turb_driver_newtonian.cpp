//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_driver_hydro.cpp
//  \brief implementation of functions in TurbulenceDriverNewtonian

#include <limits>
#include <algorithm>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "ion-neutral/ion_neutral.hpp"
#include "driver/driver.hpp"
#include "utils/grid_locations.hpp"
#include "utils/random.hpp"
#include "turb_driver.hpp"
#include "turb_driver_newtonian.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

TurbulenceDriverNewtonian::TurbulenceDriverNewtonian(MeshBlockPack *pp, ParameterInput *pin) :
  TurbulenceDriver(pp,pin){
}



auto TurbulenceDriverNewtonian::ComputeNetEnergyInjection(DvceArray5D<Real> &u, DvceArray5D<Real> &ftmp)
{
  int &is = pmy_pack->mb_cells.is;
  int &js = pmy_pack->mb_cells.js;
  int &ks = pmy_pack->mb_cells.ks;
  int &nx1 = pmy_pack->mb_cells.nx1;
  int &nx2 = pmy_pack->mb_cells.nx2;
  int &nx3 = pmy_pack->mb_cells.nx3;


  auto &mbsize = pmy_pack->pmb->mbsize;

  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  auto force_new_ = ftmp;


  array_sum::GlobalSum sum_this_mb;


  Real m0 = static_cast<Real>(nmkji);
  Real m1 = 0.0, m2 = 0.0;
  Kokkos::parallel_reduce("net_en_3d", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sum_m0, Real &sum_m1)
    {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real v1 = force_new_(m,0,k,j,i);
      Real v2 = force_new_(m,1,k,j,i);
      Real v3 = force_new_(m,2,k,j,i);

      Real u1 = u(m,IM1,k,j,i);
      Real u2 = u(m,IM2,k,j,i);
      Real u3 = u(m,IM3,k,j,i);      

      sum_m0 += u(m,IDN,k,j,i)*(v1*v1+v2*v2+v3*v3);
      sum_m1 += u1*v1 + u2*v2 + u3*v3;
    }, Kokkos::Sum<Real>(m1), Kokkos::Sum<Real>(m2)
  );


  Real the_sum [3] { m0, m1,m2};

#if MPI_PARALLEL_ENABLED
  {
    MPI_Allreduce(MPI_IN_PLACE, the_sum,3, MPI_ATHENA_REAL, MPI_SUM,MPI_COMM_WORLD);
  }
#endif

  return std::make_tuple(the_sum[0], the_sum[1], the_sum[2]);

};

auto TurbulenceDriverNewtonian::ComputeNetMomentum(DvceArray5D<Real> &ftmp)
{
  int &is = pmy_pack->mb_cells.is;
  int &js = pmy_pack->mb_cells.js;
  int &ks = pmy_pack->mb_cells.ks;
  int &nx1 = pmy_pack->mb_cells.nx1;
  int &nx2 = pmy_pack->mb_cells.nx2;
  int &nx3 = pmy_pack->mb_cells.nx3;


  auto &mbsize = pmy_pack->pmb->mbsize;

  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  auto &force_tmp_ = ftmp;


  Real m0 = static_cast<Real>(nmkji);
  Real m1 = 0.0, m2 = 0.0, m3 = 0.0;

  Kokkos::parallel_reduce("net_mom_3d", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sum_m1, Real &sum_m2, Real &sum_m3)
    {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      sum_m1 += force_tmp_(m,0,k,j,i);
      sum_m2 += force_tmp_(m,1,k,j,i);
      sum_m3 += force_tmp_(m,2,k,j,i);

    }, Kokkos::Sum<Real>(m1), Kokkos::Sum<Real>(m2), Kokkos::Sum<Real>(m3)
  );

  Real the_sum [4] { m0, m1,m2,m3};

#if MPI_PARALLEL_ENABLED
  {
    MPI_Allreduce(MPI_IN_PLACE, the_sum,4, MPI_ATHENA_REAL, MPI_SUM,MPI_COMM_WORLD);
  }
#endif

  return std::make_tuple(the_sum[0], the_sum[1], the_sum[2], the_sum[3]);
};


void TurbulenceDriverNewtonian::GlobalNormalization(DvceArray5D<Real> &ftmp){

  int &is = pmy_pack->mb_cells.is, &ie = pmy_pack->mb_cells.ie;
  int &js = pmy_pack->mb_cells.js, &je = pmy_pack->mb_cells.je;
  int &ks = pmy_pack->mb_cells.ks, &ke = pmy_pack->mb_cells.ke;
  int &nx1 = pmy_pack->mb_cells.nx1;
  int &nx2 = pmy_pack->mb_cells.nx2;
  int &nx3 = pmy_pack->mb_cells.nx3;
  auto &ncells = pmy_pack->mb_cells;
  int ncells1 = ncells.nx1 + 2*(ncells.ng);
  int ncells2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
  int ncells3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;


  auto &mbsize = pmy_pack->pmb->mbsize;

  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  auto force_tmp_ = ftmp;

  Real m0,m1,m2,m3;
  std::tie(m0,m1,m2,m3) = ComputeNetMomentum(ftmp);


  par_for("net_mom_2", DevExeSpace(), 0, (pmy_pack->nmb_thispack-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      force_tmp_(m,0,k,j,i) -= m1/m0;
      force_tmp_(m,1,k,j,i) -= m2/m0;
      force_tmp_(m,2,k,j,i) -= m3/m0;
    }
  );

  DvceArray5D<Real> u;
  if (pmy_pack->phydro != nullptr) u = (pmy_pack->phydro->u0);
  if (pmy_pack->pmhd != nullptr) u = (pmy_pack->pmhd->u0);
  if (pmy_pack->pionn != nullptr) u = (pmy_pack->phydro->u0); // assume neutral density
                                                              //     >> ionized density

  Real e0,e1,e2;
  std::tie(e0,e1,e2) = ComputeNetEnergyInjection(u, ftmp);

  Real dvol = 1.0/(nx1*nx2*nx3); // old: Lx*Ly*Lz/nx1/nx2/nx3;
  m0 = e1*dvol*(pmy_pack->pmesh->dt);
  m1 = e2*dvol;

  Real s;
  if (m1 >= 0) {
    s = -m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
  } else {
    s = m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
  }

  // Now normalize new force array
  par_for("OU_process", DevExeSpace(),0,(pmy_pack->nmb_thispack-1),0,2,0,ncells3-1,0,ncells2-1,0,ncells1-1,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
    {
      force_tmp_(m,n,k,j,i) *= s;
    }
  );

}



//----------------------------------------------------------------------------------------
//! \fn  apply forcing (explicit version)

TaskStatus TurbulenceDriverNewtonian::AddForcing(Driver *pdrive, int stage)
{
  int &is = pmy_pack->mb_cells.is, &ie = pmy_pack->mb_cells.ie;
  int &js = pmy_pack->mb_cells.js, &je = pmy_pack->mb_cells.je;
  int &ks = pmy_pack->mb_cells.ks, &ke = pmy_pack->mb_cells.ke;
  int &nx1 = pmy_pack->mb_cells.nx1;
  int &nx2 = pmy_pack->mb_cells.nx2;
  int &nx3 = pmy_pack->mb_cells.nx3;
  auto &ncells = pmy_pack->mb_cells;
  int ncells1 = ncells.nx1 + 2*(ncells.ng);
  int ncells2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
  int ncells3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;

  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
  Real fcorr=0.0;
  Real gcorr=1.0;
  if ((pmy_pack->pmesh->time > 0.0) and (tcorr > 0.0)) {
    fcorr=exp(-((beta_dt)/tcorr));
    gcorr=sqrt(1.0-fcorr*fcorr);
  }

  if (pmy_pack->pionn == nullptr) {

    // modify conserved variables
    DvceArray5D<Real> u,w;
    if (pmy_pack->phydro != nullptr) {
      u = (pmy_pack->phydro->u0);
      w = (pmy_pack->phydro->w0);
    }
    if (pmy_pack->pmhd != nullptr) {
      u = (pmy_pack->pmhd->u0);
      w = (pmy_pack->pmhd->w0);
    }

    int &nmb = pmy_pack->nmb_thispack;
    auto force_ = force;
    auto force_new_ = force_new;
    par_for("push", DevExeSpace(),0,(pmy_pack->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i)
      {
        Real den = w(m,IDN,k,j,i);
        Real v1 = (fcorr*force_(m,0,k,j,i) + gcorr*force_new_(m,0,k,j,i))*beta_dt;
        Real v2 = (fcorr*force_(m,1,k,j,i) + gcorr*force_new_(m,1,k,j,i))*beta_dt;
        Real v3 = (fcorr*force_(m,2,k,j,i) + gcorr*force_new_(m,2,k,j,i))*beta_dt;
        Real m1 = den*w(m,IVX,k,j,i);
        Real m2 = den*w(m,IVY,k,j,i);
        Real m3 = den*w(m,IVZ,k,j,i);

  //      u(m,IEN,k,j,i) += m1*v1 + m2*v2 + m3*v3 + 0.5*den*(v1*v1+v2*v2+v3*v3);
        u(m,IEN,k,j,i) += m1*v1 + m2*v2 + m3*v3;
        u(m,IM1,k,j,i) += den*v1;
        u(m,IM2,k,j,i) += den*v2;
        u(m,IM3,k,j,i) += den*v3;
      }
    );
  } else {

    // modify conserved variables
    DvceArray5D<Real> u,w,u_,w_;
    u = (pmy_pack->pmhd->u0);
    w = (pmy_pack->pmhd->w0);
    u_ = (pmy_pack->phydro->u0);
    w_ = (pmy_pack->phydro->w0);

    int &nmb = pmy_pack->nmb_thispack;
    auto force_ = force;
    auto force_new_ = force_new;
    par_for("push", DevExeSpace(),0,(pmy_pack->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i)
      {
        // TODO:need to rescale forcing depending on ionization fraction

        Real v1 = (fcorr*force_(m,0,k,j,i) + gcorr*force_new_(m,0,k,j,i))*beta_dt;
        Real v2 = (fcorr*force_(m,1,k,j,i) + gcorr*force_new_(m,1,k,j,i))*beta_dt;
        Real v3 = (fcorr*force_(m,2,k,j,i) + gcorr*force_new_(m,2,k,j,i))*beta_dt;

        Real den = w(m,IDN,k,j,i);
        Real m1 = den*w(m,IVX,k,j,i);
        Real m2 = den*w(m,IVY,k,j,i);
        Real m3 = den*w(m,IVZ,k,j,i);

  //      u(m,IEN,k,j,i) += m1*v1 + m2*v2 + m3*v3 + 0.5*den*(v1*v1+v2*v2+v3*v3);
        u(m,IEN,k,j,i) += m1*v1 + m2*v2 + m3*v3;
        u(m,IM1,k,j,i) += den*v1;
        u(m,IM2,k,j,i) += den*v2;
        u(m,IM3,k,j,i) += den*v3;


        Real den_ = w_(m,IDN,k,j,i);
        Real m1_ = den_*w_(m,IVX,k,j,i);
        Real m2_ = den_*w_(m,IVY,k,j,i);
        Real m3_ = den_*w_(m,IVZ,k,j,i);

  //      u_(m,IEN,k,j,i) += m1_*v1 + m2_*v2 + m3_*v3 + 0.5*den*(v1*v1+v2*v2+v3*v3);
        u_(m,IEN,k,j,i) += m1_*v1 + m2_*v2 + m3_*v3;
        u_(m,IM1,k,j,i) += den_*v1;
        u_(m,IM2,k,j,i) += den_*v2;
        u_(m,IM3,k,j,i) += den_*v3;
      }
    );
  }

  return TaskStatus::complete;
}





