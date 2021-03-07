//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_driver_hydro.cpp
//  \brief implementation of functions in TurbulenceDriverHydro

#include <limits>
#include <algorithm>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "utils/grid_locations.hpp"
#include "utils/random.hpp"
#include "turb_driver_hydro.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

TurbulenceDriverHydro::TurbulenceDriverHydro(MeshBlockPack *pp, ParameterInput *pin) :
  TurbulenceDriver(pp,pin){
}


void TurbulenceDriverHydro::ApplyForcing(DvceArray5D<Real> &u)
{

  if(ImEx::this_imex != ImEx::method::RKexplicit) return;

  //Update random force
  NewRandomForce(force_tmp);

  ApplyForcingSourceTermsExplicit(u);

  return;
}

void TurbulenceDriverHydro::ImplicitEquation(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI, DvceArray5D<Real> &Ru)
{
  switch (this_imex){

//    case method::RK1:
//      ImplicitEquationRK1(u,w,dtI,Ru);
//      break;

    case method::RK2:
      ImplicitEquationRK2(u,w,dtI,Ru);
      break;
    case method::RK3:
      ImplicitEquationRK3(u,w,dtI,Ru);
      break;
  }

}

array_sum::GlobalSum TurbulenceDriverHydro::ComputeNetEnergyInjection(DvceArray5D<Real> &w, DvceArray5D<Real> &ftmp)
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

  auto force_tmp_ = ftmp;


  array_sum::GlobalSum sum_this_mb;


  Kokkos::parallel_reduce("net_en_3d", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum)
    {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      auto dsum = mbsize.dx1.d_view(m) * mbsize.dx2.d_view(m) * mbsize.dx3.d_view(m);

      array_sum::GlobalSum fsum;
      fsum.the_array[IDN] = (
			    +w(m, IVX, k,j,i)*force_tmp_(m,0,k,j,i)
			    +w(m, IVY, k,j,i)*force_tmp_(m,1,k,j,i)
			    +w(m, IVZ, k,j,i)*force_tmp_(m,2,k,j,i)
			    )*w(m,IDN,k,j,i)*dsum;

      fsum.the_array[IM1] = (
			    +force_tmp_(m, IVX, k,j,i)*force_tmp_(m,0,k,j,i)
			    +force_tmp_(m, IVY, k,j,i)*force_tmp_(m,1,k,j,i)
			    +force_tmp_(m, IVZ, k,j,i)*force_tmp_(m,2,k,j,i)
			    )*w(m,IDN,k,j,i)*dsum;
      mb_sum += fsum;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb)
  );

#if MPI_PARALLEL_ENABLED
  {
    // Does this work on GPU?
    MPI_Allreduce(MPI_IN_PLACE, sum_this_mb.the_array,IM1+1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
  }
#endif

  return sum_this_mb;
};

array_sum::GlobalSum TurbulenceDriverHydro::ComputeNetMomentum(DvceArray5D<Real> &u, DvceArray5D<Real> &ftmp)
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

  auto force_tmp_ = ftmp;


  array_sum::GlobalSum sum_this_mb;


  Kokkos::parallel_reduce("net_mom_3d", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum)
    {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      auto dsum = mbsize.dx1.d_view(m) * mbsize.dx2.d_view(m) * mbsize.dx3.d_view(m);

      array_sum::GlobalSum fsum;
      fsum.the_array[IDN] = 1.0 * dsum;
      fsum.the_array[IM1] = u(m,IDN,k,j,i)*force_tmp_(m,0,k,j,i)*dsum;
      fsum.the_array[IM2] = u(m,IDN,k,j,i)*force_tmp_(m,1,k,j,i)*dsum;
      fsum.the_array[IM3] = u(m,IDN,k,j,i)*force_tmp_(m,2,k,j,i)*dsum;
      fsum.the_array[IEN] = u(m,IDN,k,j,i)*dsum;

      mb_sum += fsum;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb)
  );

#if MPI_PARALLEL_ENABLED
  {
    // Does this work on GPU?
    MPI_Allreduce(MPI_IN_PLACE, sum_this_mb.the_array,IEN+1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
  }
#endif

  return sum_this_mb;
};

array_sum::GlobalSum TurbulenceDriverHydro::ComputeNetForce(DvceArray5D<Real> &u, DvceArray5D<Real> &ftmp)
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


  array_sum::GlobalSum sum_this_mb;


  Kokkos::parallel_reduce("net_mom_3d", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum)
    {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      auto dsum = mbsize.dx1.d_view(m) * mbsize.dx2.d_view(m) * mbsize.dx3.d_view(m);

      array_sum::GlobalSum fsum;
      fsum.the_array[IDN] = 1.0 * dsum;
      fsum.the_array[IM1] = force_tmp_(m,0,k,j,i)*dsum;
      fsum.the_array[IM2] = force_tmp_(m,1,k,j,i)*dsum;
      fsum.the_array[IM3] = force_tmp_(m,2,k,j,i)*dsum;

      mb_sum += fsum;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb)
  );

#if MPI_PARALLEL_ENABLED
  {
    // Does this work on GPU?
    MPI_Allreduce(MPI_IN_PLACE, sum_this_mb.the_array,IM3+1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
  }
#endif

  return sum_this_mb;
};

//----------------------------------------------------------------------------------------
//! \fn  apply forcing (explicit version)

void TurbulenceDriverHydro::ApplyForcingSourceTermsExplicit(DvceArray5D<Real> &u)
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

  Real lx = pmy_pack->pmesh->mesh_size.x1max - pmy_pack->pmesh->mesh_size.x1min;
  Real ly = pmy_pack->pmesh->mesh_size.x2max - pmy_pack->pmesh->mesh_size.x2min;
  Real lz = pmy_pack->pmesh->mesh_size.x3max - pmy_pack->pmesh->mesh_size.x3min;
  Real dkx = 2.0*M_PI/lx;
  Real dky = 2.0*M_PI/ly;
  Real dkz = 2.0*M_PI/lz;

  int &nmb = pmy_pack->nmb_thispack;

  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  auto force_tmp_ = force_tmp;
  // Compute net force
  auto sum_this_mb = ComputeNetForce(u,force_tmp);

  Real m0 = sum_this_mb.the_array[IDN];
  Real m1 = sum_this_mb.the_array[IM1];
  Real m2 = sum_this_mb.the_array[IM2];
  Real m3 = sum_this_mb.the_array[IM3];

  m0 = std::max(m0, static_cast<Real>(std::numeric_limits<float>::min()) );

  par_for("net_mom_2", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      force_tmp_(m,0,k,j,i) -= m1/m0;
      force_tmp_(m,1,k,j,i) -= m2/m0;
      force_tmp_(m,2,k,j,i) -= m3/m0;
    }
  );

  array_sum::GlobalSum sum_this_mb_en;
  Kokkos::parallel_reduce("forcing_norm", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum)
    { 
       // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real v1 = force_tmp_(m,0,k,j,i);
      Real v2 = force_tmp_(m,1,k,j,i);
      Real v3 = force_tmp_(m,2,k,j,i);

      /* two options here
      Real u1 = u(m,IM1,k,j,i)/u(m,IDN,k,j,i);
      Real u2 = u(m,IM2,k,j,i)/u(m,IDN,k,j,i);
      Real u3 = u(m,IM3,k,j,i)/u(m,IDN,k,j,i);      


      force_sum::GlobalSum fsum;
      fsum.the_array[IDN] = (v1*v1+v2*v2+v3*v3);
      fsum.the_array[IM1] = u1*v1 + u2*v2 + u3*v3;
      */

      Real u1 = u(m,IM1,k,j,i);
      Real u2 = u(m,IM2,k,j,i);
      Real u3 = u(m,IM3,k,j,i);      


      array_sum::GlobalSum fsum;
      fsum.the_array[IDN] = u(m,IDN,k,j,i)*(v1*v1+v2*v2+v3*v3);
      fsum.the_array[IM1] = u1*v1 + u2*v2 + u3*v3;
        
      mb_sum += fsum;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb_en)
  );

  m0 = sum_this_mb_en.the_array[IDN];
  m1 = sum_this_mb_en.the_array[IM1];

  m0 = std::max(m0, static_cast<Real>(std::numeric_limits<float>::min()) );

/* old normalization
  aa = 0.5*m0;
  aa = max(aa,static_cast<Real>(1.0e-20));
  if (tcorr<=1e-20) {
    s = sqrt(dedt/dt/dvol/aa);
  } else {
    s = sqrt(dedt/tcorr/dvol/aa);
  }
*/

  // new normalization: assume constant energy injection per unit mass
  // explicit solution of <sF . (v + sF dt)> = dedt
  
  Real dvol = 1. /(nx1*nx2*nx3); // old: Lx*Ly*Lz/nx1/nx2/nx3;
  m0 = m0*dvol*(pmy_pack->pmesh->dt);
  m1 = m1*dvol;

  Real s;
  if (m1 >= 0) {
    s = -m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
  } else {
    s = m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
  }

  Real fcorr=0.0;
  Real gcorr=1.0;
  if ((pmy_pack->pmesh->time > 0.0) and (tcorr > 0.0)) {
    fcorr=exp(-((pmy_pack->pmesh->dt)/tcorr));
    gcorr=sqrt(1.0-fcorr*fcorr);
  }

  auto force_ = force;
  par_for("OU_process", DevExeSpace(), 0, nmb-1, 0, 2, 0, ncells3-1, 0, ncells2-1,
    0, ncells1-1, KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
    {
      force_(m,n,k,j,i) = fcorr*force_(m,n,k,j,i) + gcorr*s*force_tmp_(m,n,k,j,i);
    }
  );

  // modify conserved variables
  Real &dt = pmy_pack->pmesh->dt;
  par_for("push", DevExeSpace(),0,(pmy_pack->nmb_thispack-1),
    ks,ke,js,je,is,ie,KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real den = u(m,IDN,k,j,i);
      Real v1 = force_(m,0,k,j,i)*dt;
      Real v2 = force_(m,1,k,j,i)*dt;
      Real v3 = force_(m,2,k,j,i)*dt;
      Real m1 = u(m,IM1,k,j,i);
      Real m2 = u(m,IM2,k,j,i);
      Real m3 = u(m,IM3,k,j,i);

      u(m,IEN,k,j,i) += m1*v1 + m2*v2 + m3*v3 + 0.5*den*(v1*v1+v2*v2+v3*v3);
      u(m,IM1,k,j,i) += den*v1;
      u(m,IM2,k,j,i) += den*v2;
      u(m,IM3,k,j,i) += den*v3;
    }
  );

  return;
}

void TurbulenceDriverHydro::ImplicitEquationRK3(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI, DvceArray5D<Real> &Ru)
{
  auto &ncells = pmy_pack->mb_cells;
  int ncells1 = ncells.nx1 + 2*(ncells.ng);
  int ncells2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
  int ncells3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;


  int &nmb = pmy_pack->nmb_thispack;


  auto force_tmp_ = force_tmp;
  auto force_ = force;

  // This is complicated because it depends on the RK method...

  // RK3 evaluates the stiff sources at four different times
  // alpha ~ 0.25, 0., 1., 0.5 (in units of delta_t)
  
  //The first two can be constructed on after another.
  //but then need to construct (and store!) 0.5 first.
  
  // TODO only RK3 for now
  //
  Real fcorr=0.0;
  Real gcorr=1.0;
  
  switch(ImEx::current_stage){

    case 0:
      //Advance force to 0.25

      NewRandomForce(force_tmp);

      // Correlation coefficients for Ornstein-Uhlenbeck
      if ((tcorr > 0.0)) {
	  fcorr=exp(-(ImEx::ceff[ImEx::current_stage]*(pmy_pack->pmesh->dt)/tcorr));
	  gcorr=sqrt(1.0-fcorr*fcorr);
      }

      par_for("OU_process", DevExeSpace(), 0, nmb-1, 0, 2, 0, ncells3-1, 0, ncells2-1,
	0, ncells1-1, KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	{
	  force_tmp_(m,n,k,j,i) = fcorr*force_(m,n,k,j,i) + gcorr*force_tmp_(m,n,k,j,i);
	}
      );

     ApplyForcingImplicit(force_tmp, u,w,dtI); 
      
    break;

    case 1:
      // Use previous force -- Nothing to do here
      ApplyForcingImplicit(force, u,w,dtI); 
    break;

    case 2:
      //Advance force to 0.5
      
      NewRandomForce(force);

      // Correlation coefficients for Ornstein-Uhlenbeck
      fcorr=0.0;
      gcorr=1.0;
      if ((tcorr > 0.0)) {
	  fcorr=exp(-(ImEx::ceff[3] - ImEx::ceff[0])*(pmy_pack->pmesh->dt)/tcorr);
	  gcorr=sqrt(1.0-fcorr*fcorr);
      }

      par_for("OU_process", DevExeSpace(), 0, nmb-1, 0, 2, 0, ncells3-1, 0, ncells2-1,
	0, ncells1-1, KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	{
	  force_tmp_(m,n,k,j,i) = fcorr*force_tmp_(m,n,k,j,i) + gcorr*force_(m,n,k,j,i);
	}
      );

      //Advance force to 1.
      
      NewRandomForce(force);

      fcorr=0.0;
      gcorr=1.0;
      if ((tcorr > 0.0)) {
	  fcorr=exp(-(ImEx::ceff[2] - ImEx::ceff[3])*(pmy_pack->pmesh->dt)/tcorr);
	  gcorr=sqrt(1.0-fcorr*fcorr);
      }

      par_for("OU_process", DevExeSpace(), 0, nmb-1, 0, 2, 0, ncells3-1, 0, ncells2-1,
	0, ncells1-1, KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	{
	  force_(m,n,k,j,i) = fcorr*force_tmp_(m,n,k,j,i) + gcorr*force_(m,n,k,j,i);
	}
      );

      ApplyForcingImplicit(force_tmp, u,w,dtI); 
      
    break;

    case 3:
      // Use previous force -- Nothing to do here
      //
      ApplyForcingImplicit(force, u,w,dtI); 
    break;

    case 4:
      // Here dt is zero, so nothing really happens
      pmy_pack->phydro->peos->ConsToPrim(u,w);
      return; // No implicit source term

    break;

  };

  ComputeImplicitSources(u,w,dtI,Ru);
}

void TurbulenceDriverHydro::ImplicitEquationRK2(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI, DvceArray5D<Real> &Ru)
{
  auto &ncells = pmy_pack->mb_cells;
  int ncells1 = ncells.nx1 + 2*(ncells.ng);
  int ncells2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
  int ncells3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;


  int &nmb = pmy_pack->nmb_thispack;


  auto force_tmp_ = force_tmp;
  auto force_ = force;

  // This is complicated because it depends on the RK method...

  // RK3 evaluates the stiff sources at four different times
  // alpha ~ 0.5, 0., 1. (in units of delta_t)
  
  //The first two can be constructed on after another.
  //but then need to construct (and store!) 0.5 first.
  
  // TODO only RK3 for now
  //
  Real fcorr=0.0;
  Real gcorr=1.0;
  
  switch(ImEx::current_stage){

    case 0:
      //Advance force to 0.5

      NewRandomForce(force_tmp);

      // Correlation coefficients for Ornstein-Uhlenbeck
      if ((tcorr > 0.0)) {
	  fcorr=exp(-(ImEx::ceff[ImEx::current_stage]*(pmy_pack->pmesh->dt)/tcorr));
	  gcorr=sqrt(1.0-fcorr*fcorr);
      }

      par_for("OU_process", DevExeSpace(), 0, nmb-1, 0, 2, 0, ncells3-1, 0, ncells2-1,
	0, ncells1-1, KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	{
	  force_tmp_(m,n,k,j,i) = fcorr*force_(m,n,k,j,i) + gcorr*force_tmp_(m,n,k,j,i);
	}
      );

     ApplyForcingImplicit(force_tmp, u,w,dtI); 
      
    break;

    case 1:
      // Use previous force -- Nothing to do here
      ApplyForcingImplicit(force, u,w,dtI); 
    break;

    case 2:
      //Advance force to 0.5
      
      NewRandomForce(force);

      // Correlation coefficients for Ornstein-Uhlenbeck
      fcorr=0.0;
      gcorr=1.0;
      if ((tcorr > 0.0)) {
	  fcorr=exp(-(ImEx::ceff[2] - ImEx::ceff[0])*(pmy_pack->pmesh->dt)/tcorr);
	  gcorr=sqrt(1.0-fcorr*fcorr);
      }

      par_for("OU_process", DevExeSpace(), 0, nmb-1, 0, 2, 0, ncells3-1, 0, ncells2-1,
	0, ncells1-1, KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	{
	  force_(m,n,k,j,i) = fcorr*force_tmp_(m,n,k,j,i) + gcorr*force_(m,n,k,j,i);
	}
      );

      ApplyForcingImplicit(force, u,w,dtI); 
      
    break;

    case 3:
      // Here dt is zero, so nothing really happens
      pmy_pack->phydro->peos->ConsToPrim(u,w);
      return; // No implicit source term

    break;

  };

  ComputeImplicitSources(u,w,dtI,Ru);
}

void TurbulenceDriverHydro::ComputeImplicitSources(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI, DvceArray5D<Real> &Ru){

  auto &ncells = pmy_pack->mb_cells;
  int n1 = ncells.nx1 + 2*(ncells.ng);
  int n2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
  int n3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;

  auto& eos_data = pmy_pack->phydro->peos->eos_data;

  auto gm1 = eos_data.gamma -1.;


  int &nmb = pmy_pack->nmb_thispack;

  auto Rstiff = Ru;
  auto cons = u;
  auto prim = w;

  auto noff_ = ImEx::noff;

   par_for("cons_implicit", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {

      Rstiff(m,IVX-noff_,k,j,i) = -cons(m, IVX, k,j,i);
      Rstiff(m,IVY-noff_,k,j,i) = -cons(m, IVY, k,j,i);
      Rstiff(m,IVZ-noff_,k,j,i) = -cons(m, IVZ, k,j,i);
      Rstiff(m,IEN-noff_,k,j,i) = -cons(m, IEN, k,j,i);


      cons(m,IVX,k,j,i) = prim(m,IVX,k,j,i)*prim(m,IDN,k,j,i);
      cons(m,IVY,k,j,i) = prim(m,IVY,k,j,i)*prim(m,IDN,k,j,i);
      cons(m,IVZ,k,j,i) = prim(m,IVZ,k,j,i)*prim(m,IDN,k,j,i);

      auto v2 = prim(m,IVX,k,j,i)*prim(m,IVX,k,j,i) + 
	prim(m,IVY,k,j,i)*prim(m,IVY,k,j,i) +prim(m,IVZ,k,j,i)*prim(m,IVZ,k,j,i);

      cons(m,IEN,k,j,i) = prim(m,IDN,k,j,i)*0.5*v2 + prim(m,IPR,k,j,i)/gm1;


      Rstiff(m,IVX-noff_,k,j,i) = ( Rstiff(m,IVX-noff_,k,j,i)+cons(m, IVX, k,j,i))/dtI;
      Rstiff(m,IVY-noff_,k,j,i) = ( Rstiff(m,IVY-noff_,k,j,i)+cons(m, IVY, k,j,i))/dtI;
      Rstiff(m,IVZ-noff_,k,j,i) = ( Rstiff(m,IVZ-noff_,k,j,i)+cons(m, IVZ, k,j,i))/dtI;
      Rstiff(m,IEN-noff_,k,j,i) = ( Rstiff(m,IEN-noff_,k,j,i)+cons(m, IEN, k,j,i))/dtI;



    });


}

void TurbulenceDriverHydro::ApplyForcingImplicit( DvceArray5D<Real> &force_, DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI)
{
  auto &ncells = pmy_pack->mb_cells;
  int ncells1 = ncells.nx1 + 2*(ncells.ng);
  int ncells2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
  int ncells3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;

  int &nmb = pmy_pack->nmb_thispack;

  auto& eos_data = pmy_pack->phydro->peos->eos_data;
  auto gm1 = eos_data.gamma -1.;
  
  // Fill explicit prims by inverting u
  pmy_pack->phydro->peos->ConsToPrim(u,w);


  auto sum_this_mb = ComputeNetMomentum(u, force_);


  Real m0 = sum_this_mb.the_array[IDN];
  Real m1 = sum_this_mb.the_array[IM1];
  Real m2 = sum_this_mb.the_array[IM2];
  Real m3 = sum_this_mb.the_array[IM3];
  Real rhoV = sum_this_mb.the_array[IEN];

  m0 = std::max(m0, static_cast<Real>(std::numeric_limits<Real>::min()) );

  auto frce = force_;

  par_for("net_mom_2", DevExeSpace(), 0, nmb-1, 0, (ncells3-1), 0, (ncells2-1), 0, (ncells1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      frce(m,0,k,j,i) -= m1/m0/ u(m,IDN,k,j,i);
      frce(m,1,k,j,i) -= m2/m0/ u(m,IDN,k,j,i);
      frce(m,2,k,j,i) -= m3/m0/ u(m,IDN,k,j,i);
    }
  );

  auto sum_this_mb_en = ComputeNetEnergyInjection(w,force_);

  auto& Fv = sum_this_mb_en.the_array[IDN];
  auto& F2 = sum_this_mb_en.the_array[IM1];


  auto const tmp = -fabs(Fv)/(2.*dtI*F2);

  //force normalization
  auto s = tmp + sqrt(tmp*tmp+ dedt/(dtI*F2));
  if ( F2 == 0.) s=0.;


  par_for("update_prims_implicit", DevExeSpace(), 0, nmb-1, 0, (ncells3-1), 0, (ncells2-1), 0, (ncells1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      w(m,IVX,k,j,i) += dtI * u(m,IDN,k,j,i)*frce(m,0,k,j,i) *s;
      w(m,IVY,k,j,i) += dtI * u(m,IDN,k,j,i)*frce(m,1,k,j,i) *s;
      w(m,IVZ,k,j,i) += dtI * u(m,IDN,k,j,i)*frce(m,2,k,j,i) *s;

      w(m,IPR,k,j,i) += 0.5*dtI*dtI * s * s * (
	  +frce(m,0,k,j,i)*frce(m,0,k,j,i) 
	  +frce(m,1,k,j,i)*frce(m,1,k,j,i) 
	  +frce(m,2,k,j,i)*frce(m,2,k,j,i) )* u(m,IDN,k,j,i)*gm1;
    }
  );


  // Remove momentum normalization, since it will be reused in other substeps
  par_for("fix_force", DevExeSpace(), 0, nmb-1, 0, (ncells3-1), 0, (ncells2-1), 0, (ncells1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      frce(m,0,k,j,i) += m1/m0/ u(m,IDN,k,j,i);
      frce(m,1,k,j,i) += m2/m0/ u(m,IDN,k,j,i);
      frce(m,2,k,j,i) += m3/m0/ u(m,IDN,k,j,i);
    }
  );

  return;
}




