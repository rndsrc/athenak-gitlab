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
#include "turb_driver_hydro_rel.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

TurbulenceDriverHydroRel::TurbulenceDriverHydroRel(MeshBlockPack *pp, ParameterInput *pin) :
  TurbulenceDriver(pp,pin){
    pmy_pack->phydro->needs_c2p = false;
}


void TurbulenceDriverHydro::ApplyForcing(DvceArray5D<Real> &u)
{

  // Not supported
  return;
}

void TurbulenceDriverHydroRel::ImplicitEquation(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI, DvceArray5D<Real> &Ru)
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



void TurbulenceDriverHydroRel::ImplicitEquationRK3(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI, DvceArray5D<Real> &Ru)
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
	  fcorr=exp(-(0.5 - 0.24169426078821)*(pmy_pack->pmesh->dt)/tcorr);
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
	  fcorr=exp(-(0.5)*(pmy_pack->pmesh->dt)/tcorr);
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

void TurbulenceDriverHydroRel::ImplicitEquationRK2(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI, DvceArray5D<Real> &Ru)
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

void TurbulenceDriverHydroRel::ComputeImplicitSources(DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI, DvceArray5D<Real> &Ru){

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


      Real gamma_adi = gm1+1.;
      Real rho_eps = prim(m,IPR,k,j,i) / gm1;
      //FIXME ERM: Only ideal fluid for now
      Real wgas = prim(m,IDN,k,j,i) + gamma_adi / gm1 * prim(m,IPR,k,j,i);

      auto z2 = prim(m,IVX,k,j,i)*prim(m,IVX,k,j,i) + 
	prim(m,IVY,k,j,i)*prim(m,IVY,k,j,i) +prim(m,IVZ,k,j,i)*prim(m,IVZ,k,j,i);

      
      auto gamma = sqrt(1. +z2);
      cons(m,IDN,k,j,i) = w_d * gamma;
      cons(m,IEN,k,j,i) = wgas*gamma*gamma - w_p - w_d * gamma; //rho_eps * gamma_sq + (w_p + cons(IDN,k,j,i)/(gamma+1.))*(v_sq*gamma_sq);
      cons(m,IM1,k,j,i) = wgas * gamma * w_vx;
      cons(m,IM2,k,j,i) = wgas * gamma * w_vy;
      cons(m,IM3,k,j,i) = wgas * gamma * w_vz;


      Rstiff(m,IVX-noff_,k,j,i) = ( Rstiff(m,IVX-noff_,k,j,i)+cons(m, IVX, k,j,i))/dtI;
      Rstiff(m,IVY-noff_,k,j,i) = ( Rstiff(m,IVY-noff_,k,j,i)+cons(m, IVY, k,j,i))/dtI;
      Rstiff(m,IVZ-noff_,k,j,i) = ( Rstiff(m,IVZ-noff_,k,j,i)+cons(m, IVZ, k,j,i))/dtI;
      Rstiff(m,IEN-noff_,k,j,i) = ( Rstiff(m,IEN-noff_,k,j,i)+cons(m, IEN, k,j,i))/dtI;



    });


}

void TurbulenceDriverHydroRel::ApplyForcingImplicit( DvceArray5D<Real> &force_, DvceArray5D<Real> &u, DvceArray5D<Real> &w, Real const dtI)
{
  auto &ncells = pmy_pack->mb_cells;
  int ncells1 = ncells.nx1 + 2*(ncells.ng);
  int ncells2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
  int ncells3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;

  int &nmb = pmy_pack->nmb_thispack;

  auto& eos_data = pmy_pack->phydro->peos->eos_data;
  Real gm1 = eos_data.gamma - 1.0;
  Real gamma_adi = eos_data.gamma;

  Real &dfloor_ = eos_data.density_floor;
  Real &pfloor_ = eos_data.pressure_floor;
  Real ee_min = pfloor_/gm1;
  
  // Fill explicit prims by inverting u
  // pmy_pack->phydro->peos->ConsToPrim(u,w);


  auto sum_this_mb = ComputeNetMomentum(u, force_);


  Real m0 = sum_this_mb.the_array[IDN];
  Real m1 = sum_this_mb.the_array[IM1];
  Real m2 = sum_this_mb.the_array[IM2];
  Real m3 = sum_this_mb.the_array[IM3];
  Real rhoV = sum_this_mb.the_array[IEN];

//  std::cout << "m0: " << m0 << std::endl;
//  std::cout << "m1: " << m1 << std::endl;
//  std::cout << "m2: " << m2 << std::endl;
//  std::cout << "m3: " << m3 << std::endl;

  m0 = std::max(m0, static_cast<Real>(std::numeric_limits<Real>::min()) );

  auto frce = force_;

  par_for("net_mom_2", DevExeSpace(), 0, nmb-1, 0, (ncells3-1), 0, (ncells2-1), 0, (ncells1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      frce(m,0,k,j,i) -= m1/m0;
      frce(m,1,k,j,i) -= m2/m0;
      frce(m,2,k,j,i) -= m3/m0;
    }
  );


  auto sum_this_mb_en = ComputeNetEnergyInjection(w,force_);

  auto &term1 = sum_this_mb_en.the_array[0];
  auto term2  = sum_this_mb_en.the_array[1] * dtI;
  auto term3  = sum_this_mb_en.the_array[1] * (-1.*dtI*dtI);

//  std::cout << "Fv: " << Fv << std::endl;
//  std::cout << "F2: " << F2 << std::endl;


  auto const tmp = -fabs(term2)/(2.*term3);

  //force normalization
  auto s = tmp + sqrt(tmp*tmp -(dedt - term1)/term3);

  if ( F2 == 0.) s=0.;
//  std::cout << "s: " << s << std::endl;
  if(!std::is_finite(s)) s= dedt;



  par_for("update_prims_implicit", DevExeSpace(), 0, nmb-1, 0, (ncells3-1), 0, (ncells2-1), 0, (ncells1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real u_d  = u(m, IDN,k,j,i);
      Real u_m1 = u(m, IM1,k,j,i);
      Real u_m2 = u(m, IM2,k,j,i);
      Real u_m3 = u(m, IM3,k,j,i);
      Real u_e  = u(m, IEN,k,j,i);

      Real& w_d  = w(m, IDN,k,j,i);
      Real& w_vx = w(m, IVX,k,j,i);
      Real& w_vy = w(m, IVY,k,j,i);
      Real& w_vz = w(m, IVZ,k,j,i);
      Real& w_p  = w(m, IPR,k,j,i);

      // apply density floor, without changing momentum or energy
      u_d = (u_d > dfloor_) ?  u_d : dfloor_;
      w_d = u_d;

      // apply energy floor
      u_e = (u_e > ee_min) ?  u_e : ee_min;

      u_m1 += dtI*s*frce(m,0,k,j,i);
      u_m2 += dtI*s*frce(m,1,k,j,i);
      u_m3 += dtI*s*frce(m,2,k,j,i);

      // Recasting all variables

      auto q = u_e/u_d; // (C2)

      auto r = sqrt(SQR(u_m1)    // (C2)
	          + SQR(u_m2) 
		  + SQR(u_m3))/u_d;

      //Need an upper bound

      auto F2 = frce(m,0,k,j,i)* frce(m,0,k,j,i)+
		frce(m,1,k,j,i)* frce(m,1,k,j,i)+
		frce(m,2,k,j,i)* frce(m,2,k,j,i);

      auto FS = u(m,IVX,k,j,i)* frce(m,0,k,j,i)+
		u(m,IVY,k,j,i)* frce(m,1,k,j,i)+
		u(m,IVZ,k,j,i)* frce(m,2,k,j,i);

      FS/= u_d;

      FS = s*dtI*(FS + s*dtI*F2);

      // Upper bound 
      auto kk = r/(1.+q  );  // (C2) //FIXME + FS ???

      // Enforce lower velocity bound
      // Obeying this bound combined with a floor on 
      // p will guarantuee "some" result of the inversion

      kk = fmin(2.* sqrt(v_sq_max)/(1.+v_sq_max), kk); // (C13)

      // Compute bracket
      auto zm = 0.5*kk/sqrt(1. - 0.25*kk*kk); // (C23)
      auto zp = k/sqrt(1-kk*kk);             // (C23)

      // Evaluate master function
      Real fm,fp;      
      {
	auto &z = zm;
	auto &f = fm;

	auto const W = sqrt(1. + z*z); // (C15)

	w_d = u_d/W; // (C15)

	auto eps = W*q - z*r + z*z / (1.+W) + FS*z/r; // (C16)

	//NOTE: The following generalizes to ANY equation of state
	eps = fmax(pfloor_/w_d/gm1, eps); // (C18)
	w_p = w_d*gm1*eps;
	auto const h = (1. + eps) * ( 1. +  w_p/(w_d*(1.+eps))); // (C1) & (C21)

	f = z - r/h; // (C22)
      }

      {
	auto &z = zp;
	auto &f = fp;

	auto const W = sqrt(1. + z*z); // (C15)

	w_d = u_d/W; // (C15)

	auto eps = W*q - z*r + z*z / (1.+W) + FS*z/r; // (C16)

	//NOTE: The following generalizes to ANY equation of state
	eps = fmax(pfloor_/w_d/gm1, eps); // (C18)
	w_p = w_d*gm1*eps;
	auto const h = (1. + eps) * ( 1. +  w_p/(w_d*(1.+eps))); // (C1) & (C21)

	f = z - r/h; // (C22)
      }

      //For simplicity on the GPU, use the false position method


      Real z,h;
      for(int ii=0; ii< max_iterations; ++ii){

	z =  (zm*fp - zp*fm)/(fp-fm);

	auto const W = sqrt(1. + z*z); // (C15)

	w_d = u_d/W; // (C15)

	auto eps = W*q - z*r + z*z / (1.+W) + FS*z/r; // (C16)

	//NOTE: The following generalizes to ANY equation of state
	eps = fmax(pfloor_/w_d/gm1, eps); // (C18)
	w_p = w_d*gm1*eps;
	h = (1. + eps) * ( 1. +  w_p/(w_d*(1.+eps))); // (C1) & (C21)

	auto f = z - r/h; // (C22)

	// NOTE: both z and f are of order unity
	if((fabs(zm-zp) < tol ) || (fabs(f) < tol )){
	    break;
	}

	if(f * fp < 0.){
	   zm = zp;
	   fm = fp;
	   zp = z;
	   fp = f;
	}
	else{
	   fm = 0.5*fm;
	   zp = z;
	   fp = f;
	}

      }

    auto const conv = 1./(h*u_d); // (C26)
    w_vx = conv * u_m1;           // (C26)
    w_vy = conv * u_m2;           // (C26)
    w_vz = conv * u_m3;           // (C26)

    }
  );


  // Remove momentum normalization, since it will be reused in other substeps
  par_for("fix_force", DevExeSpace(), 0, nmb-1, 0, (ncells3-1), 0, (ncells2-1), 0, (ncells1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      frce(m,0,k,j,i) += m1/m0;
      frce(m,1,k,j,i) += m2/m0;
      frce(m,2,k,j,i) += m3/m0;
    }
  );

  return;
}




array_sum::GlobalSum TurbulenceDriverHydroRel::ComputeNetEnergyInjection(DvceArray5D<Real> &u, DvceArray5D<Real> &ftmp)
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


      auto S = sqrt(u(m, IVX, k,j,i) * u(m, IVX, k,j,i) + u(m, IVY, k,j,i) * u(m, IVY, k,j,i) + u(m, IVZ, k,j,i) * u(m, IVZ, k,j,i));
      auto FS = (u(m, IVX, k,j,i) * force_tmp_(m, 0, k,j,i) + u(m, IVY, k,j,i) * force_tmp_(m, 1, k,j,i) + u(m, IVZ, k,j,i) * force_tmp_(m, 2, k,j,i));
      auto F2 = (force_tmp_(m, 0, k,j,i) * force_tmp_(m, 0, k,j,i) + force_tmp_(m, 1, k,j,i) * force_tmp_(m, 1, k,j,i) + force_tmp_(m, 2, k,j,i) * force_tmp_(m, 2, k,j,i));


      auto dsum = mbsize.dx1.d_view(m) * mbsize.dx2.d_view(m) * mbsize.dx3.d_view(m);

      array_sum::GlobalSum fsum;
      fsum.the_array[0] = (FS/S)*u(m,IDN,k,j,i)*dsum;
      fsum.the_array[1] = (F2/S - FS*FS/(S*S*S))*u(m,IDN,k,j,i)*dsum;
      fsum.the_array[2] = (FS*F2/(S*S*S))*u(m,IDN,k,j,i)*dsum;

      mb_sum += fsum;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb)
  );

#if MPI_PARALLEL_ENABLED
  {
    // Does this work on GPU?
    MPI_Allreduce(MPI_IN_PLACE, sum_this_mb.the_array,3, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
  }
#endif

  return sum_this_mb;
};

array_sum::GlobalSum TurbulenceDriverHydroRel::ComputeNetMomentum(DvceArray5D<Real> &u, DvceArray5D<Real> &ftmp)
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

