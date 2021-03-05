//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_driver.cpp
//  \brief implementation of functions in TurbulenceDriver

#include <limits>
#include <algorithm>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "utils/grid_locations.hpp"
#include "utils/random.hpp"
#include "turb_driver.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

TurbulenceDriver::TurbulenceDriver(MeshBlockPack *pp, ParameterInput *pin) :
  ImEx(pp,pin),
  first_time_(true),
  force("force",1,1,1,1,1),
  force_tmp("force_tmp",1,1,1,1,1),
  x1sin("x1sin",1,1,1),
  x1cos("x1cos",1,1,1),
  x2sin("x2sin",1,1,1),
  x2cos("x2cos",1,1,1),
  x3sin("x3sin",1,1,1),
  x3cos("x3cos",1,1,1),
  amp1("amp1",1,1,1),
  amp2("amp2",1,1,1),
  amp3("amp3",1,1,1),
  seeds("seeds",1,1)
{
  // allocate memory for force registers
  int nmb = pmy_pack->nmb_thispack;
  auto &ncells = pmy_pack->mb_cells;
  int ncells1 = ncells.nx1 + 2*(ncells.ng);
  int ncells2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
  int ncells3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;

  Kokkos::realloc(force, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(force_tmp, nmb, 3, ncells3, ncells2, ncells1);

  // Implicit or explicit sources?


  std::string evolution_t = pin->GetString("forcing","sources");
  // Default is explicit integration
  if(evolution_t != "implicit"){
    ImEx::this_imex = ImEx::method::RKexplicit;
  }else{
    ImEx::allocate_storage(0,5);
  }



  // range of modes including, corresponding to kmin and kmax
  nlow = pin->GetOrAddInteger("forcing","nlow",1);
  nhigh = pin->GetOrAddInteger("forcing","nhigh",2);
  if (ncells3>1) { // 3D
    ntot = (nhigh+1)*(nhigh+1)*(nhigh+1);
    nwave = 8;
  } else if (ncells2>1) { // 2D
    ntot = (nhigh+1)*(nhigh+1);
    nwave = 4;
  } else { // 1D
    ntot = (nhigh+1);
    nwave = 2;
  }
  // power-law exponent for isotropic driving
  expo = pin->GetOrAddReal("forcing","expo",5.0/3.0);
  // energy injection rate
  dedt = pin->GetOrAddReal("forcing","dedt",0.0);
  // correlation time
  tcorr = pin->GetOrAddReal("forcing","tcorr",0.0); 

  Kokkos::realloc(x1sin, nmb, ntot, ncells1);
  Kokkos::realloc(x1cos, nmb, ntot, ncells1);
  Kokkos::realloc(x2sin, nmb, ntot, ncells2);
  Kokkos::realloc(x2cos, nmb, ntot, ncells2);
  Kokkos::realloc(x3sin, nmb, ntot, ncells3);
  Kokkos::realloc(x3cos, nmb, ntot, ncells3);

  Kokkos::realloc(amp1, nmb, ntot, nwave);
  Kokkos::realloc(amp2, nmb, ntot, nwave);
  Kokkos::realloc(amp3, nmb, ntot, nwave);

  Kokkos::realloc(seeds, nmb, ntot);
}

//----------------------------------------------------------------------------------------
//! \fn  apply forcing

void TurbulenceDriver::ApplyForcing(DvceArray5D<Real> &u)
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

  int &nt = ntot;
  int &nw = nwave;

  int &nmb = pmy_pack->nmb_thispack;

  // Following code initializes driving, and so is only executed once at start of calc.
  // Cannot be included in constructor since (it seems) Kokkos::par_for not allowed in cons.
  if (first_time_) {

    // initialize force registers/amps to zero
    auto force_ = force;
    auto force_tmp_ = force_tmp;
    par_for("force_init", DevExeSpace(), 0, nmb-1, 0, 2, 0, ncells3-1, 0, ncells2-1, 
      0, ncells1-1, KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
      {
        force_(m,n,k,j,i) = 0.0;
        force_tmp_(m,n,k,j,i) = 0.0;
      }
    );

    auto amp1_ = amp1;
    auto amp2_ = amp2;
    auto amp3_ = amp3;
    par_for("amp_init", DevExeSpace(), 0, nmb-1, 0, nt-1, 0, nw-1,
      KOKKOS_LAMBDA(int m, int n, int nw)
      {
        amp1_(m,n,nw) = 0.0;
        amp2_(m,n,nw) = 0.0;
        amp3_(m,n,nw) = 0.0;
      }
    );

    // initalize seeds
    auto seeds_ = seeds;
    par_for("seeds_init", DevExeSpace(), 0, nmb-1, 0, nt-1,
      KOKKOS_LAMBDA(int m, int n)
      {
        seeds_(m,n) = n + n*n + n*n*n; // make sure seed is different for each harmonic
      }
    );

    int nw2 = 1;
    int nw3 = 1;
    if (ncells2>1) {
      nw2 = nhigh+1;
    }
    if (ncells3>1) {
      nw3 = nhigh+1;
    }
    int nw23 = nw3*nw2;

    auto x1sin_ = x1sin;
    auto x1cos_ = x1cos;

    // Initialize sin and cos arrays
    // bad design: requires saving sin/cos during restarts
    int &nx1 = pmy_pack->mb_cells.nx1;
    auto &size = pmy_pack->pmb->mbsize;
    par_for("kx_loop", DevExeSpace(), 0, nmb-1, 0, nt-1, 0, ncells1-1,
      KOKKOS_LAMBDA(int m, int n, int i)
      { 
        int nk1 = n/nw23;
        Real kx = nk1*dkx;
        Real x1v = CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m));
      
        x1sin_(m,n,i) = sin(kx*x1v);
        x1cos_(m,n,i) = cos(kx*x1v);
      }
    );

    auto x2sin_ = x2sin;
    auto x2cos_ = x2cos;
    int &nx2 = pmy_pack->mb_cells.nx2;
    par_for("ky_loop", DevExeSpace(), 0, nmb-1, 0, nt-1, 0, ncells2-1,
      KOKKOS_LAMBDA(int m, int n, int j)
      { 
        int nk1 = n/nw23;
        int nk2 = (n - nk1*nw23)/nw2;
        Real ky = nk2*dky;
        Real x2v = CellCenterX(j-js, nx2, size.x2min.d_view(m), size.x2max.d_view(m));

        x2sin_(m,n,j) = sin(ky*x2v);
        x2cos_(m,n,j) = cos(ky*x2v);
      }
    );

    auto x3sin_ = x3sin;
    auto x3cos_ = x3cos;
    int &nx3 = pmy_pack->mb_cells.nx3;
    par_for("kz_loop", DevExeSpace(), 0, nmb-1, 0, nt-1, 0, ncells3-1,
      KOKKOS_LAMBDA(int m, int n, int k)
      { 
        int nk1 = n/nw23;
        int nk2 = (n - nk1*nw23)/nw2;
        int nk3 = n - nk1*nw23 - nk2*nw2;
        Real kz = nk3*dkz;
        Real x3v = CellCenterX(k-ks, nx3, size.x3min.d_view(m), size.x3max.d_view(m));
  
        x3sin_(m,n,k) = sin(kz*x3v);
        x3cos_(m,n,k) = cos(kz*x3v);
      }
    );

    first_time_ = false;
  }  // end of initialization

  // Followed code executed every time
  auto force_tmp_ = force_tmp;
  par_for("forcing_init", DevExeSpace(), 0, nmb-1, 0, ncells3-1, 0, ncells2-1, 
    0, ncells1-1, KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      force_tmp_(m,0,k,j,i) = 0.0;
      force_tmp_(m,1,k,j,i) = 0.0;
      force_tmp_(m,2,k,j,i) = 0.0;
    }
  );

  int nlow_sq  = nlow*nlow;
  int nhigh_sq = nhigh*nhigh;

  int nw2 = 1;
  int nw3 = 1;
  if (ncells2>1) {
    nw2 = nhigh+1;
  }
  if (ncells3>1) {
    nw3 = nhigh+1;
  }
  int nw23 = nw3*nw2;

  Real &ex = expo;
  auto seeds_ = seeds;
  auto amp1_ = amp1;
  auto amp2_ = amp2;
  auto amp3_ = amp3;
  par_for ("generate_amplitudes", DevExeSpace(), 0, nmb-1, 0, nt-1,
    KOKKOS_LAMBDA (int m, int n) 
    {
      int nk1, nk2, nk3, nsq;
      Real kx, ky, kz, norm, kmag;
      Real iky, ikz;

      nk1 = n/nw23;
      nk2 = (n - nk1*nw23)/nw2;
      nk3 = n - nk1*nw23 - nk2*nw2;
      kx = nk1*dkx;
      ky = nk2*dky;
      kz = nk3*dkz;

      nsq = nk1*nk1 + nk2*nk2 + nk3*nk3;

      kmag = sqrt(kx*kx + ky*ky + kz*kz);
      norm = 1.0/pow(kmag,(ex+2.0)/2.0); 

      // TODO(leva): check whether those coefficients are needed
      //if(nk1 > 0) norm *= 0.5;
      //if(nk2 > 0) norm *= 0.5;
      //if(nk3 > 0) norm *= 0.5;

      if (nsq >= nlow_sq && nsq <= nhigh_sq) {
        //Generate Fourier amplitudes
        if(nk3 != 0){
          ikz = 1.0/(dkz*((Real) nk3));

          amp1_(m,n,0) = RanGaussian(&(seeds_(m,n)));
          amp1_(m,n,1) = RanGaussian(&(seeds_(m,n)));
          amp1_(m,n,2) = (nk2 == 0)             ? 0.0 :RanGaussian(&(seeds_(m,n)));
          amp1_(m,n,3) = (nk2 == 0)             ? 0.0 :RanGaussian(&(seeds_(m,n)));
          amp1_(m,n,4) = (nk1 == 0)             ? 0.0 :RanGaussian(&(seeds_(m,n)));
          amp1_(m,n,5) = (nk1 == 0)             ? 0.0 :RanGaussian(&(seeds_(m,n)));
          amp1_(m,n,6) = (nk1 == 0 || nk2 == 0) ? 0.0 :RanGaussian(&(seeds_(m,n)));
          amp1_(m,n,7) = (nk1 == 0 || nk2 == 0) ? 0.0 :RanGaussian(&(seeds_(m,n)));

          amp2_(m,n,0) = RanGaussian(&(seeds_(m,n)));
          amp2_(m,n,1) = RanGaussian(&(seeds_(m,n)));
          amp2_(m,n,2) = (nk2 == 0)             ? 0.0 :RanGaussian(&(seeds_(m,n)));
          amp2_(m,n,3) = (nk2 == 0)             ? 0.0 :RanGaussian(&(seeds_(m,n)));
          amp2_(m,n,4) = (nk1 == 0)             ? 0.0 :RanGaussian(&(seeds_(m,n)));
          amp2_(m,n,5) = (nk1 == 0)             ? 0.0 :RanGaussian(&(seeds_(m,n)));
          amp2_(m,n,6) = (nk1 == 0 || nk2 == 0) ? 0.0 :RanGaussian(&(seeds_(m,n)));
          amp2_(m,n,7) = (nk1 == 0 || nk2 == 0) ? 0.0 :RanGaussian(&(seeds_(m,n)));

          // incompressibility
          amp3_(m,n,0) =  ikz*( kx*amp1_(m,n,5) + ky*amp2_(m,n,3));
          amp3_(m,n,1) = -ikz*( kx*amp1_(m,n,4) + ky*amp2_(m,n,2));
          amp3_(m,n,2) =  ikz*( kx*amp1_(m,n,7) - ky*amp2_(m,n,1));
          amp3_(m,n,3) =  ikz*(-kx*amp1_(m,n,6) + ky*amp2_(m,n,0));
          amp3_(m,n,4) =  ikz*(-kx*amp1_(m,n,1) + ky*amp2_(m,n,7));
          amp3_(m,n,5) =  ikz*( kx*amp1_(m,n,0) - ky*amp2_(m,n,6));
          amp3_(m,n,6) = -ikz*( kx*amp1_(m,n,3) + ky*amp2_(m,n,5));
          amp3_(m,n,7) =  ikz*( kx*amp1_(m,n,2) + ky*amp2_(m,n,4));

        } else if(nk2 != 0){ // kz == 0
          iky = 1.0/(dky*((Real) nk2));

          amp1_(m,n,0) = RanGaussian(&(seeds_(m,n)));
          amp1_(m,n,2) = RanGaussian(&(seeds_(m,n)));
          amp1_(m,n,4) = (nk1 == 0) ? 0.0 :RanGaussian(&(seeds_(m,n)));
          amp1_(m,n,6) = (nk1 == 0) ? 0.0 :RanGaussian(&(seeds_(m,n)));
          amp1_(m,n,1) = 0.0;
          amp1_(m,n,3) = 0.0;
          amp1_(m,n,5) = 0.0;
          amp1_(m,n,7) = 0.0;

          amp3_(m,n,0) = RanGaussian(&(seeds_(m,n)));
          amp3_(m,n,2) = RanGaussian(&(seeds_(m,n)));
          amp3_(m,n,4) = (nk1 == 0) ? 0.0 : RanGaussian(&(seeds_(m,n)));
          amp3_(m,n,6) = (nk1 == 0) ? 0.0 : RanGaussian(&(seeds_(m,n)));
          amp3_(m,n,1) = 0.0;
          amp3_(m,n,3) = 0.0;
          amp3_(m,n,5) = 0.0;
          amp3_(m,n,7) = 0.0;

          // incompressibility
          amp2_(m,n,0) =  iky*kx*amp1_(m,n,6);
          amp2_(m,n,2) = -iky*kx*amp1_(m,n,4);
          amp2_(m,n,4) = -iky*kx*amp1_(m,n,2);
          amp2_(m,n,6) =  iky*kx*amp1_(m,n,0);
          amp2_(m,n,1) = 0.0;
          amp2_(m,n,3) = 0.0;
          amp2_(m,n,5) = 0.0;
          amp2_(m,n,7) = 0.0;

        } else {// kz == ky == 0, kx != 0 by initial if statement
          amp3_(m,n,0) = RanGaussian(&(seeds_(m,n)));
          amp3_(m,n,4) = RanGaussian(&(seeds_(m,n)));
          amp3_(m,n,1) = 0.0;
          amp3_(m,n,2) = 0.0;
          amp3_(m,n,3) = 0.0;
          amp3_(m,n,5) = 0.0;
          amp3_(m,n,6) = 0.0;
          amp3_(m,n,7) = 0.0;

          amp2_(m,n,0) = RanGaussian(&(seeds_(m,n)));
          amp2_(m,n,4) = RanGaussian(&(seeds_(m,n)));
          amp2_(m,n,1) = 0.0;
          amp2_(m,n,2) = 0.0;
          amp2_(m,n,3) = 0.0;
          amp2_(m,n,5) = 0.0;
          amp2_(m,n,6) = 0.0;
          amp2_(m,n,7) = 0.0;

          // incompressibility
          amp1_(m,n,0) = 0.0;
          amp1_(m,n,4) = 0.0;
          amp1_(m,n,1) = 0.0;
          amp1_(m,n,2) = 0.0;
          amp1_(m,n,3) = 0.0;
          amp1_(m,n,5) = 0.0;
          amp1_(m,n,6) = 0.0;
          amp1_(m,n,7) = 0.0;
        }

        amp1_(m,n,0) *= norm;
        amp1_(m,n,4) *= norm;
        amp1_(m,n,1) *= norm;
        amp1_(m,n,2) *= norm;
        amp1_(m,n,3) *= norm;
        amp1_(m,n,5) *= norm;
        amp1_(m,n,6) *= norm;
        amp1_(m,n,7) *= norm;

        amp2_(m,n,0) *= norm;
        amp2_(m,n,4) *= norm;
        amp2_(m,n,1) *= norm;
        amp2_(m,n,2) *= norm;
        amp2_(m,n,3) *= norm;
        amp2_(m,n,5) *= norm;
        amp2_(m,n,6) *= norm;
        amp2_(m,n,7) *= norm;

        amp3_(m,n,0) *= norm;
        amp3_(m,n,4) *= norm;
        amp3_(m,n,1) *= norm;
        amp3_(m,n,2) *= norm;
        amp3_(m,n,3) *= norm;
        amp3_(m,n,5) *= norm;
        amp3_(m,n,6) *= norm;
        amp3_(m,n,7) *= norm;
      } 
    }
  );

  auto x1cos_ = x1cos;
  auto x1sin_ = x1sin;
  auto x2cos_ = x2cos;
  auto x2sin_ = x2sin;
  auto x3cos_ = x3cos;
  auto x3sin_ = x3sin;
  par_for("force_array", DevExeSpace(), 0, nmb-1, 0, ncells3-1, 0, ncells2-1, 
    0, ncells1-1, KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      for (int n=0; n<nt; n++) {
        int n1 = n/nw23;
        int n2 = (n - n1*nw23)/nw2;
        int n3 = n - n1*nw23 - n2*nw2;
        int nsqr = n1*n1 + n2*n2 + n3*n3;

        if (nsqr >= nlow_sq && nsqr <= nhigh_sq) {
          force_tmp_(m,0,k,j,i) += amp1_(m,n,0)*x1cos_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                   amp1_(m,n,1)*x1cos_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                   amp1_(m,n,2)*x1cos_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                   amp1_(m,n,3)*x1cos_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k)+
                                   amp1_(m,n,4)*x1sin_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                   amp1_(m,n,5)*x1sin_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                   amp1_(m,n,6)*x1sin_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                   amp1_(m,n,7)*x1sin_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k);
          force_tmp_(m,1,k,j,i) += amp2_(m,n,0)*x1cos_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                   amp2_(m,n,1)*x1cos_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                   amp2_(m,n,2)*x1cos_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                   amp2_(m,n,3)*x1cos_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k)+
                                   amp2_(m,n,4)*x1sin_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                   amp2_(m,n,5)*x1sin_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                   amp2_(m,n,6)*x1sin_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                   amp2_(m,n,7)*x1sin_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k);
          force_tmp_(m,2,k,j,i) += amp3_(m,n,0)*x1cos_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                   amp3_(m,n,1)*x1cos_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                   amp3_(m,n,2)*x1cos_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                   amp3_(m,n,3)*x1cos_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k)+
                                   amp3_(m,n,4)*x1sin_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                   amp3_(m,n,5)*x1sin_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                   amp3_(m,n,6)*x1sin_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                   amp3_(m,n,7)*x1sin_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k);
        }
      }
    }
  );


  if(implicit_update){
	  ApplyForcingSourceTermsImplicit(u);
  }else{
	  ApplyForcingSourceTermsExplicit(u);
  };

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  apply forcing (explicit version)

void TurbulenceDriver::ApplyForcingSourceTermsExplicit(DvceArray5D<Real> &u)
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

  int &nt = ntot;
  int &nw = nwave;

  int &nmb = pmy_pack->nmb_thispack;

  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  array_sum::GlobalSum sum_this_mb;
  Kokkos::parallel_reduce("net_mom_1", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum)
    {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      array_sum::GlobalSum fsum;
      fsum.the_array[IDN] = 1.0;
      fsum.the_array[IM1] = u(m,IDN,k,j,i)*force_tmp_(m,0,k,j,i);
      fsum.the_array[IM2] = u(m,IDN,k,j,i)*force_tmp_(m,1,k,j,i);
      fsum.the_array[IM3] = u(m,IDN,k,j,i)*force_tmp_(m,2,k,j,i);

      mb_sum += fsum;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb)
  );

  Real m0 = sum_this_mb.the_array[IDN];
  Real m1 = sum_this_mb.the_array[IM1];
  Real m2 = sum_this_mb.the_array[IM2];
  Real m3 = sum_this_mb.the_array[IM3];

  m0 = std::max(m0, static_cast<Real>(std::numeric_limits<float>::min()) );

  // TODO(leva): add MPI call for gm[]

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
  
  Real dvol = 1.0/(nx1*nx2*nx3); // old: Lx*Ly*Lz/nx1/nx2/nx3;
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

void TurbulenceDriver::ApplySourceTermsImplicitPreStage(DvceArray5D<Real> &u, DvceArray5D<Real> &w)
{

    //switch stages:
    //
    //
      int ncells1 = pmy_pack->mb_cells.nx1 + 2*(pmy_pack->mb_cells.ng);
      int nmb = pmy_pack->nmb_thispack;
      int nvar = nimplicit;
      auto &mbsize = pmy_pack->pmb->mbsize;
      auto u0_ = u;

      auto Ru1_ = Ru1;
      auto Ru2_ = Ru2;
      auto Ru3_ = Ru3;

      double const alphaI = 0.24169426078821; // 1./3.;
      double const betaI = 0.06042356519705;
      double const etaI = 0.12915286960590;

      double dtI = (pmy_pack->pmesh->dt); 

      int is = pmy_pack->mb_cells.is; int ie = pmy_pack->mb_cells.ie;
      int js = pmy_pack->mb_cells.js; int je = pmy_pack->mb_cells.je;
      int ks = pmy_pack->mb_cells.ks; int ke = pmy_pack->mb_cells.ke;

      auto ncells = pmy_pack->mb_cells;
      int ng = ncells.ng;
      int n1 = ncells.nx1 + 2*ng;
      int n2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*ng) : 1;
      int n3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*ng) : 1;

      int nimplicit = 5;
    
    //FIXME Hard wired to RK3 for now
    //
    //

      ImplicitKernel(u,w,alphaI*dtI,Ru1);

	par_for("implicit_stage2", DevExeSpace(), 0, (nmb-1),0, nimplicit-1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
	  KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	  {
	    u0_(m,n,k,j,i) = u0_(m,n,k,j,i) -2.*alphaI*dtI * Ru1_(m,n,k,j,i);
	  });
      ImplicitKernel(u,w,alphaI*dtI,Ru2);
};

void TurbulenceDriver::ApplyForcingSourceTermsImplicit(DvceArray5D<Real> &u, DvceArray5D<Real> &w, int stage)
{

    //switch stages:
    //
    //
      int ncells1 = pmy_pack->mb_cells.nx1 + 2*(pmy_pack->mb_cells.ng);
      int nmb = pmy_pack->nmb_thispack;
      int nvar = nimplicit;
      auto &mbsize = pmy_pack->pmb->mbsize;
      auto u0_ = u0;
      auto u1_ = u1;

      auto Ru1_ = Ru1;
      auto Ru2_ = Ru2;
      auto Ru3_ = Ru3;

      double const alphaI = 0.24169426078821; // 1./3.;
      double const betaI = 0.06042356519705;
      double const etaI = 0.12915286960590;

      double dtI = (pmy_pack->pmesh->dt); 

      int is = pmy_pack->mb_cells.is; int ie = pmy_pack->mb_cells.ie;
      int js = pmy_pack->mb_cells.js; int je = pmy_pack->mb_cells.je;
      int ks = pmy_pack->mb_cells.ks; int ke = pmy_pack->mb_cells.ke;

      auto ncells = pmy_pack->mb_cells;
      int ng = ncells.ng;
      int n1 = ncells.nx1 + 2*ng;
      int n2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*ng) : 1;
      int n3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*ng) : 1;

    
    //FIXME Hard wired to RK3 for now
    //
    //

    switch(stage){

      case 1:
	  par_for("implicit_stage3", DevExeSpace(), 0, (nmb-1),0, nimplicit-1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
	    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	    {
	      u0_(m,n+ISIndex::IPIXX,k,j,i) = u0_(m,n+ISIndex::IPIXX,k,j,i) + 
	      				  (1.-2.*alphaI) * dtI * Ru2_(m,n,k,j,i) + alphaI*dtI* Ru1_(m,n,k,j,i);
	    });
	peosIS->ConsToPrimImplicit(u0,w0,alphaI*dtI, Ru3);
	break;

      case 2:
	  par_for("implicit_stage4a", DevExeSpace(), 0, (nmb-1),0, nimplicit-1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
	    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	    {
	      u0_(m,n+ISIndex::IPIXX,k,j,i) = u0_(m,n+ISIndex::IPIXX,k,j,i) + 
		betaI*dtI * Ru1_(m,n,k,j,i) + (etaI- 0.25*(1.-alphaI)) * dtI * Ru2_(m,n,k,j,i) 
		+ (0.5 - betaI - etaI - 1.25*alphaI)*dtI* Ru3_(m,n,k,j,i);
	    });

	  par_for("implicit_stage4b", DevExeSpace(), 0, (nmb-1),0, nimplicit-1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
	    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	    {
	      Ru2_(m,n,k,j,i) = - (2./3.) *betaI*dtI * Ru1_(m,n,k,j,i) + ((1.-4.*etaI)/6.) * dtI * Ru2_(m,n,k,j,i);
	    });

	peosIS->ConsToPrimImplicit(u0,w0,alphaI*dtI,Ru1);
	break;

      case 3:
	  par_for("implicit_stage5", DevExeSpace(), 0, (nmb-1),0, nimplicit-1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
	    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	    {
	      u0_(m,n+ISIndex::IPIXX,k,j,i) = u0_(m,n+ISIndex::IPIXX,k,j,i) 
	      			+ Ru2_(m,n,k,j,i) + (-1.0 + 4.*(betaI + etaI +alphaI))/6.*dtI* Ru3_(m,n,k,j,i)
	                                      + (2./3.)*(1.-alphaI) *dtI * Ru1_(m,n,k,j,i);
	    });
	peosIS->ConsToPrim(u0,w0);
	break;

    };
};



