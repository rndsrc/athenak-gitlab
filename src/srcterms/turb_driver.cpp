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
    ImEx::allocate_storage(1,5);
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
//! \fn  Initialize forcing


void TurbulenceDriver::Initialize()
{

  if(initialized)  return;

  int &nx1 = pmy_pack->mb_cells.nx1;
  int &nx2 = pmy_pack->mb_cells.nx2;
  int &nx3 = pmy_pack->mb_cells.nx3;
  auto &ncells = pmy_pack->mb_cells;
  int ncells1 = ncells.nx1 + 2*(ncells.ng);
  int ncells2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
  int ncells3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;

  int &is = pmy_pack->mb_cells.is, &ie = pmy_pack->mb_cells.ie;
  int &js = pmy_pack->mb_cells.js, &je = pmy_pack->mb_cells.je;
  int &ks = pmy_pack->mb_cells.ks, &ke = pmy_pack->mb_cells.ke;

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

    initialized = true;
}  

void TurbulenceDriver::NewRandomForce(DvceArray5D<Real> &ftmp)
{

  if(!initialized) Initialize();

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

  // Followed code executed every time
  auto force_tmp_ = ftmp;
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


  return;
}





