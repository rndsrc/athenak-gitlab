//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kh.cpp
//  \brief Problem generator for KH instability
//  Sets up different initial conditions selected by flag "iprob"
//    - iprob=1 : tanh profile with a single mode perturbation
//    - iprob=2 : double tanh profile with a single mode perturbation

#include <iostream>
#include <sstream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"
#include "particles/particles.hpp"

#include <Kokkos_Random.hpp>

//----------------------------------------------------------------------------------------
//! \fn
//  \brief Problem Generator for KHI tests

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  // read problem parameters from input file
  int iprob  = pin->GetReal("problem","iprob");
  Real amp   = pin->GetReal("problem","amp");
  Real sigma = pin->GetReal("problem","sigma");
  Real vshear= pin->GetReal("problem","vshear");
  Real rho0  = pin->GetOrAddReal("problem","rho0",1.0);
  Real rho1  = pin->GetOrAddReal("problem","rho1",1.0);
  Real drho_rho0 = pin->GetOrAddReal("problem", "drho_rho0", 0.0);

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // Select either Hydro or MHD
  DvceArray5D<Real> w0_;
  Real gm1;
  int nfluid, nscalars;
  if (pmbp->phydro != nullptr) {
    w0_ = pmbp->phydro->w0;
    gm1 = (pmbp->phydro->peos->eos_data.gamma) - 1.0;
    nfluid = pmbp->phydro->nhydro;
    nscalars = pmbp->phydro->nscalars;
  } else if (pmbp->pmhd != nullptr) {
    w0_ = pmbp->pmhd->w0;
    gm1 = (pmbp->pmhd->peos->eos_data.gamma) - 1.0;
    nfluid = pmbp->pmhd->nmhd;
    nscalars = pmbp->pmhd->nscalars;
  }

  if (nscalars == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "KH test requires nscalars != 0" << std::endl;
    exit(EXIT_FAILURE);
  }

  // initialize primitive variables
  par_for("pgen_kh1", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    w0_(m,IEN,k,j,i) = 20.0/gm1;
    w0_(m,IVZ,k,j,i) = 0.0;

    // Lorentz factor (needed to initializve 4-velocity in SR)
    Real u00 = 1.0;
    bool is_relativistic = false;
    if (pmbp->pcoord->is_special_relativistic ||
        pmbp->pcoord->is_general_relativistic) {
      is_relativistic = true;
    }

    Real dens,pres,vx,vy,vz,scal;

    if (iprob == 1) {
      pres = 20.0;
      dens = 1.0;
      vx = -vshear*tanh(x2v/sigma);
      vy = -amp*vshear*sin(2.*M_PI*x1v)*exp( -SQR(x2v/sigma) );
      vz = 0.0;
      scal = 0.0;
      if (x2v > 0.0) scal = 1.0;
    } else if (iprob == 2) {
      pres = 1.0;
      vz = 0.0;
      if(x2v <= 0.0) {
        dens = rho0 - rho1*tanh((x2v-0.5)/sigma);
        vx = -vshear*tanh((x2v-0.5)/sigma);
        vy = -amp*vshear*sin(2.*M_PI*x1v)*exp( -SQR((x2v-0.5)/sigma) );
        if (is_relativistic) {
          u00 = 1.0/sqrt(1.0 - vx*vx - vy*vy);
        }
        scal = 0.0;
        if (x2v < 0.5) scal = 1.0;
      } else {
        dens = rho0 + rho1*tanh((x2v-0.5)/sigma);
        vx = vshear*tanh((x2v-0.5)/sigma);
        vy = amp*vshear*sin(2.*M_PI*x1v)*exp( -SQR((x2v-0.5)/sigma) );
        if (is_relativistic) {
          u00 = 1.0/sqrt(1.0 - vx*vx - vy*vy);
        }
        scal = 0.0;
        if (x2v > 0.5) scal = 1.0;
      }
    // Lecoanet test ICs
    } else if (iprob == 4) {
      pres = 10.0;
      Real a = 0.05;
      dens = 1.0 + 0.5*drho_rho0*(tanh((x2v + 0.5)/a) - tanh((x2v - 0.5)/a));
      vx = vshear*(tanh((x2v + 0.5)/a) - tanh((x2v - 0.5)/a) - 1.0);
      Real ave_sine = sin(2.*M_PI*x1v);
      if (x1v > 0.0) {
        ave_sine -= sin(2.*M_PI*(-0.5 + x1v));
      } else {
        ave_sine -= sin(2.*M_PI*(0.5 + x1v));
      }
      ave_sine /= 2.0;

      // translated x1= x - 1/2 relative to Lecoanet (2015) shifts sine function by pi
      // (half-period) and introduces U_z sign change:
      vy = -amp*ave_sine*
            (exp(-(SQR(x2v + 0.5))/(sigma*sigma)) + exp(-(SQR(x2v - 0.5))/(sigma*sigma)));
      scal = 0.5*(tanh((x2v + 0.5)/a) - tanh((x2v - 0.5)/a) + 2.0);
      vz = 0.0;
    }

    // set primitives in both newtonian and SR hydro
    w0_(m,IDN,k,j,i) = dens;
    w0_(m,IEN,k,j,i) = pres/gm1;
    w0_(m,IVX,k,j,i) = u00*vx;
    w0_(m,IVY,k,j,i) = u00*vy;
    w0_(m,IVZ,k,j,i) = u00*vz;
    // add passive scalars
    for (int n=nfluid; n<(nfluid+nscalars); ++n) {
      w0_(m,n,k,j,i) = scal;
    }
  });

  // initialize magnetic fields if MHD
  if (pmbp->pmhd != nullptr) {
    // Read magnetic field strength
    Real bx = pin->GetReal("problem","b0");
    auto &b0 = pmbp->pmhd->b0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    par_for("pgen_b0", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) = bx;
      b0.x2f(m,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = 0.0;
      if (i==ie) b0.x1f(m,k,j,i+1) = bx;
      if (j==je) b0.x2f(m,k,j+1,i) = 0.0;
      if (k==ke) b0.x3f(m,k+1,j,i) = 0.0;
      bcc0(m,IBX,k,j,i) = bx;
      bcc0(m,IBY,k,j,i) = 0.0;
      bcc0(m,IBZ,k,j,i) = 0.0;
    });
  }

  // Convert primitives to conserved
  if (pmbp->phydro != nullptr) {
    auto &u0_ = pmbp->phydro->u0;
    pmbp->phydro->peos->PrimToCons(w0_, u0_, is, ie, js, je, ks, ke);
  } else if (pmbp->pmhd != nullptr) {
    auto &u0_ = pmbp->pmhd->u0;
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);
  }

  // Initialize particles
  if (pmbp->ppart != nullptr) {

    // captures for the kernel
    auto &u0_ = (pmbp->phydro != nullptr) ? pmbp->phydro->u0 : pmbp->pmhd->u0;
    auto &mbsize = pmy_mesh_->pmb_pack->pmb->mb_size;
    auto &mblev = pmy_mesh_->pmb_pack->pmb->mb_lev;
    auto gids = pmy_mesh_->pmb_pack->gids;

    auto &indcs = pmy_mesh_->mb_indcs;
    int &is = indcs.is;
    int &js = indcs.js;
    int &ks = indcs.ks;
    int &nx1 = indcs.nx1;
    int &nx2 = indcs.nx2;
    int &nx3 = indcs.nx3;

    // init RNG
    Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);

    // count total mass across the domain
    Real total_mass = 0.0;
    const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
    const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
    const int nji  = indcs.nx2*indcs.nx1;

    Kokkos::parallel_reduce("pgen_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &total_mass) {
      // compute m,k,j,i indices of thread and evaluate
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/indcs.nx1;
      int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
      k += ks;
      j += js;

      Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2*mbsize.d_view(m).dx3;
      total_mass += u0_(m,IDN,k,j,i) * vol;
    }, total_mass);

    Real total_mass_thispack = total_mass;

#if MPI_PARALLEL_ENABLED
    // get total mass over all MPI ranks
    MPI_Allreduce(MPI_IN_PLACE, &total_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    // get number of particles for this mbpack using MC to deal with fractional particles
    Real target_nparticles = pin->GetOrAddReal("particles","target_count",100000.0);
    Real mass_per_particle = total_mass / target_nparticles;

    // create shared array to hold number of particles per zone
    DualArray2D<int> nparticles_per_zone("partperzone", nmkji,2);
    par_for("particle_count", DevExeSpace(), 0,nmkji-1,
    KOKKOS_LAMBDA(int idx) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/indcs.nx1;
      int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
      k += ks;
      j += js;
      Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2*mbsize.d_view(m).dx3;
      Real nppc = u0_(m,IDN,k,j,i) * vol / mass_per_particle;
      int nparticles = static_cast<int>(nppc);
      nppc = fabs(fmod(nppc, 1.0));
      auto rand_gen = rand_pool64.get_state();
      if (rand_gen.frand() < nppc) {
        nparticles += 1;
      }
      nparticles_per_zone.d_view(idx,0) = nparticles;
      rand_pool64.free_state(rand_gen); 
    });

    // count total number of particles in this pack
    nparticles_per_zone.template modify<DevExeSpace>();
    nparticles_per_zone.template sync<HostMemSpace>();
    int nparticles_thispack = 0;
    for (int i=0; i<nmkji; ++i) {
      nparticles_per_zone.h_view(i,1) = nparticles_thispack;
      nparticles_thispack += nparticles_per_zone.h_view(i,0);
    }
    nparticles_per_zone.template modify<HostMemSpace>();
    nparticles_per_zone.template sync<DevMemSpace>();

    // helpful debug statement
    std::cout << "total mass across domain: " << total_mass << ", total mass in pack: " << total_mass_thispack
              << ", target nparticles: " << target_nparticles << ", nparticles in pack: " << nparticles_thispack
              << std::endl;

    // reallocate space for particles and get relevant pointers
    pmy_mesh_->pmb_pack->ppart->ReallocateParticles(nparticles_thispack);

    auto &pr = pmy_mesh_->pmb_pack->ppart->prtcl_rdata;
    auto &pi = pmy_mesh_->pmb_pack->ppart->prtcl_idata;

    // initialize particles. only intended for Lagrangian-type particles
    par_for("part_init", DevExeSpace(), 0,nmkji-1,
    KOKKOS_LAMBDA(int idx) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/indcs.nx1;
      int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
      k += ks;
      j += js;

      int nparticles_in_zone = nparticles_per_zone.d_view(idx,0);
      int starting_index = nparticles_per_zone.d_view(idx,1);

      for (int p=0; p<nparticles_in_zone; ++p) {
        int pidx = p + starting_index;

        pi(PGID,pidx) = gids + m;
        pi(PLASTLEVEL,pidx) = mblev.d_view(m);

        // set particle to zone center
        pr(IPX,pidx) = CellCenterX(i-is, nx1, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
        pr(IPY,pidx) = CellCenterX(j-js, nx2, mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
        pr(IPZ,pidx) = CellCenterX(k-ks, nx3, mbsize.d_view(m).x3min, mbsize.d_view(m).x3max) -
                       mbsize.d_view(m).dx3/2;
      }
    });
  }

  return;
}
