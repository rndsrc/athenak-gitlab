//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_one_puncture.cpp
//  \brief Problem generator for finding the horizon for a single puncture placed at the origin of the domain
//

#include <algorithm>
#include <cmath>
#include <sstream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "athena_tensor.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "adm/adm.hpp"
#include "geodesic-grid/strahlkorper.hpp"
#include "coordinates/cell_locations.hpp"
#include "utils/finite_diff.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for Testing Horizon Finder

int SYMM2_Ind(int v1, int v2) {
  if (v1==0) {
    return v2;
  } else if (v1==1) {
    if (v2==0) {
      return 1;
    } else {
      return v2+2;
    }
  } else {
    if (v2==0) {
      return 2;
    } else {
      return v2+3;
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void DualArray6D<Real> metric_partial(MeshBlockPack *pmbp)
// \brief Compute derivative of g_ij
//
// This sets the d_g_kij everywhere in the MeshBlock
//----------------------------------------------------------------------------------------

template <int NGHOST>
DualArray6D<Real> metric_partial(MeshBlockPack *pmbp) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int &nghost = indcs.ng;

  //For GLOOPS
  int nmb = pmbp->nmb_thispack;

  // Initialize dg_ddd container
  int ncells1 = indcs.nx1 + 2*nghost;
  int ncells2 = indcs.nx2 + 2*nghost;
  int ncells3 = indcs.nx3 + 2*nghost;
  DualArray6D<Real> dg_ddd_full;
  Kokkos::realloc(dg_ddd_full,nmb,3,6,ncells3,ncells2,ncells1);

  auto &adm = pmbp->padm->adm;
  int scr_level = 1;
  // 2 1D scratch array and 1 2D scratch array
  size_t scr_size = ScrArray2D<Real>::shmem_size(0,0); // 3D tensor with symm
  par_for_outer("ADM constraints loop",DevExeSpace(),scr_size,scr_level,0,nmb-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {

    Real idx[] = {size.d_view(m).idx1, size.d_view(m).idx2, size.d_view(m).idx3};
    // -----------------------------------------------------------------------------------
    // derivatives
    //
    // first derivatives of g
    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        dg_ddd_full.d_view(m,c,SYMM2_Ind(a,b),k,j,i) = Dx<NGHOST>(c, idx, adm.g_dd, m,a,b,k,j,i);
      });
    }
  });

  // sync to host
  dg_ddd_full.template modify<DevExeSpace>();
  dg_ddd_full.template sync<HostMemSpace>();
  return dg_ddd_full;
}

template DualArray6D<Real> metric_partial<2>(MeshBlockPack *pmbp);
template DualArray6D<Real> metric_partial<3>(MeshBlockPack *pmbp);
template DualArray6D<Real> metric_partial<4>(MeshBlockPack *pmbp);

AthenaSurfaceTensor<Real,TensorSymm::NONE,3,0> SurfaceNullExpansion(MeshBlockPack *pmbp, Strahlkorper *S, DualArray6D<Real> dg_ddd) {
  // Load adm variables
  auto &adm = pmbp->padm->adm;
  auto &g_dd = adm.g_dd;
  auto &K_dd = adm.K_dd;

  int nangles = S->nangles;
  auto surface_jacobian = S->surface_jacobian;
  auto d_surface_jacobian = S->d_surface_jacobian;

  // *****************  Step 4 of Schnetter 2002  *******************

  // Interpolate g_dd, K_dd, and dg_ddd onto the surface
  // Still need to check the tensor interpolator!

  std::cout << "Here 2" << std::endl;

  auto g_dd_surf =  S->InterpolateToSphere(g_dd);
  auto K_dd_surf =  S->InterpolateToSphere(K_dd);
  auto dg_ddd_surf = S->InterpolateToSphere(dg_ddd); // change this into the new surface tensor notation
  std::cout << "Here 3" << std::endl;


  // Calculating g_uu on the sphere
  AthenaSurfaceTensor<Real,TensorSymm::SYM2,3,2> g_uu_surf;
  g_uu_surf.NewAthenaSurfaceTensor(nangles);

  for(int n=0; n<nangles; ++n) {
    Real detg = SpatialDet(g_dd_surf(0,0,n), g_dd_surf(0,1,n), g_dd_surf(0,2,n),
                           g_dd_surf(1,1,n), g_dd_surf(1,2,n), g_dd_surf(2,2,n));
    SpatialInv(1.0/detg,
            g_dd_surf(0,0,n), g_dd_surf(0,1,n), g_dd_surf(0,2,n),
            g_dd_surf(1,1,n), g_dd_surf(1,2,n), g_dd_surf(2,2,n),
            &g_uu_surf(0,0,n), &g_uu_surf(0,1,n), &g_uu_surf(0,2,n),
            &g_uu_surf(1,1,n), &g_uu_surf(1,2,n), &g_uu_surf(2,2,n));
  }

  // Christoffel symbols of the second kind on the surface, saved as rank3 tensor
  AthenaSurfaceTensor<Real,TensorSymm::SYM2,3,3> Gamma_udd_surf;
  Gamma_udd_surf.NewAthenaSurfaceTensor(nangles);

  for(int n=0; n<nangles; ++n) {
    for(int i=0; i<3; ++i)
    for(int j=0; i<3; ++i)
    for(int k=j; i<3; ++i) { // symmetric in j and k
      Gamma_udd_surf(i,j,k,n) = 0;
      for(int s=0; i<3; ++i) {
        Gamma_udd_surf(i,j,k,n) += 0.5*g_uu_surf(i,s,n) * (dg_ddd_surf(j,k,s,n)
                                  +dg_ddd_surf(k,s,j,n)-dg_ddd_surf(s,j,k,n));
      }
    }
  }

  // *****************  Step 6 of Schnetter 2002  *******************
  // Evaluate Derivatives of F = r - h(theta,phi)
  // First in spherical components _sb stands for spherical basis
  AthenaSurfaceTensor<Real,TensorSymm::NONE,3,1> dF_d_surf_sb;
  dF_d_surf_sb.NewAthenaSurfaceTensor(nangles);

  DualArray1D<Real> place_holder_for_partial_theta;
  DualArray1D<Real> place_holder_for_partial_phi;
  Kokkos::realloc(place_holder_for_partial_theta,nangles);
  Kokkos::realloc(place_holder_for_partial_phi,nangles);

  // still need to check on these derivatives on sphere!
  place_holder_for_partial_theta = S->ThetaDerivative(S->pointwise_radius);
  place_holder_for_partial_phi = S->PhiDerivative(S->pointwise_radius);
  for(int n=0; n<nangles; ++n) {
    // radial derivatives
    dF_d_surf_sb(0,n) = 1;
    // theta and phi derivatives
    dF_d_surf_sb(1,n) = place_holder_for_partial_theta.h_view(n);
    dF_d_surf_sb(2,n) = place_holder_for_partial_phi.h_view(n);
  }

  // Evaluate Second Derivatives of F in spherical components
  AthenaSurfaceTensor<Real,TensorSymm::SYM2,3,2> ddF_dd_surf_sb;
  ddF_dd_surf_sb.NewAthenaSurfaceTensor(nangles);

  DualArray1D<Real> place_holder_for_second_partials;
  Kokkos::realloc(place_holder_for_second_partials,nangles);

  // all second derivatives w.r.t. r vanishes as dr_F = 1
  for(int n=0; n<nangles; ++n) {
    ddF_dd_surf_sb(0,0,n) = 0;
    ddF_dd_surf_sb(0,1,n) = 0;
    ddF_dd_surf_sb(0,2,n) = 0;
  }
  // tt
  place_holder_for_second_partials = S->ThetaDerivative(place_holder_for_partial_theta);
  for(int n=0; n<nangles; ++n) {
    ddF_dd_surf_sb(1,1,n) = place_holder_for_second_partials.h_view(n);
  }
  // tp
  place_holder_for_second_partials = S->PhiDerivative(place_holder_for_partial_theta);
  for(int n=0; n<nangles; ++n) {
    ddF_dd_surf_sb(1,2,n) = place_holder_for_second_partials.h_view(n);
  }
  // pp
  place_holder_for_second_partials = S->PhiDerivative(place_holder_for_partial_phi);
  for(int n=0; n<nangles; ++n) {
    ddF_dd_surf_sb(2,2,n) = place_holder_for_second_partials.h_view(n);
  }

  // Convert to derivatives of F to cartesian basis
  AthenaSurfaceTensor<Real,TensorSymm::NONE,3,1> dF_d_surf;
  dF_d_surf.NewAthenaSurfaceTensor(nangles);

  // check the Surface Jacobian
  for(int n=0; n<nangles; ++n) {
    for(int i=0; i<3;++i){
      dF_d_surf(i,n) = 0;
      for(int u=0; u<3;++u) {
        dF_d_surf(i,n) += surface_jacobian.h_view(n,u,i)*dF_d_surf_sb(u,n);
      }
    }
  }

  // Second Covariant derivatives of F in cartesian basis
  AthenaSurfaceTensor<Real,TensorSymm::SYM2,3,2> ddF_dd_surf;
  ddF_dd_surf.NewAthenaSurfaceTensor(nangles);
  for(int n=0; n<nangles; ++n) {
    for(int i=0; i<3;++i) {
      for(int j=0; j<3;++j) {
        ddF_dd_surf(i,j,n) = 0;
        for(int v=0; v<3;++v) {
          ddF_dd_surf(i,j,n) += d_surface_jacobian.h_view(n,i,v,j)*dF_d_surf_sb(v,n);
          ddF_dd_surf(i,j,n) += -Gamma_udd_surf(v,i,j,n)*dF_d_surf(v,n); 
          for(int u=0; u<3;++u) {
            ddF_dd_surf(i,j,n) += surface_jacobian.h_view(n,v,j)*surface_jacobian.h_view(n,u,i)
                                                  *ddF_dd_surf_sb(u,v,n);
          }
        }
      }
    }
  }

  // These are all the infrastructure needed for evaluating the null expansion!

  // auxiliary variable delta_F_abs, Gundlach 1997 eqn 8 
  AthenaSurfaceTensor<Real,TensorSymm::NONE,3,0> delta_F_abs;
  delta_F_abs.NewAthenaSurfaceTensor(nangles);

  // DualArray1D<Real> delta_F_abs;
  // Kokkos::realloc(delta_F_abs,nangles);
  for(int n=0; n<nangles; ++n) {
    Real delta_F_sqr = 0;
    for(int i=0; i<3;++i) {
      for(int j=0; j<3;++j) {
        delta_F_sqr += g_uu_surf(i,j,n)*dF_d_surf(i,n)
                                *dF_d_surf(j,n);
      }
    }
    delta_F_abs(n) = sqrt(delta_F_sqr);
  }

  // contravariant form of delta_F

  AthenaSurfaceTensor<Real,TensorSymm::NONE,3,1> dF_u_surf;
  dF_u_surf.NewAthenaSurfaceTensor(nangles);

  for(int n=0; n<nangles; ++n) {
    for(int i=0; i<3;++i) {
      dF_u_surf(i,n) = 0;
      for(int j=0; j<3;++j) {
        dF_u_surf(i,n) += g_uu_surf(i,j,n)*dF_d_surf(j,n);
      }
    }
  }


  // Surface inverse metric (in cartesian coordinate), Gundlach 1997 eqn 9
  AthenaSurfaceTensor<Real,TensorSymm::SYM2,3,2> m_uu_surf;
  m_uu_surf.NewAthenaSurfaceTensor(nangles);
  // DualArray2D<Real> m_uu_surf;
  // Kokkos::realloc(m_uu_surf,nangles,6);
  for(int n=0; n<nangles; ++n) {
    for(int i=0; i<3;++i) {
      for(int j=0; j<3;++j) {
        m_uu_surf(i,j,n) = g_uu_surf(i,j,n)
                - dF_u_surf(i,n)*dF_u_surf(j,n)
                /delta_F_abs(n)/delta_F_abs(n);
      }
    }
  }

  // Surface Null Expansion, Gundlach 1997 eqn 9
  AthenaSurfaceTensor<Real,TensorSymm::NONE,3,0> H;
  H.NewAthenaSurfaceTensor(nangles);

  for(int n=0; n<nangles; ++n) {
    H(n) = 0;
    for(int i=0; i<3;++i) {
      for(int j=0; j<3;++j) {
        H(n) += m_uu_surf(i,j,n)*(ddF_dd_surf(i,j,n)
                        /delta_F_abs(n)-K_dd_surf(i,j,n));
      }
    }
  }

  return H;
}

// Analytical surface null expansion for Schwarzschild in isotropic coordinate, testing only
AthenaSurfaceTensor<Real,TensorSymm::NONE,3,0> AnalyticSurfaceNullExpansion(Strahlkorper *S) {
  int nangles = S->nangles;
  AthenaSurfaceTensor<Real,TensorSymm::NONE,3,0> H;
  H.NewAthenaSurfaceTensor(nangles);

  auto r = S->pointwise_radius;
  for(int n=0; n<nangles; ++n) {
    Real denominator = 2*r.h_view(n)+1;
    H(n) = 8*r.h_view(n)*(2*r.h_view(n)-1)/(denominator*denominator*denominator);
  }
  return H;
}

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "One Puncture test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // One Puncture nitial data 
  pmbp->pz4c->ADMOnePuncture(pmbp, pin);
  pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  std::cout<<"OnePuncture initialized; Starting Horizon Finder"<<std::endl;

  // load in adm variables
  auto &adm = pmbp->padm->adm;
  auto &z4c = pmbp->pz4c->z4c;
  auto &g_dd = adm.g_dd;
  auto &K_dd = adm.K_dd;
  // Evaluate partial derivatives of the metric over the entire domain
  // 6 dimensional array, nmb, 3, 6, ncells3, ncells2, ncells1
  //DualArray6D<Real> *dg_ddd = nullptr;
  auto dg_ddd = metric_partial<2>(pmbp);

  // Initialize a surface with radius of 2 centered at the origin

  int nlev = 20;
  int nfilt = 16;
  bool rotate_sphere = true;
  bool fluxes = true;
  Real radius = 1;
  Strahlkorper *S = nullptr;
  S = new Strahlkorper(pmbp, nlev, radius,nfilt);
  Real ctr[3] = {0.,0.,0.};
  // Surface Null Expansion, Gundlach 1997 eqn 9
  // DualArray1D<Real> H = SurfaceNullExpansion(pmbp,S,dg_ddd);
  std::cout << "Here" << std::endl;
  AthenaSurfaceTensor<Real,TensorSymm::NONE,3,0> H = SurfaceNullExpansion(pmbp, S, dg_ddd);
  std::cout << "Here 2" << std::endl;

  // Template this integration function over DualArray1D and Tensor of rank 0
  Real H_integrated = S->Integrate(H);
  std::cout << "Initial Norm of H: " << H_integrated << std::endl;

  // H-flow loop, take A = B = 1, rho = 1
  auto H_spectral = S->SpatialToSpectral(H);
  for (int itr=0; itr<0; ++itr) {
    // auto pointwise_radius = S->pointwise_radius;
    auto r_spectral = S->SpatialToSpectral(S->pointwise_radius);

    DualArray1D<Real> r_spectral_np1;
    Kokkos::realloc(r_spectral_np1,nfilt);

    for (int i=0; i<nfilt; ++i) {
      int l = (int) sqrt(i);
      r_spectral_np1.h_view(i) =  r_spectral.h_view(i) - 1/(1+10*l*(l+1))*H_spectral.h_view(i);
      std::cout << r_spectral_np1.h_view(i) << std::endl;
    }

    auto r_np1 = S->SpectralToSpatial(r_spectral_np1);

    // reset radius
    S->SetPointwiseRadius(r_np1,&ctr[3]);

    // reevaluate H
    AthenaSurfaceTensor<Real,TensorSymm::NONE,3,0> H = SurfaceNullExpansion(pmbp, S, dg_ddd);
    H_integrated = S->Integrate(H);

    std::cout << "Itr " << itr+1 << "   Norm of H: " << H_integrated << std::endl;
  }
  


  //for (int i=0;i<nfilt;++i) {
  //  std::cout << "Surface Null Expansion: " << H_spectral.h_view(i) << std::endl;
  //}
  // DualArray1D<Real> H = AnalyticSurfaceNullExpansion(S);



  // still need to write out the loop for fast flow.
  return;
}