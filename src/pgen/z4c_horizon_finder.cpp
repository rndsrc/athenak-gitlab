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


DualArray1D<Real> SurfaceNullExpansion(MeshBlockPack *pmbp, Strahlkorper *S, DualArray6D<Real> dg_ddd) {
  // Load adm variables
  auto &adm = pmbp->padm->adm;
  auto &g_dd = adm.g_dd;
  auto &K_dd = adm.K_dd;

  // Strahlkorper *S = nullptr;
  // S = new Strahlkorper(pmbp, nlev, 2,25);
  // Real ctr[3] = {0,0,0};
  // DualArray1D<Real> rad_tmp;
  int nangles = S->nangles;
  auto surface_jacobian = S->surface_jacobian;
  auto d_surface_jacobian = S->d_surface_jacobian;

  // Kokkos::realloc(rad_tmp,nangles);

  // Container for surface tensors
  DualArray2D<Real> g_dd_surf;
  DualArray2D<Real> K_dd_surf;

  // Athena Tensor structure cannot easily adapt to the Strahlkorper Class. 
  // For now using DualArrays for Tensors and keeping track of the indices.
  // HostArray3D<Real> surf_tensors;
  // AthenaHostTensor<Real,TensorSymm::SYM2, 1, 2> g_dd_surf2;
  // AthenaHostTensor<Real,TensorSymm::SYM2, 1, 2> K_dd_surf2;
  // Kokkos::realloc(surf_tensors,    nmb, (N_Z4c), ncells3, ncells2, ncells1);

  Kokkos::realloc(g_dd_surf,nangles,6); // xx, xy, xz, yy, yz, zz
  Kokkos::realloc(K_dd_surf,nangles,6);

  // Interpolate g_dd and K_dd onto the surface
  g_dd_surf =  S->InterpolateToSphere(g_dd);
  K_dd_surf =  S->InterpolateToSphere(K_dd);


  // Evaluate Derivatives of F = r - h(theta,phi) in spherical components
  DualArray2D<Real> dF_d_surf;
  Kokkos::realloc(dF_d_surf,nangles,3);

  DualArray1D<Real> place_holder_for_partial_theta;
  DualArray1D<Real> place_holder_for_partial_phi;
  Kokkos::realloc(place_holder_for_partial_theta,nangles);
  Kokkos::realloc(place_holder_for_partial_phi,nangles);

  place_holder_for_partial_theta = S->ThetaDerivative(S->pointwise_radius);
  place_holder_for_partial_phi = S->PhiDerivative(S->pointwise_radius);

  for(int n=0; n<nangles; ++n) {
    // radial derivatives
    dF_d_surf.h_view(n,0) = 1;
    // theta and phi derivatives
    dF_d_surf.h_view(n,1) = place_holder_for_partial_theta.h_view(n);
    dF_d_surf.h_view(n,2) = place_holder_for_partial_phi.h_view(n);
  }

  // Evaluate Second Derivatives of F in spherical components
  DualArray2D<Real> ddF_dd_surf;
  Kokkos::realloc(ddF_dd_surf,nangles,6); // rr, rt, rp, tt, tp, pp

  DualArray1D<Real> place_holder_for_second_partials;
  Kokkos::realloc(place_holder_for_second_partials,nangles);

  // all second derivatives w.r.t. r vanishes as dr_F = 1
  for(int n=0; n<nangles; ++n) {
    ddF_dd_surf.h_view(n,0) = 0;
    ddF_dd_surf.h_view(n,1) = 0;
    ddF_dd_surf.h_view(n,2) = 0;
  }
  // tt
  place_holder_for_second_partials = S->ThetaDerivative(place_holder_for_partial_theta);
  for(int n=0; n<nangles; ++n) {
    ddF_dd_surf.h_view(n,3) = place_holder_for_second_partials.h_view(n);
  }
  // tp
  place_holder_for_second_partials = S->PhiDerivative(place_holder_for_partial_theta);
  for(int n=0; n<nangles; ++n) {
    ddF_dd_surf.h_view(n,4) = place_holder_for_second_partials.h_view(n);
  }
  // pp
  place_holder_for_second_partials = S->PhiDerivative(place_holder_for_partial_phi);
  for(int n=0; n<nangles; ++n) {
    ddF_dd_surf.h_view(n,5) = place_holder_for_second_partials.h_view(n);
  }
  // Calculating g_uu on the sphere
  DualArray2D<Real> g_uu_surf;
  Kokkos::realloc(g_uu_surf,nangles,6); // xx, xy, xz, yy, yz, zz
  for(int n=0; n<nangles; ++n) {
    Real detg = SpatialDet(g_dd_surf.h_view(n,0), g_dd_surf.h_view(n,1), g_dd_surf.h_view(n,2),
                           g_dd_surf.h_view(n,3), g_dd_surf.h_view(n,4), g_dd_surf.h_view(n,5));
    SpatialInv(1.0/detg,
            g_dd_surf.h_view(n,0), g_dd_surf.h_view(n,1), g_dd_surf.h_view(n,2),
            g_dd_surf.h_view(n,3), g_dd_surf.h_view(n,4), g_dd_surf.h_view(n,5),
            &g_uu_surf.h_view(n,0), &g_uu_surf.h_view(n,1), &g_uu_surf.h_view(n,2),
            &g_uu_surf.h_view(n,3), &g_uu_surf.h_view(n,4), &g_uu_surf.h_view(n,5));
  }

  // Covariant derivatives of F in cartesian basis
  DualArray2D<Real> delta_F_d_surf;
  Kokkos::realloc(delta_F_d_surf,nangles,3);
  for(int n=0; n<nangles; ++n) {
    for(int i=0; i<3;++i){
      delta_F_d_surf.h_view(n,i) = 0;
      for(int u=0; u<3;++u) {
        delta_F_d_surf.h_view(n,i) += surface_jacobian.h_view(n,u,i)*dF_d_surf.h_view(n,u);
      }
    }
  }

  // interpolate dg_ddd to surface
  auto dg_ddd_surf = S->InterpolateToSphere(dg_ddd);

  // Place holder for Christoffel symbols of the second kind interpolated to the surface
  DualArray3D<Real> Gamma_udd_surf;
  Kokkos::realloc(Gamma_udd_surf,nangles,3,6);
  for(int n=0; n<nangles; ++n) {
    for(int i=0; i<3; ++i)
    for(int j=0; i<3; ++i)
    for(int k=0; i<3; ++i) {
      Gamma_udd_surf.h_view(n,i,SYMM2_Ind(j,k)) = 0;
      for(int s=0; i<3; ++i) {
        Gamma_udd_surf.h_view(n,i,SYMM2_Ind(j,k)) += 0.5*g_uu_surf.h_view(n,SYMM2_Ind(i,s)) * (dg_ddd_surf.h_view(n,j,SYMM2_Ind(k,s))
                                                    +dg_ddd_surf.h_view(n,k,SYMM2_Ind(s,j))-dg_ddd_surf.h_view(n,s,SYMM2_Ind(j,k)));
      }
    }
  }

  // Second Covariant derivatives of F in cartesian basis
  DualArray3D<Real> deltadelta_F_dd_surf;
  Kokkos::realloc(deltadelta_F_dd_surf,nangles,3,3);
  for(int n=0; n<nangles; ++n) {
    for(int i=0; i<3;++i) {
      for(int j=0; j<3;++j) {
        deltadelta_F_dd_surf.h_view(n,i,j) = 0;
        for(int v=0; v<3;++v) {
          deltadelta_F_dd_surf.h_view(n,i,j) += d_surface_jacobian.h_view(n,i,v,j)*dF_d_surf.h_view(n,v);
          deltadelta_F_dd_surf.h_view(n,i,j) += -Gamma_udd_surf.h_view(n,v,SYMM2_Ind(i,j))*delta_F_d_surf.h_view(n,v); 
          for(int u=0; u<3;++u) {
            deltadelta_F_dd_surf.h_view(n,i,j) += surface_jacobian.h_view(n,v,j)*surface_jacobian.h_view(n,u,i)
                                                  *ddF_dd_surf.h_view(n,SYMM2_Ind(u,v));
          }
        }
      }
    }
  }

  // These are all the infrastructure needed for evaluating the null expansion!

  // auxiliary variable delta_F_abs, Gundlach 1997 eqn 8 
  DualArray1D<Real> delta_F_abs;
  Kokkos::realloc(delta_F_abs,nangles);
  for(int n=0; n<nangles; ++n) {
    Real delta_F_sqr = 0;
    for(int i=0; i<3;++i) {
      for(int j=0; j<3;++j) {
        delta_F_sqr += g_uu_surf.h_view(n,SYMM2_Ind(i,j))*delta_F_d_surf.h_view(n,i)
                                *delta_F_d_surf.h_view(n,j);
      }
    }
    delta_F_abs.h_view(n) = sqrt(delta_F_sqr);
  }

  // contravariant form of delta_F
  DualArray2D<Real> delta_F_u_surf;
  Kokkos::realloc(delta_F_u_surf,nangles,3);
  for(int n=0; n<nangles; ++n) {
    for(int i=0; i<3;++i) {
      delta_F_u_surf.h_view(n,i) = 0;
      for(int j=0; j<3;++j) {
        delta_F_u_surf.h_view(n,i) += g_uu_surf.h_view(n,SYMM2_Ind(i,j))*delta_F_d_surf.h_view(n,j);
      }
    }
  }


  // Surface inverse metric, Gundlach 1997 eqn 9
  DualArray2D<Real> m_uu_surf;
  Kokkos::realloc(m_uu_surf,nangles,6);
  for(int n=0; n<nangles; ++n) {
    for(int i=0; i<3;++i) {
      for(int j=0; j<3;++j) {
        m_uu_surf.h_view(n,SYMM2_Ind(i,j)) = g_uu_surf.h_view(n,SYMM2_Ind(i,j))
                                        - delta_F_u_surf.h_view(n,i)*delta_F_u_surf.h_view(n,j)
                                        /delta_F_abs.h_view(n)/delta_F_abs.h_view(n);
      }
    }
  }

  // Surface Null Expansion, Gundlach 1997 eqn 9
  DualArray1D<Real> H;
  Kokkos::realloc(H,nangles);
  for(int n=0; n<nangles; ++n) {
    H.h_view(n) = 0;
    for(int i=0; i<3;++i) {
      for(int j=0; j<3;++j) {
        H.h_view(n) += m_uu_surf.h_view(n,SYMM2_Ind(i,j))*(deltadelta_F_dd_surf.h_view(n,i,j)
                        /delta_F_abs.h_view(n)-K_dd_surf.h_view(n,SYMM2_Ind(i,j)));
      }
    }
  }
  return H;
}

DualArray1D<Real> AnalyticSurfaceNullExpansion(Strahlkorper *S) {
  int nangles = S->nangles;
  DualArray1D<Real> H;
  Kokkos::realloc(H,nangles);
  auto r = S->pointwise_radius;
  for(int n=0; n<nangles; ++n) {
    Real denominator = 2*r.h_view(n)+1;
    H.h_view(n) = 8*r.h_view(n)*(2*r.h_view(n)-1)/(denominator*denominator*denominator);
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

  int nlev = 10;
  bool rotate_sphere = true;
  bool fluxes = true;

  Strahlkorper *S = nullptr;
  S = new Strahlkorper(pmbp, nlev, 18,25);
  
  // Surface Null Expansion, Gundlach 1997 eqn 9
  DualArray1D<Real> H = SurfaceNullExpansion(pmbp,S,dg_ddd);
  // DualArray1D<Real> H = AnalyticSurfaceNullExpansion(S);
  Real H_integrated = S->Integrate(H);
  std::cout << "Surface Null Expansion: " << H_integrated << std::endl;

  return;
}



