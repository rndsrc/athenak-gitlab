//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file imex.cpp
//  \brief implementation of the implicit part of RK-IMEX methods

#include "imex.hpp"

ImEx::ImEx(MeshBlockPack *pp, ParameterInput *pin):
  pmy_pack(pp),
  Ru1("Ru1",1,1,1,1,1),
  Ru2("Ru2",1,1,1,1,1),
  Ru3("Ru3",1,1,1,1,1)
{
  std::string evolution_t = pin->GetString("time","evolution");
  if (evolution_t != "static") {
    auto integrator = pin->GetOrAddString("time", "integrator", "rk2");
    if (integrator == "rk1") {
      this_imex = method::RK1;
      ceff[0] = 1.;
    } else if (integrator == "rk2") {
      std::cout << "Selected ImEx-RK2" << std::endl;
      this_imex = method::RK2;
      ceff[0] =  0.5;
      ceff[1] =  0.;
      ceff[2] =  1.;
    } else if (integrator == "rk3") {
      std::cout << "Selected ImEx-RK3" << std::endl;
      this_imex = method::RK3;
      ceff[0] =  0.24169426078821;
      ceff[1] =  0.;
      ceff[2] =  1.;
      ceff[3] =  0.5;
    };
  }
}

void ImEx::allocate_storage(int _noff, int _nimplicit){

  nimplicit = _nimplicit;
  noff = _noff;

  int nmb = pmy_pack->nmb_thispack;
  auto &ncells = pmy_pack->mb_cells;
  int ncells1 = ncells.nx1 + 2*(ncells.ng);
  int ncells2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
  int ncells3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;

  switch (this_imex){
    case method::RK1:
  	Kokkos::realloc(Ru1, nmb, nimplicit, ncells3, ncells2, ncells1);
      break;
    case method::RK2:
  	Kokkos::realloc(Ru1, nmb, nimplicit, ncells3, ncells2, ncells1);
  	Kokkos::realloc(Ru2, nmb, nimplicit, ncells3, ncells2, ncells1);
      break;
    case method::RK3:
  	Kokkos::realloc(Ru1, nmb, nimplicit, ncells3, ncells2, ncells1);
  	Kokkos::realloc(Ru2, nmb, nimplicit, ncells3, ncells2, ncells1);
  	Kokkos::realloc(Ru3, nmb, nimplicit, ncells3, ncells2, ncells1);
      break;
  }
}


void ImEx::ApplySourceTermsImplicitPreStage(DvceArray5D<Real> &u, DvceArray5D<Real> &w)
{

  
  switch (this_imex){

//    case method::RK1:
//      ApplySourceTermsImplicitPreStageRK1(u,w);
//      break;

    case method::RK2:
      ApplySourceTermsImplicitPreStageRK2(u,w);
      break;
    case method::RK3:
      ApplySourceTermsImplicitPreStageRK3(u,w);
      break;
  }

//      std::cout << "ImEx: Current stage" << current_stage << std::endl;
};

void ImEx::ApplySourceTermsImplicit(DvceArray5D<Real> &u, DvceArray5D<Real> &w, int stage)
{
  switch (this_imex){

//    case method::RK1:
//      ApplySourceTermsImplicitRK2(u,w,stage);
//      break;
    case method::RK2:
      ApplySourceTermsImplicitRK2(u,w,stage);
      break;
    case method::RK3:
      ApplySourceTermsImplicitRK3(u,w,stage);
      break;
  }

//      std::cout << "ImEx: Current stage" << current_stage << std::endl;
};

void ImEx::ApplySourceTermsImplicitPreStageRK2(DvceArray5D<Real> &u, DvceArray5D<Real> &w)
{

    //switch stages:
    //
    //
      int nmb = pmy_pack->nmb_thispack;
      int nvar = nimplicit;
      auto u0_ = u;

      auto Ru1_ = Ru1;
      auto Ru2_ = Ru2;

      double const alphaI = 0.5;

      double dtI = (pmy_pack->pmesh->dt); 

      auto ncells = pmy_pack->mb_cells;
      int ng = ncells.ng;
      int n1 = ncells.nx1 + 2*ng;
      int n2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*ng) : 1;
      int n3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*ng) : 1;

      int noff_ = noff;
    
      current_stage = 0;

      ImplicitEquation(u,w,alphaI*dtI,Ru1);

      par_for("implicit_stage2", DevExeSpace(), 0, (nmb-1),0, nvar-1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
	  KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	  {
	    u0_(m,n+noff_,k,j,i) = u0_(m,n+noff_,k,j,i) - dtI * Ru1_(m,n,k,j,i);
	  });

      current_stage=1;
      ImplicitEquation(u,w,alphaI*dtI,Ru2);


};

void ImEx::ApplySourceTermsImplicitRK2(DvceArray5D<Real> &u, DvceArray5D<Real> &w, int stage)
{

    //switch stages:
    //
    //
      int nmb = pmy_pack->nmb_thispack;
      auto u0_ = u;

      int noff_ = noff;

      auto Ru1_ = Ru1;
      auto Ru2_ = Ru2;
      auto Ru3_ = Ru3;

      double const alphaI = 0.5;

      double dtI = (pmy_pack->pmesh->dt); 

      auto ncells = pmy_pack->mb_cells;
      int ng = ncells.ng;
      int n1 = ncells.nx1 + 2*ng;
      int n2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*ng) : 1;
      int n3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*ng) : 1;

      // Increment internal counter

    switch(stage){

      case 1:
      	  current_stage=2;
	  par_for("implicit_stage3", DevExeSpace(), 0, (nmb-1),0, nimplicit-1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
	    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	    {
	      u0_(m,n+noff_,k,j,i) = u0_(m,n+noff_,k,j,i) + alphaI * dtI * Ru1_(m,n,k,j,i);
	    });
	ImplicitEquation(u,w,alphaI*dtI, Ru1);
	break;

      case 2:
      	  current_stage=3;
	  par_for("implicit_stage4", DevExeSpace(), 0, (nmb-1),0, nimplicit-1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
	    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	    {
	      u0_(m,n+noff_,k,j,i) = u0_(m,n+noff_,k,j,i) + 0.25 * dtI * ( Ru1_(m,n,k,j,i) + Ru2_(m,n,k,j,i));
	    });

	ImplicitEquation(u,w,0.,Ru1);
	break;

    };
};

void ImEx::ApplySourceTermsImplicitPreStageRK3(DvceArray5D<Real> &u, DvceArray5D<Real> &w)
{

    //switch stages:
    //
    //
      int nmb = pmy_pack->nmb_thispack;
      int nvar = nimplicit;
      auto u0_ = u;

      auto Ru1_ = Ru1;
      auto Ru2_ = Ru2;

      double const alphaI = 0.24169426078821; // 1./3.;

      double dtI = (pmy_pack->pmesh->dt); 

      auto ncells = pmy_pack->mb_cells;
      int ng = ncells.ng;
      int n1 = ncells.nx1 + 2*ng;
      int n2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*ng) : 1;
      int n3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*ng) : 1;

    
      int noff_ = noff;
    
      current_stage = 0;

      ImplicitEquation(u,w,alphaI*dtI,Ru1);

	par_for("implicit_stage2", DevExeSpace(), 0, (nmb-1),0, nvar-1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
	  KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	  {
	    u0_(m,n+noff_,k,j,i) = u0_(m,n+noff_,k,j,i) -2.*alphaI*dtI * Ru1_(m,n,k,j,i);
	  });

      current_stage=1;
      ImplicitEquation(u,w,alphaI*dtI,Ru2);
};

void ImEx::ApplySourceTermsImplicitRK3(DvceArray5D<Real> &u, DvceArray5D<Real> &w, int stage)
{

    //switch stages:
    //
    //
      int nmb = pmy_pack->nmb_thispack;
      auto u0_ = u;

      int noff_ = noff;

      auto Ru1_ = Ru1;
      auto Ru2_ = Ru2;
      auto Ru3_ = Ru3;

      double const alphaI = 0.24169426078821; // 1./3.;
      double const betaI = 0.06042356519705;
      double const etaI = 0.12915286960590;

      double dtI = (pmy_pack->pmesh->dt); 

      auto ncells = pmy_pack->mb_cells;
      int ng = ncells.ng;
      int n1 = ncells.nx1 + 2*ng;
      int n2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*ng) : 1;
      int n3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*ng) : 1;

    
      // Increment internal counter

    switch(stage){

      case 1:
          current_stage=2;
	  par_for("implicit_stage3", DevExeSpace(), 0, (nmb-1),0, nimplicit-1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
	    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	    {
	      u0_(m,n+noff_,k,j,i) = u0_(m,n+noff_,k,j,i) + 
	      				  (1.-2.*alphaI) * dtI * Ru2_(m,n,k,j,i) + alphaI*dtI* Ru1_(m,n,k,j,i);
	    });
	ImplicitEquation(u,w,alphaI*dtI, Ru3);
	break;

      case 2:
          current_stage=3;
	  par_for("implicit_stage4a", DevExeSpace(), 0, (nmb-1),0, nimplicit-1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
	    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	    {
	      u0_(m,n+noff_,k,j,i) = u0_(m,n+noff_,k,j,i) + 
		betaI*dtI * Ru1_(m,n,k,j,i) + (etaI- 0.25*(1.-alphaI)) * dtI * Ru2_(m,n,k,j,i) 
		+ (0.5 - betaI - etaI - 1.25*alphaI)*dtI* Ru3_(m,n,k,j,i);
	    });

	  par_for("implicit_stage4b", DevExeSpace(), 0, (nmb-1),0, nimplicit-1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
	    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	    {
	      Ru2_(m,n,k,j,i) = - (2./3.) *betaI*dtI * Ru1_(m,n,k,j,i) + ((1.-4.*etaI)/6.) * dtI * Ru2_(m,n,k,j,i);
	    });

	ImplicitEquation(u,w,alphaI*dtI,Ru1);
	break;

      case 3:
          current_stage=4;
	  par_for("implicit_stage5", DevExeSpace(), 0, (nmb-1),0, nimplicit-1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
	    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
	    {
	      u0_(m,n+noff_,k,j,i) = u0_(m,n+noff_,k,j,i) 
	      			+ Ru2_(m,n,k,j,i) + (-1.0 + 4.*(betaI + etaI +alphaI))/6.*dtI* Ru3_(m,n,k,j,i)
	                                      + (2./3.)*(1.-alphaI) *dtI * Ru1_(m,n,k,j,i);
	    });
	ImplicitEquation(u,w,0.,Ru1);
	break;

    };
};
