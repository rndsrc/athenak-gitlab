//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_grhyd.cpp
//  \brief derived class that implements ideal gas EOS in general relativistic hydro

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos.hpp"

#include "eos_framework/Hybrid/hybrid_implementation.hh"
#include "eos_framework/Cold/cold_pwpoly_implementation.hh"
#include "eos_framework/Cold/cold_table_implementation.hh"

#include "eos_framework/Cold/hot_slice.hh"

//#include "eos_framework/3D_Table/tabulated_implementation.hh"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"



//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

TabulatedGRHydro::TabulatedGRHydro(MeshBlockPack *pp, ParameterInput *pin)
  : EquationOfState(pp, pin)
{
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("eos","gamma");
  eos_data.iso_cs = 0.0;

  if constexpr (std::is_same<AthenaEOS, EOS_Polytropic>::value){

    // Only used for testing

    EOS_Polytropic::gamma_th_m1 = eos_data.gamma -1.;
    Cold_PWPoly::num_pieces = 1.;
    Cold_PWPoly::k_tab[0] =  1.e-8;
    Cold_PWPoly::rho_tab[0] =0.;
    Cold_PWPoly::eps_tab[0] =0.;
    Cold_PWPoly::P_tab[0] =0.;
    Cold_PWPoly::gamma_tab[0] =eos_data.gamma;
  }

  if constexpr (std::is_same<AthenaEOS, EOS_Tabulated>::value){

    auto eos_type = pin->GetString("eos","eos_type");
    auto eos_filename = pin->GetString("eos","eos_filename").c_str();

    if (eos_type ==  "stellarcollapse"){
      EOS_Tabulated::readtable_scollapse(eos_filename, true, true);
    }else if (eos_type == "compose"){
      EOS_Tabulated::readtable_compose(eos_filename);
    }else{
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
		<< "<eos>/eos_type = '" << eos_type << "' not implemented" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    EOS_Tabulated::extend_table_high = true;

    // Compute Isentropic or Cold EOS for ID


    // The following could be included in an input file, but the parameters
    // usually don't change much between EOS

    auto id_eos_type = pin->GetString("eos","id_eos_type");

    int downsample_num_points = 500;
    int cold_table_num_points = 300;

    if (id_eos_type == "constant_temperature" ) {
      auto temp_id = pin->GetReal("eos","initial_temperature");
      HotSlice_beta_eq(downsample_num_points, cold_table_num_points, temp_id);
    } else if (id_eos_type == "constant_entropy" ) {
      auto entropy_id = pin->GetReal("eos","initial_entropy");
      HotSlice_beta_eq_isentropic(downsample_num_points, cold_table_num_points,
				  entropy_id);
    }else{
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
		<< "<eos>/id_eos_type = '" << id_eos_type << "' not implemented" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn Real EquationC22()
// \brief Inline function to compute function f(z) defined in eq. C22 of Galeazzi et al.
// The ConsToPrim algorithms finds the root of this function f(z)=0

KOKKOS_INLINE_FUNCTION
Real EquationC22(Real z, Real &u_d, Real q, Real r, Real gm1, Real pfloor)
{
  Real const w = sqrt(1.0 + z*z);         // (C15)
  Real wd = u_d/w;                  // (C15)
  Real eps = w*q - z*r + (z*z)/(1.0 + w); // (C16)

  typename AthenaEOS::error_type error;
  double ye = 0.0;
  double temp;
  double w_p= AthenaEOS::press_temp__eps_rho_ye(temp,eps,wd,ye,error);
  Real h = 1. + eps + w_p/wd;

  return (z - r/h); // (C22)
}

//----------------------------------------------------------------------------------------
// \!fn void ConsToPrim()
// \brief Converts conserved into primitive variables.
// Operates over entire MeshBlock, including ghost cells.

void TabulatedGRHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto coord = pmy_pack->coord.coord_data;
  Real gm1 = eos_data.gamma - 1.0;
  Real gamma_prime = eos_data.gamma/gm1;
  Real &pfloor_ = eos_data.pressure_floor;
  Real &dfloor_ = eos_data.density_floor;

  // Parameters
  int const max_iterations = 25;
  Real const tol = 1.0e-12;
  Real const v_sq_max = 1.0 - tol;

  par_for("hyd_con2prim", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real& u_d  = cons(m,IDN,k,j,i);
      Real& u_e  = cons(m,IEN,k,j,i);
      Real& u_m1 = cons(m,IM1,k,j,i);
      Real& u_m2 = cons(m,IM2,k,j,i);
      Real& u_m3 = cons(m,IM3,k,j,i);

      Real& w_d  = prim(m,IDN,k,j,i);
      Real& w_p  = prim(m,IPR,k,j,i);
      Real& w_vx = prim(m,IVX,k,j,i);
      Real& w_vy = prim(m,IVY,k,j,i);
      Real& w_vz = prim(m,IVZ,k,j,i);

      Real& w_temp = prim(m, ITEMP,k,j,i);

      Real &x1min = coord.mb_size.d_view(m).x1min;
      Real &x1max = coord.mb_size.d_view(m).x1max;
      int nx1 = coord.mb_indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = coord.mb_size.d_view(m).x2min;
      Real &x2max = coord.mb_size.d_view(m).x2max;
      int nx2 = coord.mb_indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = coord.mb_size.d_view(m).x3min;
      Real &x3max = coord.mb_size.d_view(m).x3max;
      int nx3 = coord.mb_indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));

      bool floor_hit = false;

      // Extract components of metric
      Real g_[NMETRIC], gi_[NMETRIC];
      ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, false,
                              coord.bh_spin, g_, gi_);

      // Only execute cons2prim if outside excised region
      if (rad > coord.bh_rmin) {
        // We are evolving T^t_t, but the SR C2P algorithm is only consistent
        // with alpha^2 T^{tt}
        // compute T^{tt} = g^0\mu T^t_\mu
        Real ue_tmp = gi_[I00]*u_e + gi_[I01]*u_m1 + gi_[I02]*u_m2 + gi_[I03]*u_m3;

        // This is only true if sqrt{-g}=1!
        ue_tmp *= (-1./gi_[I00]);  // Multiply by alpha^2

        // Need to multiply the conserved density by alpha, so that it
        // contains a lorentz factor
        Real alpha = sqrt(-1.0/gi_[I00]);
        Real ud_tmp = u_d*alpha;

        // Subtract density for consistency with the rest of the algorithm
        ue_tmp -= ud_tmp;

        // Need to treat the conserved momenta. Also they lack an alpha
        // This is only true if sqrt{-g}=1!
        Real um1_tmp = u_m1*alpha;
        Real um2_tmp = u_m2*alpha;
        Real um3_tmp = u_m3*alpha;

        // apply density floor, without changing momentum or energy
        if (ud_tmp < dfloor_) {
          ud_tmp = dfloor_;
          floor_hit = true;
        }

        // apply energy floor
        if (ue_tmp < pfloor_/gm1) {
          ue_tmp = pfloor_/gm1;
          floor_hit = true;
        }

        // Recast all variables (eq C2)
        // Need to raise indices on u_m1, which transforms using the spatial
        // 3-metric.  This is slightly more involved
        //
        // Gourghoulon says: g^ij = gamma^ij - beta^i beta^j/alpha^2
        //       g^0i = beta^i/alpha^2
        //       g^00 = -1/ alpha^2
        // Hence gamma^ij =  g^ij - g^0i g^0j/g^00
        Real m1u = ((gi_[I11] - gi_[I01]*gi_[I01]/gi_[I00])*um1_tmp +
                    (gi_[I12] - gi_[I01]*gi_[I02]/gi_[I00])*um2_tmp +
                    (gi_[I13] - gi_[I01]*gi_[I03]/gi_[I00])*um3_tmp);  // (C26)

        Real m2u = ((gi_[I12] - gi_[I01]*gi_[I02]/gi_[I00])*um1_tmp +
                    (gi_[I22] - gi_[I02]*gi_[I02]/gi_[I00])*um2_tmp +
                    (gi_[I23] - gi_[I02]*gi_[I03]/gi_[I00])*um3_tmp);  // (C26)

        Real m3u = ((gi_[I13] - gi_[I01]*gi_[I03]/gi_[I00])*um1_tmp +
                    (gi_[I23] - gi_[I02]*gi_[I03]/gi_[I00])*um2_tmp +
                    (gi_[I33] - gi_[I03]*gi_[I03]/gi_[I00])*um3_tmp);  // (C26)

        Real q = ue_tmp/ud_tmp;
        Real r = sqrt(um1_tmp*m1u + um2_tmp*m2u + um3_tmp*m3u)/ud_tmp;

        // Enforce lower velocity bound (eq. C13). This bound combined with a floor on
        // the value of p will guarantee "some" result of the inversion
        Real kk = fmin(2.* sqrt(v_sq_max)/(1.0 + v_sq_max) - tol, r/(1.+q));

        // Compute bracket (C23)
        auto zm = 0.5*kk/sqrt(1.0 - 0.25*kk*kk);
        auto zp = kk/sqrt(1.0 - kk*kk);

        // Evaluate master function (eq C22) at bracket values
        Real fm = EquationC22(zm, ud_tmp, q, r, gm1, pfloor_);
        Real fp = EquationC22(zp, ud_tmp, q, r, gm1, pfloor_);

        // For simplicity on the GPU, find roots using the false position method
        int iterations = max_iterations;
        // If bracket within tolerances, don't bother doing any iterations
        if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
          iterations = -1;
        }
        Real z = 0.5*(zm + zp);

        for (int ii=0; ii < iterations; ++ii) {
          z =  (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
          Real f = EquationC22(z, ud_tmp, q, r, gm1, pfloor_);

          // Quit if convergence reached
          // NOTE: both z and f are of order unity
          if ((fabs(zm-zp) < tol ) || (fabs(f) < tol )) {
            break;
          }

          if (f*fp < 0.0) {  // assign zm-->zp if root bracketed by [z,zp]
             zm = zp;
             fm = fp;
             zp = z;
             fp = f;
          } else {  // assign zp-->z if root bracketed by [zm,z]
             fm = 0.5*fm;  // 1/2 comes from "Illinois algorithm" to accelerate convergence
             zp = z;
             fp = f;
          }
        }

        // iterations ended, compute primitives from resulting value of z
        Real const w = sqrt(1.0 + z*z);  // (C15)
        w_d = ud_tmp/w;  // (C15)
        Real eps = w*q - z*r + (z*z)/(1.0 + w);  // (C16)

	typename AthenaEOS::error_type error;
	double ye = 0.0;
	w_p= AthenaEOS::press_temp__eps_rho_ye(w_temp,eps,w_d,ye,error);
	Real h = 1. + eps + w_p/w_d;

        Real const conv = 1.0/(h*ud_tmp); // (C26)
        w_vx = conv*m1u;  // (C26)
        w_vy = conv*m2u;  // (C26)
        w_vz = conv*m3u;  // (C26)

        // convert scalars (if any)
        for (int n=nhyd; n<(nhyd+nscal); ++n) {
          prim(m,n,k,j,i) = cons(m,n,k,j,i)/ud_tmp;
        }
      }

      // reset conserved variables inside excised regions or if floor is hit
      if (rad <= coord.bh_rmin || floor_hit) {
        const Real
          &g_00 = g_[I00], &g_01 = g_[I01], &g_02 = g_[I02], &g_03 = g_[I03],
          &g_10 = g_[I01], &g_11 = g_[I11], &g_12 = g_[I12], &g_13 = g_[I13],
          &g_20 = g_[I02], &g_21 = g_[I12], &g_22 = g_[I22], &g_23 = g_[I23],
          &g_30 = g_[I03], &g_31 = g_[I13], &g_32 = g_[I23], &g_33 = g_[I33];

        // Calculate 4-velocity
        Real alpha = sqrt(-1.0/gi_[I00]);
        Real tmp = g_[I11]*w_vx*w_vx + 2.0*g_[I12]*w_vx*w_vy + 2.0*g_[I13]*w_vx*w_vz
                 + g_[I22]*w_vy*w_vy + 2.0*g_[I23]*w_vy*w_vz
                 + g_[I33]*w_vz*w_vz;
        Real gamma = sqrt(1.0 + tmp);
        Real u0 = gamma/alpha;
        Real u1 = w_vx - alpha * gamma * gi_[I01];
        Real u2 = w_vy - alpha * gamma * gi_[I02];
        Real u3 = w_vz - alpha * gamma * gi_[I03];
        Real u_0 = g_00*u0 + g_01*u1 + g_02*u2 + g_03*u3;
        Real u_1 = g_10*u0 + g_11*u1 + g_12*u2 + g_13*u3;
        Real u_2 = g_20*u0 + g_21*u1 + g_22*u2 + g_23*u3;
        Real u_3 = g_30*u0 + g_31*u1 + g_32*u2 + g_33*u3;

        // Set conserved quantities
        Real wgas_u0 = (w_d + gamma_prime * w_p) * u0;
        u_d = w_d * u0;
        u_e = wgas_u0 * u_0 + w_p;
        u_m1 = wgas_u0 * u_1;
        u_m2 = wgas_u0 * u_2;
        u_m3 = wgas_u0 * u_3;

        // convert scalars (if any)
        for (int n=nhyd; n<(nhyd+nscal); ++n) {
          cons(m,n,k,j,i) = prim(m,n,k,j,i)*u_d;
        }
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void PrimToCons()
// \brief Converts primitive into conserved variables.  Operates only over active cells.

void TabulatedGRHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int is = indcs.is; int ie = indcs.ie;
  int js = indcs.js; int je = indcs.je;
  int ks = indcs.ks; int ke = indcs.ke;
  auto coord = pmy_pack->coord.coord_data;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gamma_prime = eos_data.gamma/(eos_data.gamma - 1.0);

  par_for("hyd_prim2cons", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      // Extract components of metric
      Real &x1min = coord.mb_size.d_view(m).x1min;
      Real &x1max = coord.mb_size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = coord.mb_size.d_view(m).x2min;
      Real &x2max = coord.mb_size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = coord.mb_size.d_view(m).x3min;
      Real &x3max = coord.mb_size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real g_[NMETRIC], gi_[NMETRIC];
      ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, false,
                              coord.bh_spin, g_, gi_);

      const Real
        &g_00 = g_[I00], &g_01 = g_[I01], &g_02 = g_[I02], &g_03 = g_[I03],
        &g_10 = g_[I01], &g_11 = g_[I11], &g_12 = g_[I12], &g_13 = g_[I13],
        &g_20 = g_[I02], &g_21 = g_[I12], &g_22 = g_[I22], &g_23 = g_[I23],
        &g_30 = g_[I03], &g_31 = g_[I13], &g_32 = g_[I23], &g_33 = g_[I33];

      Real& w_d  = prim(m,IDN,k,j,i);
      Real& w_p  = prim(m,IPR,k,j,i);
      const Real& w_vx = prim(m,IVX,k,j,i);
      const Real& w_vy = prim(m,IVY,k,j,i);
      const Real& w_vz = prim(m,IVZ,k,j,i);

      Real& w_temp = prim(m, ITEMP,k,j,i);

      // Calculate 4-velocity
      Real alpha = sqrt(-1.0/gi_[I00]);
      Real tmp = g_[I11]*w_vx*w_vx + 2.0*g_[I12]*w_vx*w_vy + 2.0*g_[I13]*w_vx*w_vz
               + g_[I22]*w_vy*w_vy + 2.0*g_[I23]*w_vy*w_vz
               + g_[I33]*w_vz*w_vz;
      Real gamma = sqrt(1.0 + tmp);
      Real u0 = gamma/alpha;
      Real u1 = w_vx - alpha * gamma * gi_[I01];
      Real u2 = w_vy - alpha * gamma * gi_[I02];
      Real u3 = w_vz - alpha * gamma * gi_[I03];
      Real u_0 = g_00*u0 + g_01*u1 + g_02*u2 + g_03*u3;
      Real u_1 = g_10*u0 + g_11*u1 + g_12*u2 + g_13*u3;
      Real u_2 = g_20*u0 + g_21*u1 + g_22*u2 + g_23*u3;
      Real u_3 = g_30*u0 + g_31*u1 + g_32*u2 + g_33*u3;

      // Set conserved quantities
      Real& u_d  = cons(m,IDN,k,j,i);
      Real& u_e  = cons(m,IEN,k,j,i);
      Real& u_m1 = cons(m,IM1,k,j,i);
      Real& u_m2 = cons(m,IM2,k,j,i);
      Real& u_m3 = cons(m,IM3,k,j,i);

      typename AthenaEOS::error_type error;
      double ye = 0.0;
      double eps;
      w_p= AthenaEOS::press_temp__eps_rho_ye(w_temp,eps,w_d,ye,error);
      Real wgas_u0 = w_d*(1. + eps + w_p/w_d) * u0;
      u_d  = w_d * u0;
      u_e  = wgas_u0 * u_0 + w_p;
      u_m1 = wgas_u0 * u_1;
      u_m2 = wgas_u0 * u_2;
      u_m3 = wgas_u0 * u_3;

      // convert scalars (if any)
      for (int n=nhyd; n<(nhyd+nscal); ++n) {
        cons(m,n,k,j,i) = prim(m,n,k,j,i)*u_d;
      }
    }
  );

  return;
}
