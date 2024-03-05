//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb.cpp
//  \brief Problem generator for turbulence
#include <iostream> // cout

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"
#include "srcterms/srcterms.hpp"

// Prototypes for user-defined history functions
void HistoryTurbRel(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::Turb_()
//  \brief Problem Generator for turbulence

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->phydro == nullptr && pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Turbulence problem generator can only be run with Hydro "
                 "and/or MHD, but no "
              << "<hydro> or <mhd> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmbp->pcoord->is_special_relativistic) {
    user_hist_func = HistoryTurbRel;
  }

  // capture variables for kernel
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;

  Real cs = pin->GetOrAddReal("eos", "iso_sound_speed", 1.0);
  Real beta = pin->GetOrAddReal("problem", "beta", 1.0);
  Real amp0 = pin->GetOrAddReal("problem", "amp0", 1.0);

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    Real d_i = pin->GetOrAddReal("problem", "d_i", 1.0);
    Real d_n = pin->GetOrAddReal("problem", "d_n", 1.0);
    auto &u0 = pmbp->phydro->u0;
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0 / eos.gamma;

    // Set initial conditions
    par_for(
        "pgen_turb", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je,
        is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
          u0(m, IDN, k, j, i) = d_n;
          u0(m, IM1, k, j, i) = 0.0;
          u0(m, IM2, k, j, i) = 0.0;
          u0(m, IM3, k, j, i) = 0.0;
          if (eos.is_ideal) {
            u0(m, IEN, k, j, i) = p0 / gm1 + 0.5 *
                                                 (SQR(u0(m, IM1, k, j, i)) +
                                                  SQR(u0(m, IM2, k, j, i)) +
                                                  SQR(u0(m, IM3, k, j, i))) /
                                                 u0(m, IDN, k, j, i);
          }
        });
  }

  // Initialize MHD variables ---------------------------------
  if (pmbp->pmhd != nullptr) {
    Real d_i = pin->GetOrAddReal("problem", "d_i", 1.0);
    Real d_n = pin->GetOrAddReal("problem", "d_n", 1.0);
    Real B0 = cs * std::sqrt(2.0 * d_i / beta);
    auto &u0 = pmbp->pmhd->u0;
    auto &b0 = pmbp->pmhd->b0;
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0 / eos.gamma;

    // Set initial conditions
    par_for(
        "pgen_turb", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je,
        is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
          u0(m, IDN, k, j, i) = 1.0;
          u0(m, IM1, k, j, i) = 0.0;
          u0(m, IM2, k, j, i) = 0.0;
          u0(m, IM3, k, j, i) = 0.0;

          // initialize B
          b0.x1f(m, k, j, i) = 0.0;
          b0.x2f(m, k, j, i) = 0.0;
          b0.x3f(m, k, j, i) = B0;
          if (i == ie) {
            b0.x1f(m, k, j, i + 1) = 0.0;
          }
          if (j == je) {
            b0.x2f(m, k, j + 1, i) = 0.0;
          }
          if (k == ke) {
            b0.x3f(m, k + 1, j, i) = B0;
          }

          if (eos.is_ideal) {
            u0(m, IEN, k, j, i) =
                p0 / gm1 + 0.5 * B0 * B0 +  // fix contribution from dB
                0.5 *
                    (SQR(u0(m, IM1, k, j, i)) + SQR(u0(m, IM2, k, j, i)) +
                     SQR(u0(m, IM3, k, j, i))) /
                    u0(m, IDN, k, j, i);
          }
        });
  }

  // Initialize ion-neutral variables -------------------------
  if (pmbp->pionn != nullptr) {
    Real d_i = pin->GetOrAddReal("problem", "d_i", 1.0);
    Real d_n = pin->GetOrAddReal("problem", "d_n", 1.0);
    Real B0 = cs * std::sqrt(2.0 * (d_i + d_n) / beta);

    // MHD
    auto &u0 = pmbp->pmhd->u0;
    auto &b0 = pmbp->pmhd->b0;
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = d_i / eos.gamma;  // TODO(@user): multiply by ionized density

    // Set initial conditions
    par_for(
        "pgen_turb_mhd", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js,
        je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
          u0(m, IDN, k, j, i) = d_i;
          u0(m, IM1, k, j, i) = 0.0;
          u0(m, IM2, k, j, i) = 0.0;
          u0(m, IM3, k, j, i) = 0.0;

          // initialize B
          b0.x1f(m, k, j, i) = 0.0;
          b0.x2f(m, k, j, i) = 0.0;
          b0.x3f(m, k, j, i) = B0;
          if (i == ie) {
            b0.x1f(m, k, j, i + 1) = 0.0;
          }
          if (j == je) {
            b0.x2f(m, k, j + 1, i) = 0.0;
          }
          if (k == ke) {
            b0.x3f(m, k + 1, j, i) = B0;
          }

          if (eos.is_ideal) {
            u0(m, IEN, k, j, i) =
                p0 / gm1 + 0.5 * B0 * B0 +  // fix contribution from dB
                0.5 *
                    (SQR(u0(m, IM1, k, j, i)) + SQR(u0(m, IM2, k, j, i)) +
                     SQR(u0(m, IM3, k, j, i))) /
                    u0(m, IDN, k, j, i);
          }
        });
    // Hydro
    auto &u0_ = pmbp->phydro->u0;
    EOS_Data &eos_ = pmbp->phydro->peos->eos_data;
    Real gm1_ = eos_.gamma - 1.0;
    Real p0_ = d_n / eos_.gamma;  // TODO(@user): multiply by neutral density

    // Set initial conditions
    par_for(
        "pgen_turb_hydro", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke,
        js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
          u0_(m, IDN, k, j, i) = d_n;
          u0_(m, IM1, k, j, i) = 0.0;
          u0_(m, IM2, k, j, i) = 0.0;
          u0_(m, IM3, k, j, i) = 0.0;
          if (eos_.is_ideal) {
            u0_(m, IEN, k, j, i) =
                p0_ / gm1_ +
                0.5 *
                    (SQR(u0_(m, IM1, k, j, i)) + SQR(u0_(m, IM2, k, j, i)) +
                     SQR(u0_(m, IM3, k, j, i))) /
                    u0_(m, IDN, k, j, i);
          }
        });
  }

  return;
}

//----------------------------------------------------------------------------------------
// Function for computing relativistic turbulence diagnostics

void HistoryTurbRel(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &size = pm->pmb_pack->pmb->mb_size;

  // set nvars, adiabatic index, primitive array w0, and field array bcc0 if
  // is_mhd
  int nvars;
  Real gamma;
  bool is_mhd = false;
  DvceArray5D<Real> w0_, bcc0_, u0_;

  EquationOfState *peos;
  SourceTerms *psrc = nullptr;

  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
    gamma = pmbp->phydro->peos->eos_data.gamma;
    w0_ = pmbp->phydro->w0;
    u0_ = pmbp->phydro->u0;
    peos = (pmbp->phydro->peos);
    psrc = pmbp->phydro->psrc;
  } else if (pmbp->pmhd != nullptr) {
    is_mhd = true;
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    gamma = pmbp->pmhd->peos->eos_data.gamma;
    w0_ = pmbp->pmhd->w0;
    u0_ = pmbp->pmhd->u0;
    bcc0_ = pmbp->pmhd->bcc0;
    peos = (pmbp->pmhd->peos);
    psrc = pmbp->pmhd->psrc;
  }

  bool use_cooling = false;
  Real cooling_rate;
  Real cooling_power;
  if (psrc != nullptr) {
    if (psrc->source_terms_enabled) {
      use_cooling = psrc->rel_cooling;

      cooling_rate = psrc->crate_rel;
      cooling_power = psrc->cpower_rel;
    }
  }

  EOS_Data &eos_data = peos->eos_data;

  int num_hist = 0;

  pdata->label[num_hist] = "mass";
  num_hist++;
  pdata->label[num_hist] = "1-mom";
  num_hist++;
  pdata->label[num_hist] = "2-mom";
  num_hist++;
  pdata->label[num_hist] = "3-mom";
  num_hist++;
  if (eos_data.is_ideal) {
    pdata->label[num_hist] = "tot-E";
    num_hist++;
  }
  // Internal energy is next
  pdata->label[num_hist] = "INT-E";
  num_hist++;
  // Kinetic energy is next
  pdata->label[num_hist] = "K-E";
  num_hist++;
  if (is_mhd) {
    // Magnetic energy is next
    pdata->label[num_hist] = "M-E";
    num_hist++;
  }
  if (use_cooling) {
    // Cooling is next
    pdata->label[num_hist] = "E-cooling";
    num_hist++;
    pdata->label[num_hist] = "S-cooling";
    num_hist++;
  }

  // Store number of hist vars
  pdata->nhist = num_hist;

  // Sanity check
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int nx1 = indcs.nx1;
  int js = indcs.js;
  int nx2 = indcs.nx2;
  int ks = indcs.ks;
  int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack) * nx3 * nx2 * nx1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  array_sum::GlobalSum sum_this_mb;
  Kokkos::parallel_reduce(
      "HistSums", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
        // compute n,k,j,i indices of thread
        int m = (idx) / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / nx1;
        int i = (idx - m * nkji - k * nji - j * nx1) + is;
        k += ks;
        j += js;

        Real vol = size.d_view(m).dx1 * size.d_view(m).dx2 * size.d_view(m).dx3;

        // MHD conserved variables:
        array_sum::GlobalSum hvars;
        int nn = 0;
        hvars.the_array[IDN] = vol * u0_(m, IDN, k, j, i);
        nn++;
        hvars.the_array[IM1] = vol * u0_(m, IM1, k, j, i);
        nn++;
        hvars.the_array[IM2] = vol * u0_(m, IM2, k, j, i);
        nn++;
        hvars.the_array[IM3] = vol * u0_(m, IM3, k, j, i);
        nn++;
        if (eos_data.is_ideal) {
          hvars.the_array[IEN] = vol * u0_(m, IEN, k, j, i);
          nn++;
        }

        Real lorentz = u0_(m, IDN, k, j, i) / w0_(m, IDN, k, j, i);

        // INT-E
        hvars.the_array[nn] = vol * lorentz * w0_(m, IEN, k, j, i);
        nn++;

        Real z2 = lorentz * lorentz - 1.;
        Real pressure = eos_data.IdealGasPressure(w0_(m, IEN, k, j, i));
        Real rhoh = w0_(m, IDN, k, j, i) + w0_(m, IEN, k, j, i) + pressure;

        // K-E
        hvars.the_array[nn] = vol * rhoh * z2;
        nn++;

        if (is_mhd) {
          Real b2 = SQR(bcc0_(m, IBX, k, j, i)) + SQR(bcc0_(m, IBY, k, j, i)) +
                    SQR(bcc0_(m, IBZ, k, j, i));
          Real Bdotv = bcc0_(m, IBX, k, j, i) * w0_(m, IVX, k, j, i) +
                       bcc0_(m, IBY, k, j, i) * w0_(m, IVY, k, j, i) +
                       bcc0_(m, IBZ, k, j, i) * w0_(m, IVZ, k, j, i);
          b2 += SQR(Bdotv);
          b2 /= (lorentz * lorentz);

          hvars.the_array[nn] = vol * lorentz * b2;
          nn++;
        }

        if (use_cooling) {
          // Ideal EOS only
          Real temp = pressure / w0_(m, IDN, k, j, i);

          // E cooling
          hvars.the_array[nn] = vol * u0_(m, IDN, k, j, i) *
                                pow((temp * cooling_rate), cooling_power);
          nn++;

          // S cooling
          hvars.the_array[nn] = vol * w0_(m, IDN, k, j, i) * z2 *
                                pow((temp * cooling_rate), cooling_power);
          nn++;
        }

        // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
        for (int n = nn; n < NHISTORY_VARIABLES; ++n) {
          hvars.the_array[n] = 0.0;
        }

        // sum into parallel reduce
        mb_sum += hvars;
      },
      Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));
  Kokkos::fence();

  // store data into hdata array
  for (int n = 0; n < pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_mb.the_array[n];
  }

  // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
  for (int n = pdata->nhist; n < NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }

  return;
}
