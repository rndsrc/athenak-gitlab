//   Elias Roland Most
//   <emost@th.physik.uni-frankfurt.de>
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#include "tabulated.hh"
#define H5_USE_16_API 1
#include <hdf5.h>

#ifndef EOS_TABULATED_READTABLE_COMPOSE_HH
#define EOS_TABULATED_READTABLE_COMPOSE_HH

#ifndef EOS_TABULATED_READTABLE_SCOLLAPSE_HH
// Catch HDF5 errors
#define HDF5_ERROR(fn_call)                                          \
  do {                                                               \
    int _error_code = fn_call;                                       \
    if (_error_code < 0) {                                           \
      printf(    "HDF5 call '%s' returned error code %d", #fn_call, \
                  _error_code);                                      \
    }                                                                \
  } while (0)

static inline int file_is_readable(const char *filename) {
  FILE *fp = NULL;
  fp = fopen(filename, "r");
  if (fp != NULL) {
    fclose(fp);
    return 1;
  }
  return 0;
}
#endif

// Use these two defines to easily read in a lot of variables in the same way
// The first reads in one variable of a given type completely
#define READ_EOS_HDF5_COMPOSE(GROUP,NAME, VAR, TYPE, MEM)                     \
  do {                                                                  \
    hid_t dataset;                                                      \
    HDF5_ERROR(dataset = H5Dopen2(GROUP, NAME, H5P_DEFAULT));            \
    HDF5_ERROR(H5Dread(dataset, TYPE, MEM, H5S_ALL, H5P_DEFAULT, VAR)); \
    HDF5_ERROR(H5Dclose(dataset));                                      \
  } while (0)

#define READ_ATTR_HDF5_COMPOSE(GROUP,NAME, VAR, TYPE)                     \
  do {                                                                  \
    hid_t dataset;                                                      \
    HDF5_ERROR(dataset = H5Aopen(GROUP, NAME, H5P_DEFAULT));            \
    HDF5_ERROR(H5Aread(dataset, TYPE, VAR)); \
    HDF5_ERROR(H5Aclose(dataset));                                      \
  } while (0)

void EOS_Tabulated::readtable_compose(const char *nuceos_table_name) {
  using namespace Margherita_constants;

  constexpr size_t NTABLES = EOS_Tabulated::EV::NUM_VARS;

#ifndef STANDALONE
  CCTK_VInfo(CCTK_THORNSTRING, "*******************************");
  CCTK_VInfo(CCTK_THORNSTRING, "Reading COMPOSE nuc_eos table file:");
  CCTK_VInfo(CCTK_THORNSTRING, "%s", nuceos_table_name);
  CCTK_VInfo(CCTK_THORNSTRING, "*******************************");
#endif

  hid_t file;
  if (!file_is_readable(nuceos_table_name)) {
#ifndef STANDALONE
    CCTK_VError(__LINE__, __FILE__, CCTK_THORNSTRING,
                "Could not read nuceos_table_name %s \n", nuceos_table_name);
#else
    std::cout << "Cannot open table" << std::endl;
#endif
  }

  HDF5_ERROR(file = H5Fopen(nuceos_table_name, H5F_ACC_RDONLY, H5P_DEFAULT));
  hid_t parameters;

  HDF5_ERROR(parameters = H5Gopen(file, "/Parameters"));

  int nrho, ntemp, nye;

  // Read size of tables
  READ_ATTR_HDF5_COMPOSE(parameters,"pointsnb", &nrho, H5T_NATIVE_INT);
  READ_ATTR_HDF5_COMPOSE(parameters,"pointst", &ntemp, H5T_NATIVE_INT);
  READ_ATTR_HDF5_COMPOSE(parameters,"pointsyq", &nye, H5T_NATIVE_INT);

  // Allocate memory for tables

  double *logrho = new double[nrho];
  double *logtemp = new double[ntemp];
  double *yes = new double[nye];

  auto num_points =
      std::array<size_t, 3>{size_t(nrho), size_t(ntemp), size_t(nye)};

  // Read additional tables and variables
  READ_EOS_HDF5_COMPOSE(parameters,"nb", logrho, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_EOS_HDF5_COMPOSE(parameters,"t", logtemp, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_EOS_HDF5_COMPOSE(parameters,"yq", yes, H5T_NATIVE_DOUBLE, H5S_ALL);

  // Thermo Table
  // Number of variables in the thermo table

  hid_t thermo_id;
  HDF5_ERROR(thermo_id = H5Gopen(file, "/Thermo_qty"));
  int nthermo;
  READ_ATTR_HDF5_COMPOSE(thermo_id,"pointsqty", &nthermo, H5T_NATIVE_INT);

  // Read thermo index array
  int *thermo_index = new int[nthermo];
  READ_EOS_HDF5_COMPOSE(thermo_id,"index_thermo", thermo_index, H5T_NATIVE_INT, H5S_ALL);

  // Allocate memory and read table
  double *thermo_table = new double[nthermo * nrho * ntemp * nye];
  READ_EOS_HDF5_COMPOSE(thermo_id,"thermo", thermo_table, H5T_NATIVE_DOUBLE, H5S_ALL);

  // Now read compositions!

  // number of available particle information
  int ncomp=0;
  hid_t comp_id;

  int status_e = H5Eset_auto(NULL, NULL);
  int status_comp = H5Gget_objinfo(file,"/Composition_pairs",0,nullptr);
  if(status_comp ==0){
    HDF5_ERROR(comp_id = H5Gopen(file, "/Composition_pairs"));
    READ_ATTR_HDF5_COMPOSE(comp_id, "pointspairs", &ncomp, H5T_NATIVE_INT);
  }

  int *index_yi = nullptr;
  double *yi_table = nullptr;

  if(ncomp > 0){

    // index identifying particle type
    index_yi = new int[ncomp];
    READ_EOS_HDF5_COMPOSE(comp_id,"index_yi", index_yi, H5T_NATIVE_INT, H5S_ALL);

    // Read composition
    yi_table = new double[ncomp * nrho * ntemp * nye];
    READ_EOS_HDF5_COMPOSE(comp_id,"yi", yi_table, H5T_NATIVE_DOUBLE, H5S_ALL);
  }

  /// IMPORTANT////
  // we always assume nav = 1 as this is NOT standardized!

  // Read average charge and mass numbers
  int nav=0;
  double *zav_table = nullptr;
  double *yav_table = nullptr;
  double *aav_table = nullptr;

  int status_av = H5Gget_objinfo(file,"Composition_quadruples",0,nullptr);

  hid_t av_id;

  if(status_av ==0){
    HDF5_ERROR(av_id = H5Gopen(file, "/Composition_quadruples"));
    READ_ATTR_HDF5_COMPOSE(av_id, "pointsav", &nav, H5T_NATIVE_INT);
  }

  if(nav >0){

    assert(nav == 1 &&
	   "nav != 1 in this table, so there is none or more than "
	   "one definition of an average nucleus."
	   "Please check and generalize accordingly.");

    // Read average tables
    zav_table = new double[nrho * ntemp * nye];
    yav_table = new double[nrho * ntemp * nye];
    aav_table = new double[nrho * ntemp * nye];
    READ_EOS_HDF5_COMPOSE(av_id, "zav", zav_table, H5T_NATIVE_DOUBLE, H5S_ALL);
    READ_EOS_HDF5_COMPOSE(av_id, "yav", yav_table, H5T_NATIVE_DOUBLE, H5S_ALL);
    READ_EOS_HDF5_COMPOSE(av_id, "aav", aav_table, H5T_NATIVE_DOUBLE, H5S_ALL);
  }

  HDF5_ERROR(H5Fclose(file));

  // Need to sort the thermo indices to match the Margherita ordering

  constexpr size_t PRESS_C = 1;
  constexpr size_t S_C = 2;
  constexpr size_t MUN_C = 3;
  constexpr size_t MUP_C = 4;
  constexpr size_t MUE_C = 5;
  // CHECK: is this really the same eps as in the stellar collapse tables?
  constexpr size_t EPS_C = 7;
  constexpr size_t CS2_C = 12;

  auto const find_index = [&](size_t const &index) {
    for (int i = 0; i < nthermo; ++i) {
      if (thermo_index[i] == index) return i;
    }
    assert(!"Could not find index of all required quantities. This should not "
            "happen.");
    return -1;
  };

  // IMPORTANT: The order here needs to match EV in tabulated.hh !
  int thermo_index_conv[7]{find_index(PRESS_C), find_index(EPS_C),
                           find_index(S_C),     find_index(CS2_C),
                           find_index(MUE_C),   find_index(MUP_C),
                           find_index(MUN_C)};

  // Copy to alltables with correct margherita ordering!

  auto alltables =
      std::unique_ptr<double[]>(new double[nrho * ntemp * nye * NTABLES]);

  for (int iv = EV::PRESS; iv <= EV::MUN; iv++)
    for (int k = 0; k < nye; k++)
      for (int j = 0; j < ntemp; j++)
        for (int i = 0; i < nrho; i++) {
          auto const iv_thermo = thermo_index_conv[iv];
          int indold = i + nrho * (j + ntemp * (k + nye * iv_thermo));
          int indnew = iv + NTABLES * (i + nrho * (j + ntemp * k));
          alltables[indnew] = thermo_table[indold];
        }

  auto const find_index_yi = [&](size_t const &index) {
    for (int i = 0; i < ncomp; ++i) {
      if (index_yi[i] == index) return i;
    }
    assert(!"Could not find index of all required quantities. This should not "
            "happen.");
    return -1;
  };

  // Fix average compositions!
  for (int k = 0; k < nye; k++)
    for (int j = 0; j < ntemp; j++)
      for (int i = 0; i < nrho; i++) {
        int indold = i + nrho * (j + ntemp * k);
        int indnew = NTABLES * (i + nrho * (j + ntemp * k));

	if(nav >0){
	  // ABAR
	  alltables[EV::ABAR + indnew] = aav_table[indold];
	  // ZBAR
	  alltables[EV::ZBAR + indnew] = zav_table[indold];
	  // Xh
	  alltables[EV::XH + indnew] = aav_table[indold] * yav_table[indold];
	}
	if(ncomp>0){
	  // Xn
	  alltables[EV::XN + indnew] =
	      yi_table[indold + nrho * nye * ntemp * find_index_yi(10)];
	  // Xp
	  alltables[EV::XP + indnew] =
	      yi_table[indold + nrho * nye * ntemp * find_index_yi(11)];
	  // Xa
	  alltables[EV::XA + indnew] =
	      4. * yi_table[indold + nrho * nye * ntemp * find_index_yi(4002)];
	}
      }

  // Free all storage
  delete[] thermo_index;
  delete[] thermo_table;

  if(index_yi != nullptr) delete[] index_yi;
  if(yi_table != nullptr) delete[] yi_table;

  if(zav_table != nullptr) delete[] zav_table;
  if(yav_table != nullptr) delete[] yav_table;
  if(aav_table != nullptr) delete[] aav_table;

  // assumed baryon mass, we always take the neutron mass
  // Note: The neutron mass is not optimal for this, but it is what the
  //      CompOSE tables provide...
  // IMPORTANT: DO NOT DELETE THIS LINE!
  // The baryon mass is also needed in the leakage and is stored in
  // the public member EOS_Tabulated.
  baryon_mass = m_neutron_MeV * MeV_to_erg / c2_cgs;

  // convert units, convert logs to natural log
  // The latter is great, because exp() is way faster than pow()
  // pressure
  for (int i = 0; i < nrho; i++) {
    logrho[i] = log(logrho[i] * baryon_mass * cm3_to_fm3 * RHOGF);
  }

  for (int i = 0; i < ntemp; i++) {
    logtemp[i] = log(logtemp[i]);
  }

  double *epstable;

  // allocate epstable; a linear-scale eps table
  // that allows us to extrapolate to negative eps
  if (!(epstable = (double *)malloc(nrho * ntemp * nye * sizeof(double)))) {
#ifndef STANDALONE
    CCTK_VError(__LINE__, __FILE__, CCTK_THORNSTRING,
                "Cannot allocate memory for eps table\n");
#else
    std::cout << "Cannot allocate memory for EOS table" << std::endl;
#endif
  }

  c2p_eps_min = 1.e99;

  c2p_h_min = 1.e99;
  c2p_h_max = 0.;
  c2p_press_max = 0.;

  energy_shift = 0;

  // Get eps_min
  for (int i = 0; i < nrho * ntemp * nye; i++) {
    int idx = EOS_Tabulated::EV::EPS + NTABLES * i;
    c2p_eps_min = Margherita_helpers::min(c2p_eps_min, alltables[idx]);
  };

  // convert units
  for (int i = 0; i < nrho * ntemp * nye; i++) {
    double pressL, epsL, rhoL;
    {  // pressure
      int idx = EOS_Tabulated::EV::PRESS + NTABLES * i;
      alltables[idx] = log(alltables[idx] * MeV_to_erg * cm3_to_fm3 * PRESSGF);
      pressL = exp(alltables[idx]);
      c2p_press_max = Margherita_helpers::max(c2p_press_max, pressL);
    }

    {  // eps (This is the correct eps as we use it)
      int idx = EOS_Tabulated::EV::EPS + NTABLES * i;
      // shift epsilon to a positive range if necessary
      if (c2p_eps_min < 0) {
        energy_shift = -2.0 * c2p_eps_min;
        alltables[idx] += energy_shift;
      }
      epstable[i] = alltables[idx];
      alltables[idx] = log(alltables[idx]);
      epsL = epstable[i] - energy_shift;
    }

    {  // cs2
      int idx = EOS_Tabulated::EV::CS2 + NTABLES * i;
      if (alltables[idx] < 0) alltables[idx] = 0;
      alltables[idx] = Margherita_helpers::min(0.9999999, alltables[idx]);
    }

    {  // chemical potentials

      int idx_p = EOS_Tabulated::EV::MUP + NTABLES * i;
      int idx_n = EOS_Tabulated::EV::MUN + NTABLES * i;
      int idx_e = EOS_Tabulated::EV::MUE + NTABLES * i;

      auto const mu_q = alltables[idx_p];
      // Note that this does not include the rest mass contribution of the
      // neutron!
      auto const mu_b = alltables[idx_n];

      // mu_p = mu_b + mu_q = mu_n + mu_q
      // Important: To be consistent we should actually subtract the mass
      // difference between proton and neutron here, but this makes beta eq.
      // complicated. Hence we leave it this way, but fix it in the Leakage!
      alltables[idx_p] += mu_b;
      // mu_e = mu_le - mu_q
      // CHECK: we have mu_l = effective lepton chemical potential (4.7) here
      //        after (3.23) it says, mu_e = mu_le - mu_q
      //        charge neutrality says n_l = n_q = n_le + n_lmu
      //        but how do we get mu_e then?
      //   ERM: We make the explicit assumption that we have no muons....
      //        I know we have to fix this later, but for most EOS this is ok
      //        And if we had muons, I'm pretty sure the Leakage would become
      //        inconsistent...
      // Page 8: Assumptions on the relation between
      //         the electron and muon chemical potentials are discussed
      //         in the description of each model separately.
      // Page 10: In this case, the balance between the
      //          electron and muon densities depends on the assumed relation of
      //          the electron and muon chemical potentials.
      alltables[idx_e] -= mu_q;
    }

    const int irhoL = i % nrho;
    rhoL = exp(logrho[irhoL]);
    const double hL = 1. + epsL + pressL / rhoL;
    c2p_h_min = Margherita_helpers::min(c2p_h_min, hL);
    c2p_h_max = Margherita_helpers::max(c2p_h_max, hL);
  }

  temp0 = exp(logtemp[0]);
  temp1 = exp(logtemp[1]);

  eos_rhomax = exp(logrho[nrho - 1]);
  eos_rhomin = exp(logrho[0]);

  eos_tempmax = exp(logtemp[ntemp - 1]);
  eos_tempmin = exp(logtemp[0]);

  eos_yemax = yes[nye - 1];
  eos_yemin = yes[0];

  auto logrho_ptr = std::unique_ptr<double[]>(new double[nrho]);
  auto logtemp_ptr = std::unique_ptr<double[]>(new double[ntemp]);
  auto ye_ptr = std::unique_ptr<double[]>(new double[nye]);

  for (int i = 0; i < nrho; ++i) logrho_ptr[i] = logrho[i];
  for (int i = 0; i < ntemp; ++i) logtemp_ptr[i] = logtemp[i];
  for (int i = 0; i < nye; ++i) ye_ptr[i] = yes[i];

  delete[] logrho;
  delete[] logtemp;
  delete[] yes;
  delete[] epstable;

  EOS_Tabulated::alltables = linear_interp_uniform_ND_t<double, 3, EV::NUM_VARS>(
      std::move(alltables), std::move(num_points), std::move(logrho_ptr),
      std::move(logtemp_ptr), std::move(ye_ptr));
};

#endif
