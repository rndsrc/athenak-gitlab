//
//  This routine is taken from EOS_Omni,
//  originally  written by
//
//  Christian Ott, Roland Haas, Evan O'Connor
//
//  and others. This code was originally released under LGPLv2 and is
//  rereleased here under GPLv2 in accordance with LGPLv2. Please refer
//  to stellarcollapse.org and the einsteintoolkit.org for more information.
//
//  Small modifications were done by
//
//   Elias Roland Most
//   <emost@th.physik.uni-frankfurt.de>
//   Ludwig Jens Papenfort
//   <papenfort@th.physik.uni-frankfurt.de>
//
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

#ifndef EOS_TABULATED_READTABLE_SCOLLAPSE_HH
#define EOS_TABULATED_READTABLE_SCOLLAPSE_HH

// Catch HDF5 errors
#define HDF5_ERROR(fn_call)                                          \
  do {                                                               \
    int _error_code = fn_call;                                       \
    if (_error_code < 0) {                                           \
      CCTK_VError(__LINE__, __FILE__, CCTK_THORNSTRING,              \
                  "HDF5 call '%s' returned error code %d", #fn_call, \
                  _error_code);                                      \
    }                                                                \
  } while (0)

// Use these two defines to easily read in a lot of variables in the same way
// The first reads in one variable of a given type completely
#define READ_EOS_HDF5(NAME, VAR, TYPE, MEM)                             \
  do {                                                                  \
    hid_t dataset;                                                      \
    HDF5_ERROR(dataset = H5Dopen(file, NAME));                          \
    HDF5_ERROR(H5Dread(dataset, TYPE, MEM, H5S_ALL, H5P_DEFAULT, VAR)); \
    HDF5_ERROR(H5Dclose(dataset));                                      \
  } while (0)
// The second reads a given variable into a hyperslab of the alltables_temp
// array
#define READ_EOSTABLE_HDF5(NAME, OFF)                                    \
  do {                                                                   \
    hsize_t offset[2] = {OFF, 0};                                        \
    H5Sselect_hyperslab(mem3, H5S_SELECT_SET, offset, NULL, var3, NULL); \
    READ_EOS_HDF5(NAME, alltables_temp, H5T_NATIVE_DOUBLE, mem3);        \
  } while (0)

#ifndef EOS_TABULATED_READTABLE_COMPOSE_HH
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

// FIXME: ERM: What does this mean?
// TODO: replace with version in ET EOS_Omni. NOTE: table arrangement changed.

// Cactus calls this function. It reads in the table and calls a fortran
// function to setup values for the fortran eos module
void EOS_Tabulated::readtable_scollapse(const char *nuceos_table_name,
                                        bool do_energy_shift,
                                        bool recompute_mu_nu) {
  using namespace Margherita_constants;

  constexpr size_t NTABLES = EOS_Tabulated::EV::NUM_VARS;

//  DECLARE_CCTK_PARAMETERS
#ifndef STANDALONE

  CCTK_VInfo(CCTK_THORNSTRING, "*******************************");
  CCTK_VInfo(CCTK_THORNSTRING, "Reading nuc_eos table file:");
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

  int nrho, ntemp, nye;

  // Read size of tables
  READ_EOS_HDF5("pointsrho", &nrho, H5T_NATIVE_INT, H5S_ALL);
  READ_EOS_HDF5("pointstemp", &ntemp, H5T_NATIVE_INT, H5S_ALL);
  READ_EOS_HDF5("pointsye", &nye, H5T_NATIVE_INT, H5S_ALL);

  // Allocate memory for tables
  double *alltables_temp;

  double *logrho;
  double *logtemp;
  double *yes;

  auto num_points =
      std::array<size_t, 3>{size_t(nrho), size_t(ntemp), size_t(nye)};

  if (!(alltables_temp =
            (double *)malloc(nrho * ntemp * nye * NTABLES * sizeof(double)))) {
#ifndef STANDALONE
    CCTK_VError(__LINE__, __FILE__, CCTK_THORNSTRING,
                "Cannot allocate memory for EOS table");
#else
    std::cout << "Cannot allocate memory for EOS table" << std::endl;
#endif
  }
  if (!(logrho = (double *)malloc(nrho * sizeof(double)))) {
#ifndef STANDALONE
    CCTK_VError(__LINE__, __FILE__, CCTK_THORNSTRING,
                "Cannot allocate memory for EOS table");
#else
    std::cout << "Cannot allocate memory for EOS table" << std::endl;
#endif
  }
  if (!(logtemp = (double *)malloc(ntemp * sizeof(double)))) {
#ifndef STANDALONE
    CCTK_VError(__LINE__, __FILE__, CCTK_THORNSTRING,
                "Cannot allocate memory for EOS table");
#else
    std::cout << "Cannot allocate memory for EOS table" << std::endl;
#endif
  }
  if (!(yes = (double *)malloc(nye * sizeof(double)))) {
#ifndef STANDALONE
    CCTK_VError(__LINE__, __FILE__, CCTK_THORNSTRING,
                "Cannot allocate memory for EOS table");
#else
    std::cout << "Cannot allocate memory for EOS table" << std::endl;
#endif
  }

  // Prepare HDF5 to read hyperslabs into alltables_temp
  hsize_t table_dims[2] = {NTABLES, (hsize_t)nrho * ntemp * nye};
  hsize_t var3[2] = {1, (hsize_t)nrho * ntemp * nye};
  hid_t mem3 = H5Screate_simple(2, table_dims, NULL);

  // Read alltables_temp
  READ_EOSTABLE_HDF5("logpress", EOS_Tabulated::EV::PRESS);
  READ_EOSTABLE_HDF5("logenergy", EOS_Tabulated::EV::EPS);
  READ_EOSTABLE_HDF5("entropy", EOS_Tabulated::EV::S);
  //  READ_EOSTABLE_HDF5("munu", 3);
  READ_EOSTABLE_HDF5("cs2", EOS_Tabulated::EV::CS2);
  //  READ_EOSTABLE_HDF5("dedt", 5);
  //  READ_EOSTABLE_HDF5("dpdrhoe", 6);
  //  READ_EOSTABLE_HDF5("dpderho", 7);
  // chemical potentials
  //  READ_EOSTABLE_HDF5("muhat", 8);
  READ_EOSTABLE_HDF5("mu_e", EOS_Tabulated::EV::MUE);
  READ_EOSTABLE_HDF5("mu_p", EOS_Tabulated::EV::MUP);
  READ_EOSTABLE_HDF5("mu_n", EOS_Tabulated::EV::MUN);
  // compositions
  READ_EOSTABLE_HDF5("Xa", EOS_Tabulated::EV::XA);
  READ_EOSTABLE_HDF5("Xh", EOS_Tabulated::EV::XH);
  READ_EOSTABLE_HDF5("Xn", EOS_Tabulated::EV::XN);
  READ_EOSTABLE_HDF5("Xp", EOS_Tabulated::EV::XP);
  // average nucleus
  READ_EOSTABLE_HDF5("Abar", EOS_Tabulated::EV::ABAR);
  READ_EOSTABLE_HDF5("Zbar", EOS_Tabulated::EV::ZBAR);
  // Gamma
  //  READ_EOSTABLE_HDF5("gamma", 18);

  // Read additional tables and variables
  READ_EOS_HDF5("logrho", logrho, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_EOS_HDF5("logtemp", logtemp, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_EOS_HDF5("ye", yes, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_EOS_HDF5("energy_shift", &energy_shift, H5T_NATIVE_DOUBLE, H5S_ALL);

  // Read in baryon mass if contained in the table
  hid_t mb_data;
  auto status = H5Lexists(file, "/mass_factor", H5P_DEFAULT);

  if (status) {
    std::cout << "Reading baryon mass from file." << std::endl;
    HDF5_ERROR(mb_data = H5Dopen(file, "mass_factor"));
    H5Dread(mb_data, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            &baryon_mass);
  } else {
    std::cout << "Using default baryon mass." << std::endl;
  }
  std::cout << "mb = " << baryon_mass << "g/cm^3" << std::endl;

  auto have_rel_cs2 = H5Lexists(file, "/have_rel_cs2", H5P_DEFAULT);
  if (have_rel_cs2) std::cout << "CS2 is already relativistic!" << std::endl;

  HDF5_ERROR(H5Sclose(mem3));
  HDF5_ERROR(H5Fclose(file));

  // change ordering of alltables array so that
  // the table kind is the fastest changing index

  auto alltables =
      std::unique_ptr<double[]>(new double[nrho * ntemp * nye * NTABLES]);

  for (int iv = 0; iv < NTABLES; iv++)
    for (int k = 0; k < nye; k++)
      for (int j = 0; j < ntemp; j++)
        for (int i = 0; i < nrho; i++) {
          int indold = i + nrho * (j + ntemp * (k + nye * iv));
          int indnew = iv + NTABLES * (i + nrho * (j + ntemp * k));
          alltables[indnew] = alltables_temp[indold];
        }

  // free memory of temporary array
  free(alltables_temp);

  // convert units, convert logs to natural log
  // The latter is great, because exp() is way faster than pow()
  // pressure
  energy_shift = energy_shift * EPSGF * do_energy_shift;
  for (int i = 0; i < nrho; i++) {
    // rewrite:
    // logrho[i] = log(pow(10.0,logrho[i]) * RHOGF);
    // by using log(a^b*c) = b*log(a)+log(c)
    logrho[i] = logrho[i] * log(10.) + log(RHOGF);
  }

  for (int i = 0; i < ntemp; i++) {
    // logtemp[i] = log(pow(10.0,logtemp[i]));
    logtemp[i] = logtemp[i] * log(10.0);
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

  // convert units
  for (int i = 0; i < nrho * ntemp * nye; i++) {
    double pressL, epsL, rhoL;
    {  // pressure
      int idx = EOS_Tabulated::EV::PRESS + NTABLES * i;
      alltables[idx] = alltables[idx] * log(10.0) + log(PRESSGF);
      pressL = exp(alltables[idx]);
      c2p_press_max = Margherita_helpers::max(c2p_press_max, pressL);
    }

    {  // eps
      int idx = EOS_Tabulated::EV::EPS + NTABLES * i;
      alltables[idx] = alltables[idx] * log(10.0) + log(EPSGF);
      epstable[i] = exp(alltables[idx]);
      c2p_eps_min =
          Margherita_helpers::min(c2p_eps_min, epstable[i] - energy_shift);
      epsL = epstable[i] - energy_shift;
    }

    {  // cs2
      int idx = EOS_Tabulated::EV::CS2 + NTABLES * i;
      alltables[idx] *= LENGTHGF * LENGTHGF / TIMEGF / TIMEGF;
      if (alltables[idx] < 0) alltables[idx] = 0;
    }

    //  { // dedT
    //    int idx = 5 + NTABLES * i;
    //    alltables[idx] *= EPSGF;
    //  }
    //
    //  { // dpdrhoe
    //    int idx = 6 + NTABLES * i;
    //    alltables[idx] *= PRESSGF / RHOGF;
    //  }
    //
    //  { // dpderho
    //    int idx = 7 + NTABLES * i;
    //    alltables[idx] *= PRESSGF / EPSGF;
    //  }

    const int irhoL = i % nrho;
    rhoL = exp(logrho[irhoL]);
    const double hL = 1. + epsL + pressL / rhoL;
    c2p_h_min = Margherita_helpers::min(c2p_h_min, hL);
    c2p_h_max = Margherita_helpers::max(c2p_h_max, hL);

    {
      int idx = EOS_Tabulated::EV::CS2 + NTABLES * i;
      if (!have_rel_cs2) {
        alltables[idx] /= hL;
      }
      alltables[idx] = Margherita_helpers::min(0.9999999, alltables[idx]);
    }
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

  free(logrho);
  free(logtemp);
  free(yes);
  free(epstable);

  EOS_Tabulated::alltables = linear_interp_uniform_ND_t<double, 3, EV::NUM_VARS>(
      std::move(alltables), std::move(num_points), std::move(logrho_ptr),
      std::move(logtemp_ptr), std::move(ye_ptr));
};

#endif
