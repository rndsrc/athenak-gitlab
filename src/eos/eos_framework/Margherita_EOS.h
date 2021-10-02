#ifndef _HH_MARGHERITA_EOS
#define _HH_MARGHERITA_EOS

#define NEW_TABLE  // Needed for NRaphson C2P
#define STANDALONE  // Needed for NRaphson C2P

#include "3D_Table/tabulated.hh"

//#include "Mag/mag.hh"

#include "Cold/cold_pwpoly.hh"
#include "Cold/cold_pwpoly_implementation.hh"

#include "Cold/cold_table.hh"
#include "Cold/cold_table_implementation.hh"


#include "Hybrid/hybrid.hh"

// Not explicitly instantiating this template here
// has caused problems at the linking stage.
template class EOS_Hybrid<Cold_PWPoly>;
using EOS_Polytropic = EOS_Hybrid<Cold_PWPoly>;

template class Cold_Table_t<0,linear_interp_t>;
using Cold_Table = Cold_Table_t<0, linear_interp_t>;

template class Cold_Table_t<Cold_Table::v_index::NUM_VARS - 2,linear_interp_t>;
using Hot_Slice = Cold_Table_t<Cold_Table::v_index::NUM_VARS - 2,linear_interp_t>;

template class EOS_Hybrid<Cold_Table>;
using EOS_Hybrid_ColdTable = EOS_Hybrid<Cold_Table>;

#endif
