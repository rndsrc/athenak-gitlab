//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh.cpp
//  \brief implementation of constructor and functions in Mesh class

#include <iostream>
#include <cinttypes>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "outputs/io_wrapper.hpp"
#include "mesh.hpp"
#include "coordinates/cell_locations.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// Mesh constructor: initializes some mesh variables at start of calculation using
// parameters in input file.  Most objects in Mesh are constructed in the BuildTree()
// function, so that they can store a pointer to the Mesh which can be reliably referenced
// only after the Mesh constructor has finished

Mesh::Mesh(ParameterInput *pin)
  : one_d(false), 
    two_d(false),
    three_d(false),
    multi_d(false),
    shearing_periodic(false)
{
  // Set physical size and number of cells in mesh (root level)
  mesh_size.x1min = pin->GetReal("mesh", "x1min");
  mesh_size.x1max = pin->GetReal("mesh", "x1max");
  mesh_size.x2min = pin->GetReal("mesh", "x2min");
  mesh_size.x2max = pin->GetReal("mesh", "x2max");
  mesh_size.x3min = pin->GetReal("mesh", "x3min");
  mesh_size.x3max = pin->GetReal("mesh", "x3max");

  mesh_indcs.ng  = pin->GetOrAddReal("mesh", "nghost", 2);
  mesh_indcs.nx1 = pin->GetInteger("mesh", "nx1");
  mesh_indcs.nx2 = pin->GetInteger("mesh", "nx2");
  mesh_indcs.nx3 = pin->GetInteger("mesh", "nx3");

  // define some useful flags that indicate 1D/2D/3D calculations
  if (mesh_indcs.nx3 > 1) {
    three_d = true;
    multi_d = true;
  } else if (mesh_indcs.nx2 > 1) {
    two_d = true;
    multi_d = true;
  } else {
    one_d = true;
  }

  // Set BC flags for ix1/ox1 boundaries and error check
  mesh_bcs[BoundaryFace::inner_x1] = GetBoundaryFlag(pin->GetString("mesh", "ix1_bc"));
  mesh_bcs[BoundaryFace::outer_x1] = GetBoundaryFlag(pin->GetString("mesh", "ox1_bc"));
  if ((mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic ||
       mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::periodic) &&
       mesh_bcs[BoundaryFace::inner_x1] != mesh_bcs[BoundaryFace::outer_x1]) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Both inner and outer x1 bcs must be periodic" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Check if x1 boundaries are shearing periodic. When flag set to true, shearing BCs
  // will be called in ApplyPhysicalBCs() in Hydro and/or MHD.
  if (mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic) {
    shearing_periodic = pin->GetOrAddBoolean("mesh","speriodic",false);
  }

  // Set BC flags for ix2/ox2 boundaries and error check
  if (multi_d) {
    mesh_bcs[BoundaryFace::inner_x2] = GetBoundaryFlag(pin->GetString("mesh", "ix2_bc"));
    mesh_bcs[BoundaryFace::outer_x2] = GetBoundaryFlag(pin->GetString("mesh", "ox2_bc"));
    if ((mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic ||
         mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::periodic) &&
         mesh_bcs[BoundaryFace::inner_x2] != mesh_bcs[BoundaryFace::outer_x2]) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Both inner and outer x2 bcs must be periodic" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    // ix2/ox2 BC flags set to undef for 1D problems
    mesh_bcs[BoundaryFace::inner_x2] = BoundaryFlag::undef;
    mesh_bcs[BoundaryFace::outer_x2] = BoundaryFlag::undef;
  }

  // Set BC flags for ix3/ox3 boundaries and error check
  if (three_d) {
    mesh_bcs[BoundaryFace::inner_x3] = GetBoundaryFlag(pin->GetString("mesh", "ix3_bc"));
    mesh_bcs[BoundaryFace::outer_x3] = GetBoundaryFlag(pin->GetString("mesh", "ox3_bc"));
    if ((mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic ||
         mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::periodic) &&
         mesh_bcs[BoundaryFace::inner_x3] != mesh_bcs[BoundaryFace::outer_x3]) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Both inner and outer x3 bcs must be periodic" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    // ix3/ox3 BC flags set to undef for 1D or 2D problems
    mesh_bcs[BoundaryFace::inner_x3] = BoundaryFlag::undef;
    mesh_bcs[BoundaryFace::outer_x3] = BoundaryFlag::undef;
  }

  // set boolean flags indicating type of refinement (if any), and whether mesh is
  // periodic, depending on input strings
  adaptive = 
    (pin->GetOrAddString("mesh", "refinement", "none") == "adaptive") ? true : false;
  multilevel =
    ((adaptive) || (pin->GetString("mesh", "refinement") == "static")) ?  true : false;

  // error check physical size of mesh (root level) from input file.
  if (mesh_size.x1max <= mesh_size.x1min) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Input x1max must be larger than x1min: x1min=" << mesh_size.x1min
        << " x1max=" << mesh_size.x1max << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_size.x2max <= mesh_size.x2min) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Input x2max must be larger than x2min: x2min=" << mesh_size.x2min
        << " x2max=" << mesh_size.x2max << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_size.x3max <= mesh_size.x3min) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Input x3max must be larger than x3min: x3min=" << mesh_size.x3min
        << " x3max=" << mesh_size.x3max << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // error check requested number of grid cells for entire root domain
  if (mesh_indcs.nx1 < 4) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In mesh block in input file nx1 must be >= 4, but nx1=" << mesh_indcs.nx1
        << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_indcs.nx2 < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In mesh block in input file nx2 must be >= 1, but nx2=" << mesh_indcs.nx2
        << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_indcs.nx3 < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In mesh block in input file nx3 must be >= 1, but nx3=" << mesh_indcs.nx3
        << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_indcs.nx2 == 1 && mesh_indcs.nx3 > 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In mesh block in input file: nx2=1, nx3=" << mesh_indcs.nx3 
        << ", but 2D problems in x1-x3 plane not supported" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // error check number of ghost zones
  if (mesh_indcs.ng < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
      << "More than 2 ghost zones required, but nghost=" <<mesh_indcs.ng << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ((multilevel) && (mesh_indcs.ng % 2 != 0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
      << "Number of ghost zones must be divisible by two for SMR/AMR calculations, "
      << "but nghost=" << mesh_indcs.ng << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // passed error checks, compute grid spacing in (virtual) mesh grid
  mesh_size.dx1 = (mesh_size.x1max-mesh_size.x1min)/static_cast<Real>(mesh_indcs.nx1);
  mesh_size.dx2 = (mesh_size.x2max-mesh_size.x2min)/static_cast<Real>(mesh_indcs.nx2);
  mesh_size.dx3 = (mesh_size.x3max-mesh_size.x3min)/static_cast<Real>(mesh_indcs.nx3);
}

//----------------------------------------------------------------------------------------
// Mesh constructor for restarts. Load the restart file

//----------------------------------------------------------------------------------------
// destructor

Mesh::~Mesh()
{
  delete [] costlist;
  delete [] ranklist;
  delete [] loclist;
  delete [] gidslist;
  delete [] nmblist;
  if (adaptive) { // deallocate arrays for AMR
    delete [] nref;
    delete [] nderef;
    delete [] rdisp;
    delete [] ddisp;
    delete [] bnref;
    delete [] bnderef;
    delete [] brdisp;
    delete [] bddisp;
  }
}

//----------------------------------------------------------------------------------------
// BuildTree: constructs MeshBlockTree, MeshBlockPack (containing physics modules), and
// MeshBlock(s).  Does initial load balance based on simple cost estimate.

void Mesh::BuildTree(ParameterInput *pin)
{
  // Calculate # of cells in MeshBlock read from input parameters, error check
  RegionIndcs mb_indices;
  mb_indices.ng  = mesh_indcs.ng;
  mb_indices.nx1 = pin->GetOrAddInteger("meshblock", "nx1", mesh_indcs.nx1);
  if (multi_d) {
    mb_indices.nx2 = pin->GetOrAddInteger("meshblock", "nx2", mesh_indcs.nx2);
  } else {
    mb_indices.nx2 = mesh_indcs.nx2;
  }
  if (three_d) {
    mb_indices.nx3 = pin->GetOrAddInteger("meshblock", "nx3", mesh_indcs.nx3);
  } else {
    mb_indices.nx3 = mesh_indcs.nx3;
  }

  // error check consistency of the block and mesh
  if (   mesh_indcs.nx1 % mb_indices.nx1 != 0
      || mesh_indcs.nx2 % mb_indices.nx2 != 0
      || mesh_indcs.nx3 % mb_indices.nx3 != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Mesh must be evenly divisible by MeshBlocks" << std::endl
              << "Check Mesh and MeshBlock dimensions in input file" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ( mb_indices.nx1 < 4 ||
      (mb_indices.nx2 < 4 && multi_d) ||
      (mb_indices.nx3 < 4 && three_d) ) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MeshBlock must be >= 4 cells in each active dimension" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // calculate the number of MeshBlocks at root level in each dir
  nmb_rootx1 = mesh_indcs.nx1/mb_indices.nx1;
  nmb_rootx2 = mesh_indcs.nx2/mb_indices.nx2;
  nmb_rootx3 = mesh_indcs.nx3/mb_indices.nx3;

  // find maximum number of MeshBlocks at root level in any dir
  int nmbmax = (nmb_rootx1 > nmb_rootx2) ? nmb_rootx1 : nmb_rootx2;
  nmbmax = (nmbmax > nmb_rootx3) ? nmbmax : nmb_rootx3;

  // Find smallest N such that 2^N > max number of MeshBlocks in any dimension (nmbmax)
  // Then N is logical level of root grid.  2^N implemented as left-shift (1<<root_level)
  for (root_level=0; ((1<<root_level) < nmbmax); root_level++) {}
  int current_level = root_level; 

  // Construct tree and create root grid
  ptree = std::make_unique<MeshBlockTree>(this);
  ptree->CreateRootGrid();

  // Error check properties of input paraemters for SMR/AMR meshes.
  if (adaptive) {
    max_level = pin->GetOrAddInteger("mesh", "numlevel", 1) + root_level - 1;
    if (max_level > 31) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Number of refinement levels must be smaller than " 
                << 31 - root_level + 1 << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    max_level = 31;
  }

  // For meshes with refinement, construct new nodes for <refinement> blocks in input file

  if (multilevel) {
    // error check that number of cells in MeshBlock divisible by two
    if (mb_indices.nx1 % 2 != 0 || 
       (mb_indices.nx2 % 2 != 0 && multi_d) ||
       (mb_indices.nx3 % 2 != 0 && three_d)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Number of cells in MeshBlock must be divisible by 2 "
                << "with SMR or AMR." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // cycle through ParameterInput list and find "refinement" blocks (SMR), extract data
    // Expand MeshBlockTree to include "refinement" regions specified in input file:
    for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
      if (it->block_name.compare(0, 10, "refinement") == 0) {
        RegionSize ref_size;
        ref_size.x1min = pin->GetReal(it->block_name, "x1min");
        ref_size.x1max = pin->GetReal(it->block_name, "x1max");
        if (multi_d) {
          ref_size.x2min = pin->GetReal(it->block_name, "x2min");
          ref_size.x2max = pin->GetReal(it->block_name, "x2max");
        } else {
          ref_size.x2min = mesh_size.x2min;
          ref_size.x2max = mesh_size.x2max;
        }
        if (three_d) { 
          ref_size.x3min = pin->GetReal(it->block_name, "x3min");
          ref_size.x3max = pin->GetReal(it->block_name, "x3max");
        } else {
          ref_size.x3min = mesh_size.x3min;
          ref_size.x3max = mesh_size.x3max;
        }
        int phy_ref_lev = pin->GetInteger(it->block_name, "level");
        int log_ref_lev = phy_ref_lev + root_level;
        if (log_ref_lev > current_level) current_level = log_ref_lev;

        // error check parameters in "refinement" blocks
        if (phy_ref_lev < 1) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ 
              << std::endl << "Refinement level must be larger than 0 (root level = 0)"
              << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (log_ref_lev > max_level) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Refinement level exceeds maximum allowed ("
              << max_level << ")" << std::endl << "Reduce/specify 'numlevel' in <mesh> "
              << "input block if using AMR" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (   ref_size.x1min > ref_size.x1max
            || ref_size.x2min > ref_size.x2max
            || ref_size.x3min > ref_size.x3max)  {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Invalid refinement region (xmax < xmin in one direction)."
              << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (   ref_size.x1min < mesh_size.x1min || ref_size.x1max > mesh_size.x1max
            || ref_size.x2min < mesh_size.x2min || ref_size.x2max > mesh_size.x2max
            || ref_size.x3min < mesh_size.x3min || ref_size.x3max > mesh_size.x3max) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Refinement region must be fully contained within root mesh"
              << std::endl;
          std::exit(EXIT_FAILURE);
        }

        // note: if following is too slow, it could be replaced with bi-section search.
        // Suppose entire root domain is tiled with MeshBlocks at the desired refinement
        // level. Find range of x1-integer indices of such MeshBlocks that cover the
        // refinement region
        std::int32_t lx1min = 0, lx1max = 0;
        std::int32_t lx2min = 0, lx2max = 0;
        std::int32_t lx3min = 0, lx3max = 0;
        std::int32_t lxmax = nmb_rootx1*(1<<phy_ref_lev);
        for (lx1min=0; lx1min<lxmax; lx1min++) {
          if (LeftEdgeX(lx1min+1,lxmax,mesh_size.x1min,mesh_size.x1max) > ref_size.x1min)
            break;
        }
        for (lx1max=lx1min; lx1max<lxmax; lx1max++) {
          if (LeftEdgeX(lx1max+1,lxmax,mesh_size.x1min,mesh_size.x1max) >= ref_size.x1max)
            break;
        }
        if (lx1min % 2 == 1) lx1min--;
        if (lx1max % 2 == 0) lx1max++;

        // Find range of x2-indices of such MeshBlocks that cover the refinement region
        if (multi_d) { // 2D or 3D
          lxmax = nmb_rootx2*(1<<phy_ref_lev);
          for (lx2min=0; lx2min<lxmax; lx2min++) {
            if (LeftEdgeX(lx2min+1, lxmax, mesh_size.x2min, mesh_size.x2max) >
                ref_size.x2min)
            break;
          }
          for (lx2max=lx2min; lx2max<lxmax; lx2max++) {
            if (LeftEdgeX(lx2max+1, lxmax, mesh_size.x2min, mesh_size.x2max) >=
                ref_size.x2max)
            break;
          }
          if (lx2min % 2 == 1) lx2min--;
          if (lx2max % 2 == 0) lx2max++;
        }

        // Find range of x3-indices of such MeshBlocks that cover the refinement region
        if (three_d) { // 3D
          lxmax = nmb_rootx3*(1<<phy_ref_lev);
          for (lx3min=0; lx3min<lxmax; lx3min++) {
            if (LeftEdgeX(lx3min+1, lxmax, mesh_size.x3min, mesh_size.x3max) >
                ref_size.x3min)
            break;
          }
          for (lx3max=lx3min; lx3max<lxmax; lx3max++) {
            if (LeftEdgeX(lx3max+1, lxmax, mesh_size.x3min, mesh_size.x3max) >=
                ref_size.x3max)
            break;
          }
          if (lx3min % 2 == 1) lx3min--;
          if (lx3max % 2 == 0) lx3max++;
        }

        // Now add nodes to the MeshBlockTree corresponding to these MeshBlocks
        if (one_d) {  // 1D
          for (std::int32_t i=lx1min; i<lx1max; i+=2) {
            LogicalLocation nloc;
            nloc.level=log_ref_lev, nloc.lx1=i, nloc.lx2=0, nloc.lx3=0;
            int nnew;
            ptree->AddNode(nloc, nnew);
          }
        }
        if (two_d) {  // 2D
          for (std::int32_t j=lx2min; j<lx2max; j+=2) {
            for (std::int32_t i=lx1min; i<lx1max; i+=2) {
              LogicalLocation nloc;
              nloc.level=log_ref_lev, nloc.lx1=i, nloc.lx2=j, nloc.lx3=0;
              int nnew;
              ptree->AddNode(nloc, nnew);
            }
          }
        }
        if (three_d) {  // 3D
          for (std::int32_t k=lx3min; k<lx3max; k+=2) {
            for (std::int32_t j=lx2min; j<lx2max; j+=2) {
              for (std::int32_t i=lx1min; i<lx1max; i+=2) {
                LogicalLocation nloc;
                nloc.level = log_ref_lev, nloc.lx1 = i, nloc.lx2 = j, nloc.lx3 = k;
                int nnew;
                ptree->AddNode(nloc, nnew);
              }
            }
          }
        }

      }
    }
  } // if(multilevel)

  if (!adaptive) max_level = current_level;

  // initial mesh hierarchy construction is completed here
  ptree->CountMeshBlock(nmb_total);

  costlist = new double[nmb_total];
  ranklist = new int[nmb_total];
  loclist  = new LogicalLocation[nmb_total];

  gidslist = new int[global_variable::nranks];
  nmblist  = new int[global_variable::nranks];
  if (adaptive) { // allocate arrays for AMR
    nref = new int[global_variable::nranks];
    nderef = new int[global_variable::nranks];
    rdisp = new int[global_variable::nranks];
    ddisp = new int[global_variable::nranks];
    bnref = new int[global_variable::nranks];
    bnderef = new int[global_variable::nranks];
    brdisp = new int[global_variable::nranks];
    bddisp = new int[global_variable::nranks];
  }

  // following returns LogicalLocation list sorted by Z-ordering
  ptree->CreateMeshBlockList(loclist, nullptr, nmb_total);

#if MPI_PARALLEL_ENABLED
  // check there is at least one MeshBlock per MPI rank
  if (nmb_total < global_variable::nranks) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Fewer MeshBlocks (nmb_total=" << nmb_total << ") than MPI ranks (nranks="
        << global_variable::nranks << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif

  // initialize cost array with the simplest estimate; all the blocks are equal
  for (int i=0; i<nmb_total; i++) {costlist[i] = 1.0;}
  LoadBalance(costlist, ranklist, gidslist, nmblist, nmb_total);

  // create MeshBlockPack for this rank
  gids = gidslist[global_variable::my_rank];
  gide = gids + nmblist[global_variable::my_rank] - 1;
  nmb_thisrank = nmblist[global_variable::my_rank];

  pmb_pack = new MeshBlockPack(this, pin, gids, gide, mb_indices);
  pmb_pack->pmb->SetNeighbors(ptree, ranklist);
  
/*******/
//  for (int m=0; m<pmb_pack->nmb_thispack; ++m) {
//    std::cout << "******* Block=" << pmb_pack->pmb->mbgid.h_view(m) << std::endl;
//    for (int n=0; n<6; ++n) {
//      std::cout << "n=" << n << " bc_flag=" << GetBoundaryString(pmb_pack->pmb->mb_bcs(m,n)) << std::endl;
//    }
//    for (int n=0; n<pmb_pack->pmb->nnghbr; ++n) {
//      std::cout << "n=" << n << " gid=" << pmb_pack->pmb->nghbr[n].gid.h_view(m) << " level=" << pmb_pack->pmb->nghbr[n].lev.h_view(m) << " rank=" << pmb_pack->pmb->nghbr[n].rank.h_view(m) << " destn=" << pmb_pack->pmb->nghbr[n].destn.h_view(m) << std::endl;
//    }
//  }
/**********/

  ResetLoadBalanceCounters();
  // if (global_variable::my_rank == 0) {PrintMeshDiagnostics();}

  // set initial time/cycle parameters
  time = pin->GetOrAddReal("time", "start_time", 0.0);
  dt   = std::numeric_limits<float>::max();
  cfl_no = pin->GetReal("time", "cfl_number");
  ncycle = 0;
  
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::PrintMeshDiagnostics()
//  \brief prints information about mesh structure, always called at start of every
//  calculation at end of BuildTree

void Mesh::PrintMeshDiagnostics()
{
  std::cout << std::endl;
  std::cout <<"Root grid = "<< nmb_rootx1 <<" x "<< nmb_rootx2 <<" x "<< nmb_rootx3
            <<" MeshBlocks"<< std::endl;
  std::cout <<"Total number of MeshBlocks = " << nmb_total << std::endl;
  std::cout <<"Number of logical  levels of refinement = "<< max_level
            <<" (" << (max_level + 1) << " levels total)" << std::endl;
  std::cout <<"Number of physical levels of refinement = "<< (max_level - root_level) 
            <<" (" << (max_level - root_level + 1) << " levels total)" << std::endl;

  // if more than one physical level: compute/output # of blocks and cost per level
  if ((max_level - root_level) > 1) {
    int nb_per_plevel[max_level];
    float cost_per_plevel[max_level];
    for (int i=0; i<=max_level; ++i) {
      nb_per_plevel[i] = 0;
      cost_per_plevel[i] = 0.0;
    }
    for (int i=0; i<nmb_total; i++) {
      nb_per_plevel[(loclist[i].level - root_level)]++;
      cost_per_plevel[(loclist[i].level - root_level)] += costlist[i];
    }
    for (int i=root_level; i<=max_level; i++) {
      if (nb_per_plevel[i-root_level] != 0) {
        std::cout << "  Physical level = " << i-root_level << " (logical level = " << i
                  << "): " << nb_per_plevel[i-root_level] << " MeshBlocks, cost = "
                  << cost_per_plevel[i-root_level] <<  std::endl;
      }
    }
  }

  std::cout << "Number of parallel ranks = " << global_variable::nranks << std::endl;
  // if more than one rank: compute/output # of blocks and cost per rank
  if (global_variable::nranks > 1) {
    int nb_per_rank[global_variable::nranks];
    int cost_per_rank[global_variable::nranks];
    for (int i=0; i<global_variable::nranks; ++i) {
      nb_per_rank[i] = 0;
      cost_per_rank[i] = 0;
    }
    for (int i=0; i<nmb_total; i++) {
      nb_per_rank[ranklist[i]]++;
      cost_per_rank[ranklist[i]] += costlist[i];
    }
    for (int i=0; i<global_variable::nranks; ++i) {
      std::cout << "  Rank = " << i << ": " << nb_per_rank[i] <<" MeshBlocks, cost = "
                << cost_per_rank[i] << std::endl;
    }

    // output total cost and load balancing info
    double mincost = std::numeric_limits<double>::max();
    double maxcost = 0.0, totalcost = 0.0;
    for (int i=root_level; i<=max_level; i++) {
      for (int j=0; j<nmb_total; j++) {
        if (loclist[j].level == i) {
          mincost = std::min(mincost,costlist[i]);
          maxcost = std::max(maxcost,costlist[i]);
          totalcost += costlist[i];
        }
      }
    }
    std::cout << "Load Balancing:" << std::endl;
    std::cout << "  Minimum cost = " << mincost << ", Maximum cost = " << maxcost
              << ", Average cost = " << totalcost/nmb_total << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::WriteMeshStructure(int ndim)
//  \brief writes file containing MeshBlock positions and sizes that can be used to create
//  plots using 'plot_mesh.py' script.  Only works for 2D/3D data.  Called from main if
//  '-m' option is given on command line. 

void Mesh::WriteMeshStructure()
{
  if (one_d) {
    std::cout << "WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Mesh only 1D, so no 'mesh_structure.dat' file produced" << std::endl;
    return;
  }

  FILE *fp = nullptr;
  if ((fp = std::fopen("mesh_structure.dat","wb")) == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ 
        << std::endl << "Cannot open 'mesh_structure.dat' file for output" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &size = this->pmb_pack->coord.coord_data.mb_size;
  for (int i=root_level; i<=max_level; i++) {
  for (int j=0; j<nmb_total; j++) {
    if (loclist[j].level == i) {
      std::int32_t &lx1 = loclist[j].lx1;
      std::int32_t &lx2 = loclist[j].lx2;
      std::int32_t &lx3 = loclist[j].lx3;
      std::fprintf(fp,"#MeshBlock %d on rank=%d with cost=%g\n", j, ranklist[j],
                   costlist[j]);
      std::fprintf(
          fp,"#  Logical level %d, location = (%" PRId32 " %" PRId32 " %" PRId32")\n",
          loclist[j].level, lx1, lx2, lx3);
      if (two_d) { // 2D
        std::fprintf(fp,"%g %g\n", size.h_view(j).x1min, size.h_view(j).x2min);
        std::fprintf(fp,"%g %g\n", size.h_view(j).x1max, size.h_view(j).x2min);
        std::fprintf(fp,"%g %g\n", size.h_view(j).x1max, size.h_view(j).x2max);
        std::fprintf(fp,"%g %g\n", size.h_view(j).x1min, size.h_view(j).x2max);
        std::fprintf(fp,"%g %g\n", size.h_view(j).x1min, size.h_view(j).x2min);
        std::fprintf(fp,"\n\n");
      }
      if (three_d) { // 3D
        Real &x1min = size.h_view(j).x1min, &x1max = size.h_view(j).x1max;
        Real &x2min = size.h_view(j).x2min, &x2max = size.h_view(j).x2max;
        Real &x3min = size.h_view(j).x3min, &x3max = size.h_view(j).x3max;
        std::fprintf(fp,"%g %g %g\n", x1min, x2min, x3min);
        std::fprintf(fp,"%g %g %g\n", x1max, x2min, x3min);
        std::fprintf(fp,"%g %g %g\n", x1max, x2max, x3min);
        std::fprintf(fp,"%g %g %g\n", x1min, x2max, x3min);
        std::fprintf(fp,"%g %g %g\n", x1min, x2min, x3min);
        std::fprintf(fp,"%g %g %g\n", x1min, x2min, x3max);
        std::fprintf(fp,"%g %g %g\n", x1max, x2min, x3max);
        std::fprintf(fp,"%g %g %g\n", x1max, x2min, x3min);
        std::fprintf(fp,"%g %g %g\n", x1max, x2min, x3max);
        std::fprintf(fp,"%g %g %g\n", x1max, x2max, x3max);
        std::fprintf(fp,"%g %g %g\n", x1max, x2max, x3min);
        std::fprintf(fp,"%g %g %g\n", x1max, x2max, x3max);
        std::fprintf(fp,"%g %g %g\n", x1min, x2max, x3max);
        std::fprintf(fp,"%g %g %g\n", x1min, x2max, x3min);
        std::fprintf(fp,"%g %g %g\n", x1min, x2max, x3max);
        std::fprintf(fp,"%g %g %g\n", x1min, x2min, x3max);
        std::fprintf(fp,"%g %g %g\n", x1min, x2min, x3min);
        std::fprintf(fp, "\n\n");
      }
    }
  }}
  std::fclose(fp);
  std::cout << "See the 'mesh_structure.dat' file for MeshBlock data" << std::endl;
  std::cout << "Use 'plot_mesh.py' script to visualize data" << std::endl << std::endl;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn GetBoundaryFlag(std::string input_string)
//  \brief Parses input string to return scoped enumerator flag specifying boundary
//  condition. Typically called in Mesh() ctor and in pgen/*.cpp files.

BoundaryFlag Mesh::GetBoundaryFlag(const std::string& input_string) 
{
  if (input_string == "reflect") {
    return BoundaryFlag::reflect;
  } else if (input_string == "outflow") {
    return BoundaryFlag::outflow;
  } else if (input_string == "user") {
    return BoundaryFlag::user;
  } else if (input_string == "periodic") {
    return BoundaryFlag::periodic;
  } else if (input_string == "undef") {
    return BoundaryFlag::undef;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Input string = '" << input_string << "' is an invalid boundary type"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn GetBoundaryString(BoundaryFlag input_flag)
//  \brief Parses enumerated type BoundaryFlag internal integer representation to return
//  string describing the boundary condition. Typicall used to format descriptive errors
//  or diagnostics. Inverse of GetBoundaryFlag().

std::string Mesh::GetBoundaryString(BoundaryFlag input_flag)
{
  switch (input_flag) {
    case BoundaryFlag::block:  // 0
      return "block";
    case BoundaryFlag::reflect:
      return "reflect";
    case BoundaryFlag::outflow:
      return "outflow";
    case BoundaryFlag::user:
      return "user";
    case BoundaryFlag::periodic:
      return "periodic";
    case BoundaryFlag::undef:
      return "undef";
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "Input enum class BoundaryFlag=" << static_cast<int>(input_flag)
         << " is an invalid boundary type" << std::endl;
      std::exit(EXIT_FAILURE);
      break;
  }
}

//----------------------------------------------------------------------------------------
//! \fn EnrollBoundaryFunction(BoundaryFace dir, BValFunc my_bc)
//! \brief Enroll a user-defined boundary function

void Mesh::EnrollBoundaryFunction(BoundaryFace dir, BoundaryFnPtr my_bc) {
  if (dir < 0 || dir > 5) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "EnrollBoundaryFunction called on bndry=" << dir << " not valid" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_bcs[dir] != BoundaryFlag::user) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Boundary condition flag must be set to 'user' in the <mesh> block in input"
        << " file to use user-enrolled BCs on bndry=" << dir << std::endl;
    std::exit(EXIT_FAILURE);
  }
  BoundaryFunc[static_cast<int>(dir)]=my_bc;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn CheckUserBoundaries()
//! \brief checks if user boundary functions are correctly enrolled
//! This compatibility check is performed in the Driver after calling ProblemGenerator()

void Mesh::CheckUserBoundaries() {
  for (int i=0; i<6; i++) {
    if (mesh_bcs[i] == BoundaryFlag::user) {
      if (BoundaryFunc[i] == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "User-defined boundary function is specified in input "
                  << " file but boundary function not enrolled; bndry=" << i << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
  return;
}

