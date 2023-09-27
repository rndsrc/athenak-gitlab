#ifndef ATHENA_RADIATION_FEMN_GEODESIC_GRID_MATRICES_HPP
#define ATHENA_RADIATION_FEMN_GEODESIC_GRID_MATRICES_HPP

//========================================================================================
// GR radiation code for AthenaK with FEM_N & FP_N
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn.hpp
//  \brief Matrices for the FEM_N and FP_N schemes for GR Boltzmann transport

#include "athena.hpp"

namespace radiationfemn {

// ---------------------------------------------------------------------------
// Quadrature points and weights (radiation_femn_geodesic_grid_quadrature.cpp)
void LoadQuadrature(std::string &scheme_name,
                    int &scheme_num_points,
                    HostArray1D<Real> &scheme_weights,
                    HostArray2D<Real> &scheme_points);

// ---------------------------------------------------------------------
// Geodesic grid generation functions (radiation_femn_geodesic_grid.cpp)
void
GeodesicGridBaseGenerate(int &geogrid_level,
                         int &geogrid_num_points,
                         int &geogrid_num_edges,
                         int &geogrid_num_triangles,
                         HostArray1D<Real> &x,
                         HostArray1D<Real> &y,
                         HostArray1D<Real> &z,
                         HostArray1D<Real> &r,
                         HostArray1D<Real> &theta,
                         HostArray1D<Real> &phi,
                         HostArray2D<int> &edges,
                         HostArray2D<int> &triangles);    // Generate base geodesic grid
void GeodesicGridRefine(int &geogrid_level,
                        int &geogrid_num_points,
                        int &geogrid_num_edges,
                        int &geogrid_num_triangles,
                        HostArray1D<Real> &x,
                        HostArray1D<Real> &y,
                        HostArray1D<Real> &z,
                        HostArray1D<Real> &r,
                        HostArray1D<Real> &theta,
                        HostArray1D<Real> &phi,
                        HostArray2D<int> &edges,
                        HostArray2D<int> &triangles);          // Refine geodesic grid by one level

int FindEdgesIndex(int e1, int e2, HostArray2D<int> &edges);  // Given two edge indices, find

// ---------------------------------------------------------------------------
// basis functions & helper functions (radiation_femn_geodesic_grid_basis.cpp)
void BarycentricToCartesian(Real x1,
                            Real y1,
                            Real z1,
                            Real x2,
                            Real y2,
                            Real z2,
                            Real x3,
                            Real y3,
                            Real z3,
                            Real xi1,
                            Real xi2,
                            Real xi3,
                            Real &xval,
                            Real &yval,
                            Real &zval);

// Type 1: 'Overlapping tent' (Default FEM_N choice)
Real FEMBasis1Type1(Real xi1, Real xi2, Real xi3);
Real FEMBasis2Type1(Real xi1, Real xi2, Real xi3);
Real FEMBasis3Type1(Real xi1, Real xi2, Real xi3);

// Type 2: 'Non-overlapping tent'
Real FEMBasis1Type2(Real xi1, Real xi2, Real xi3);
Real FEMBasis2Type2(Real xi1, Real xi2, Real xi3);
Real FEMBasis3Type2(Real xi1, Real xi2, Real xi3);

// Type 3: 'Overlapping honeycomb'
Real FEMBasis1Type3(Real xi1, Real xi2, Real xi3);
Real FEMBasis2Type3(Real xi1, Real xi2, Real xi3);
Real FEMBasis3Type3(Real xi1, Real xi2, Real xi3);

// Type 4: 'Non-overlapping honeycomb' (S_N choice)
Real FEMBasis1Type4(Real xi1, Real xi2, Real xi3);
Real FEMBasis2Type4(Real xi1, Real xi2, Real xi3);
Real FEMBasis3Type4(Real xi1, Real xi2, Real xi3);

// FEM basis, pick from type
Real FEMBasis(Real xi1, Real xi2, Real xi3, int basis_index, int basis_choice);

// FPN basis
Real FPNBasis(int l, int m, Real phi, Real theta);
Real dFPNBasisdOmega(int l, int m, Real phi, Real theta, int var_index);
Real PtildehatJac(Real phi, Real theta, int tilde_index, int hat_index);

// some other useful functions
Real FEMBasisABasisB(int a, int b, int t1, int t2, int t3, Real xi1, Real xi2, Real xi3, int basis_choice);
Real FEMBasisA(int a, int t1, int t2, int t3, Real xi1, Real xi2, Real xi3, int basis_choice);
Real dFEMBasisdxi(Real xi1, Real xi2, Real xi3, int basis_index, int basis_choice, int xi_index);

Real PdFEMBasisdOmega(int ihat, int a, int t1, int t2, int t3, Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3,
                      Real xi1, Real xi2, Real xi3, int basis_choice);

Real CosPhiSinTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3);
Real SinPhiSinTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3);
Real CosTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3);
Real SinPhiCosecTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3);
Real CosPhiCosTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3);
Real CosPhiCosecTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3);
Real SinPhiCosTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3);
Real SinTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3);
Real MomentumUnitEnergy(int mu, Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3);
Real MomentumUnitEnergy(int mu, Real phi, Real theta);
// Find triangles which share an edge
void FindTriangles(int a, int b, const HostArray2D<int> &triangles, HostArray2D<int> &edge_triangles, bool &is_edge);

// -------------------------------------------------------------------------------------------------------------------------------------------
// Integration routines over geodesic grid (radiation_femn_geodesic_grid_matrix_integrate.cpp and radiation_femn_geodesic_grid_quadrature.cpp)
Real CalculateDeterminantJacobian(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3);
Real IntegrateMatrixSphericalTriangle(int a, int b, int basis, int t1, int t2, int t3, const HostArray1D<Real> &x, const HostArray1D<Real> &y, const HostArray1D<Real> &z,
                                        const HostArray1D<Real> &scheme_weights, const HostArray2D<Real> &scheme_points, int matrixnumber, int nu = -42, int mu = -42,
                                        int ihat = -42);
Real IntegrateMatrixFEMN(int a, int b, int basis, const HostArray1D<Real> &x, const HostArray1D<Real> &y, const HostArray1D<Real> &z, const HostArray1D<Real> &scheme_weights,
                         const HostArray2D<Real> &scheme_points, const HostArray2D<int> &triangles, int matrixchoice, int nu, int mu, int ihat);
Real IntegrateMatrixFPN(int la, int ma, int lb, int mb, const HostArray1D<Real> &scheme_weights, const HostArray2D<Real> &scheme_points,
                          int matrixchoice, int nu, int mu, int ihat);
} // namespace radiationfemn

#endif //ATHENA_RADIATION_FEMN_GEODESIC_GRID_MATRICES_HPP