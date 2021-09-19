/*
 * =====================================================================================
 *
 *       Filename:  NewtonRaphson.hh
 *
 *    Description:  General Newton Raphson
 *
 *        Version:  1.0
 *        Created:  02/08/2017 11:39:29
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Elias Roland Most (ERM), emost@itp.uni-frankfurt.de
 *   Organization:  Goethe University Frankfurt
 *
 * =====================================================================================
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#ifndef __HH_NEWTON_NEW__
#define __HH_NEWTON_NEW__

#include "LUP.cc"

template <typename T, size_t dim>
class LUPSolve {
  public:
  using vec = std::array<T, dim>;
  using matrix = std::array<std::array<T, dim>, dim>;

 private:

  matrix LU, DF;
  typename LUP::int_ptr<dim> LU_index;
  vec LU_rhs, LU_dx;

 public:
  inline size_t Solve(matrix &A, vec &x, vec &b) {
    using namespace LUP;
    auto ierr = LUdecompose<T, dim>(A, LU, LU_index);
    if (ierr != 0) return 2;
    LUsolve<T, dim>(LU, b, x, LU_index);
    LUpolish<T, dim>(A, LU, x, b, LU_rhs, LU_dx, LU_index);
    return 0;
  }
};

template <typename T, size_t dim>
class CramersSolve {
  public:
  using vec = std::array<T, dim>;
  using matrix = std::array<std::array<T, dim>, dim>;

 private:

 public:
  static inline size_t Solve(matrix &A, vec &x, vec &b) {
    assert(!"Not implemented for this dimension yet!");
  }
};

template <typename T>
class CramersSolve<T, 2> {
  public:
  static constexpr size_t dim = 2;
  using vec = std::array<T, dim>;
  using matrix = std::array<std::array<T, dim>, dim>;


  static inline size_t Solve(matrix &A, vec &x, vec &b) {
    const auto det = A[0][0] * A[1][1] - A[1][0] * A[0][1];

    x[0] = (b[0] * A[1][1] - b[1] * A[0][1]) / det;
    x[1] = (A[0][0] * b[1] - A[1][0] * b[0]) / det;

    return (det != 0) ? 0 : 2;
  }
};

template <typename T>
class CramersSolve<T, 3> {
  public:
  static constexpr size_t dim = 3;
  using vec = std::array<T, dim>;
  using matrix = std::array<std::array<T, dim>, dim>;

  static inline size_t Solve(matrix &A, vec &x, vec &b) {
    const auto det = A[0][0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
                     A[0][2] * A[1][0] * A[2][1] - A[2][0] * A[1][1] * A[0][2] -
                     A[2][1] * A[1][2] * A[0][0] - A[2][2] * A[1][0] * A[0][1];

    x[0] = b[0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] * b[2] +
           A[0][2] * b[1] * A[2][1] - b[2] * A[1][1] * A[0][2] -
           A[2][1] * A[1][2] * b[0] - A[2][2] * b[1] * A[0][1];

    x[1] = A[0][0] * b[1] * A[2][2] + b[0] * A[1][2] * A[2][0] +
           A[0][2] * A[1][0] * b[2] - A[2][0] * b[1] * A[0][2] -
           b[2] * A[1][2] * A[0][0] - A[2][2] * A[1][0] * b[0];

    x[2] = A[0][0] * A[1][1] * b[2] + A[0][1] * b[1] * A[2][0] +
           b[0] * A[1][0] * A[2][1] - A[2][0] * A[1][1] * b[0] -
           A[2][1] * b[1] * A[0][0] - b[2] * A[1][0] * A[0][1];

    x[0] /= det;
    x[1] /= det;
    x[2] /= det;

    return (det != 0) ? 0 : 2;
  }
};

template<typename vec>
struct NR_policy_std{

  public:
    using T = typename vec::value_type;
  static constexpr double NR_tol = 1.0e-13;
  static constexpr size_t NR_max_step_size = 100;
  static constexpr size_t ls_iterations = 10;
  static constexpr auto NR_rel_tol_x = std::numeric_limits<T>::epsilon();

  inline static bool check_convergence(vec &F, vec &x, vec &x0) {
    double norm = 0.0;
    for (int i = 0; i < x.size(); i++)
      if (norm < std::fabs(F[i])) norm = std::fabs(F[i]);
    if (norm < NR_tol) {
      // printf("Norm: %.15f \n",norm);
      return true;
    }
    // Check also relative convergence in x
    norm = 0.0;
#pragma unroll
    for (int i = 0; i < x.size(); i++) {
      double rel = (std::fabs(x[i] - x0[i])) / std::max(std::fabs(x[i]), 1.0);
      if (rel > norm) norm = rel;
    }
    if (norm < NR_rel_tol_x) return true;
    return false;
  };

  inline static void adjust_step_size(vec &x, const double &maxstep) {
    auto stepsize = T{0};
    for(auto &xi : x) stepsize += xi*xi; 
    stepsize = std::sqrt(stepsize);
    if (stepsize - maxstep > 0.0){ 
     	for(auto &xi : x) xi *= maxstep/stepsize; 
    };
  };
};

template <typename T, size_t dim, template <typename, size_t> class LinearSolve,
          bool LineSearch, template <typename> class NR_policy_t = NR_policy_std>
class NewtonRaphson {
 public:
  using vec = std::array<T, dim>;
  using matrix = std::array<std::array<T, dim>, dim>;
  using NR_policy = NR_policy_t<vec>;

 private:
  static constexpr double alp_conv = 1.0e-4;
  static constexpr double lambda_min_gl = 0.1;
  static constexpr double lambda_max = 0.5;
  static constexpr double NR_tol = NR_policy::NR_tol;
  static constexpr size_t NR_max_step_size = NR_policy::NR_max_step_size;
  static constexpr size_t ls_iterations = NR_policy::ls_iterations;

  inline static T scalar_Product(vec &a, vec &b) {
    T sp{0.0};
#pragma unroll
    for (int i = 0; i < dim; i++) sp += a[i] * b[i];
    return sp;
  }

  inline static T norm2(vec &a) {
    T sp{0.0};
#pragma unroll
    for (int i = 0; i < dim; i++) sp += a[i] * a[i];
    return sp;
  }

  inline static void VectorMatrix_Multiply(vec &x, matrix &M, vec &y) {
#pragma unroll
    for (int i = 0; i < dim; i++) {
      y[i] = 0.0;
#pragma unroll
      for (int j = 0; j < dim; j++) y[i] += M[i][j] * x[j];
    }
  }

  inline static void copy_Vector(vec &x, vec &y) {
#pragma unroll
    for (int i = 0; i < dim; i++) y[i] = x[i];
  }

  inline static void ScalarVector_Multiply(const T &l, vec &x) {
#pragma unroll
    for (int i = 0; i < dim; i++) x[i] = l * x[i];
  }

  inline static void add_Vectors(vec &x, const T &lambda, vec &y, vec &z) {
#pragma unroll
    for (int i = 0; i < dim; i++) z[i] = x[i] + lambda * y[i];
  }

  inline static void add_Vectors(vec &x, vec &y, vec &z) {
#pragma unroll
    for (int i = 0; i < dim; i++) z[i] = x[i] + y[i];
  }

  inline static void add_toVector(vec &x, vec &y) {
#pragma unroll
    for (int i = 0; i < dim; i++) x[i] = x[i] + y[i];
  }

  inline static bool check_NRaphson_convergence(vec &F, vec &x, vec &x0) {
    return NR_policy::check_convergence(F,x,x0);
  }

  inline static void Print_vars(vec &U, vec &F, const int it) {
    std::cout << "**********************************" << std::endl;
    std::cout << "Iteration: " << it << std::endl;
#pragma unroll
    for (int i = 0; i < dim; i++)
      std::cout << "U[" << i << "]=" << U[i] << std::endl;
#pragma unroll
    for (int i = 0; i < dim; i++)
      std::cout << "F[" << i << "]=" << F[i] << std::endl;
  }

  inline static void adjust_step_size(vec &x, const double &maxstep) {
    NR_policy::adjust_step_size(x,maxstep); 
  }

  inline static bool check_convergence_LS(const T &g0, const T &g1,
                                          const T &gp1) {
    return (g1 - g0 - alp_conv * gp1 < 0.0);
  }

  template <typename F_t>
  static void linesearch(matrix &DF, vec &x0, vec &p, vec &F0, F_t &NFunction) {
    vec dF0, DF1, xtmp, delta_x;

    auto const lambda_min = lambda_min_gl;

    add_Vectors(x0, p, xtmp);
    auto F = NFunction(xtmp);

    auto g0 = 0.5 * norm2(F0);
    auto g1 = 0.5 * norm2(F);

    T lambda{1.0};

    VectorMatrix_Multiply(F0, DF, dF0);
    auto gp0 = scalar_Product(dF0, p);
    bool converged = check_convergence_LS(g0, g1, gp0);

    if (converged) return;

    // Reduce stepsize by doing a quadratic fit

    lambda = std::max(lambda_min, -0.5 * gp0 / (g1 - g0 - gp0));

    add_Vectors(x0, lambda, p, xtmp);
    F = NFunction(xtmp);

    auto gp1 = lambda * scalar_Product(dF0, p);

    auto g2 = 0.5 * norm2(F);
    converged = check_convergence_LS(g0, g2, gp1);

    if (converged) {
      ScalarVector_Multiply(lambda, p);
      return;
    }
    double lambda1 = 1.0;
    double lambda2 = lambda;

    for (int it = 0; it < ls_iterations; it++) {
      auto a = g1 - gp0 * lambda1 - g0;
      auto b = g2 - gp0 * lambda2 - g0;

      auto a_tmp = a / (lambda1 * lambda1) - b / (lambda2 * lambda2);
      b = -lambda2 * a / (lambda1 * lambda1) +
          b * lambda1 / (lambda2 * lambda2);
      a = a_tmp / (lambda1 - lambda2);
      b /= (lambda1 - lambda2);
      if (std::fabs(a) < 1.0e-32) {
        auto tmp = -(b + std::copysign(1.0, b) * sqrt(b * b - 3.0 * a * gp0));
        lambda = (b < 0) ? tmp / (3.0 * a) : gp0 / tmp;
      } else {
        lambda = -0.5 * gp0 / b;
      }
      lambda = std::min(lambda_max * lambda1,
                        std::max(lambda_min * lambda1, lambda));
      // Check convergence criterion
      add_Vectors(x0, lambda, p, xtmp);
      F = NFunction(xtmp);

      auto gp2 = lambda * scalar_Product(dF0, p);
      g1 = g2;
      g2 = 0.5 * norm2(F);
      converged = check_convergence_LS(g0, g2, gp2);

      lambda1 = lambda2;
      lambda2 = lambda;

      if (converged) {
        ScalarVector_Multiply(lambda, p);
        return;
      }
    }
    return;
  }

 public:
  template <typename F_t>
  static inline matrix FDJacobian(F_t &Func, vec &U) {
    constexpr double eps = 0.5e-11;

    auto F = Func(U);
    matrix DF;
// Compute using finite differencing
#pragma unroll
    for (int i = 0; i < dim; ++i) {
      vec h;
#pragma unroll
      for (int k = 0; k < dim; k++) {
        h[k] = 0.0;
      }
      h[i] = (std::fabs(U[i]) < 2.0e-15) ? eps : eps * (U[i]);
      vec Uh;

      add_Vectors(U, h, Uh);
      // Apparently improves perfomance if not optimised away by compiler
      add_Vectors(Uh, -1.0, U, h);

      auto Fh = Func(Uh);

      vec dFdU;
      add_Vectors(Fh, -1.0, F, dFdU);
      ScalarVector_Multiply(1.0 / h[i], dFdU);
#pragma unroll
      for (int j = 0; j < dim; ++j) {
        DF[j][i] = dFdU[j];
      }
    }

    return DF;
  }

  template <typename DF_t, typename F_t>
  static size_t Solve(F_t &NFunction, DF_t &NJacobian, vec &x,
                      size_t const iterations) {
    LinearSolve<T, dim> LS;

    vec x0;
#pragma unroll
    for (int i = 0; i < dim; i++) x0[i] = 1.0e6 * x[i];

    auto F = NFunction(x);
    auto DF = NJacobian(NFunction, x);
    bool converged = check_NRaphson_convergence(F, x, x0);
    int polish = 0;
    double max_stepsize =
        NR_max_step_size *
        std::max(sqrt(scalar_Product(x, x)), static_cast<T>(dim));

    if (converged) return 0;

    for (int it = 0; it < iterations; it++) {
      ScalarVector_Multiply(-1.0, F);

#ifdef DEBUG_NEWTON
      Print_vars(x, F, it);
      for (int ii = 0; ii < dim; ++ii)
        for (int jj = 0; jj < dim; ++jj)
          std::cout << "DF[" << ii << "][" << jj << "]= " << DF[ii][jj]
                    << std::endl;
#endif
      vec p;
      if (dim > 1) {
        LS.Solve(DF, p, F);
      } else {
        p[0] = F[0] / DF[0][0];
      }

      ScalarVector_Multiply(-1.0, F);
      adjust_step_size(x, max_stepsize);
      if (LineSearch) {
        linesearch<F_t>(DF, x, p, F, NFunction);
      }

      copy_Vector(x, x0);
      add_toVector(x, p);

      F = NFunction(x);
      DF = NJacobian(NFunction, x);

      converged = check_NRaphson_convergence(F, x, x0);
      if (converged) {
#ifdef DEBUG_NEWTON
        std::cout << "NR iterations: " << it + 1 << std::endl;
        Print_vars(x, F, it);
#endif
        return 0;
      }
    }
    return 1;
  }

  template <typename F_t>
  inline static size_t Solve(F_t &NFunction, vec &x, size_t const iterations) {
    return Solve(NFunction, FDJacobian<F_t>, x, iterations);
  }
};

#endif
