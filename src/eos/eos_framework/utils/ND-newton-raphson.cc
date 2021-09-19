#ifndef __HH_NEWTON_ND
#define __HH_NEWTON_ND

#include <cmath>
#include <limits>
#include <stdio.h>

#include "LUP.cc"

// Uncomment to active linesearch
#define NDLINESEARCH

namespace NewtonRaphsonND {

using namespace LUP;

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) < (Y) ? (Y) : (X))

static constexpr double alp_conv = 1.0e-4;
static constexpr double lambda_min = 0.1;
static constexpr double lambda_max = 0.5;
static constexpr double NR_tol = 1.0e-15;
static constexpr double NR_rel_tol_x = std::numeric_limits<double>::epsilon();
static constexpr int NR_max_step_size = 100;
static constexpr int ls_iterations = 10;

// You need to implement these two functions
/*
inline static void NFunction(vec_ptr F,vec_ptr U){
F[0]=U[0]-0.25*cos(U[0])+0.25*sin(U[1]);
F[1]=U[1]-0.25*cos(U[0])+0.5*sin(U[1]);
}

inline static void NJacobian(matrix_ptr DF, vec_ptr U){
  DF[0][0]=1.0+0.25*sin(U[0]);
  DF[0][1]=0.25*cos(U[1]);
  DF[1][0]=0.25*sin(U[0]);
  DF[1][1]=1.0+0.5*cos(U[1]);
}
*/

template <typename T, size_t dim>
inline static double scalar_Product(vec_ptr<T, dim> &a, vec_ptr<T, dim> &b) {
  T sp{0.0};
  for (int i = 0; i < dim; i++)
    sp += a[i] * b[i];
  return sp;
}
template <class T, size_t dim> inline static double norm2(vec_ptr<T, dim> &a) {
  T sp{0.0};
  for (int i = 0; i < dim; i++)
    sp += a[i] * a[i];
  return sp;
}
template <typename T, size_t m, size_t n>
inline static void VectorMatrix_Multiply(vec_ptr<T, m> &x,
                                         matrix_ptr<T, n, m> &M,
                                         vec_ptr<T, n> &y) {
  for (int i = 0; i < n; i++) {
    y[i] = 0.0;
    for (int j = 0; j < m; j++)
      y[i] += M[i][j] * x[j];
  }
}
template <class T, size_t dim>
inline static void copy_Vector(vec_ptr<T, dim> &x, vec_ptr<T, dim> &y) {
  for (int i = 0; i < dim; i++)
    y[i] = x[i];
}
template <class T, size_t dim>
inline static void ScalarVector_Multiply(const T &l, vec_ptr<T, dim> &x) {
  for (int i = 0; i < dim; i++)
    x[i] = l * x[i];
}

template <class T, size_t dim>
inline static void add_Vectors(vec_ptr<T, dim> &x, const T &lambda,
                               vec_ptr<T, dim> &y, vec_ptr<T, dim> &z) {
  for (int i = 0; i < dim; i++)
    z[i] = x[i] + lambda * y[i];
}
template <class T, size_t dim>
inline static void add_Vectors(vec_ptr<T, dim> &x, vec_ptr<T, dim> &y,
                               vec_ptr<T, dim> &z) {
  for (int i = 0; i < dim; i++)
    z[i] = x[i] + y[i];
}

template <class T, size_t dim>
inline static void add_toVector(vec_ptr<T, dim> &x, vec_ptr<T, dim> &y) {
  for (int i = 0; i < dim; i++)
    x[i] = x[i] + y[i];
}
template <class T>
inline static bool check_convergence_LS(const T &g0, const T &g1,
                                        const T &gp1) {
  return (g1 - g0 - alp_conv * gp1 < 0.0);
}

template <class T, size_t dim>
inline static void enforce_physical_limits(vec_ptr<T, dim> &x0,
                                           vec_ptr<T, dim> &p) {
  // while((x0[0]+p[0]>1.0) || (x0[0]+p[0]<0.0))
  //  ScalarVector_Multiply(0.5,p,p);
}

template <class T, size_t dim, size_t iterations, typename F_t>
static void linesearch(matrix_ptr<T, dim, dim> &DF, vec_ptr<T, dim> &x0,
                       vec_ptr<T, dim> &p, vec_ptr<T, dim> &F0,
                       F_t &NFunction) {

  vec_ptr<T, dim> dF0;
  vec_ptr<T, dim> DF1;
  vec_ptr<T, dim> xtmp, delta_x;

  add_Vectors<T, dim>(x0, p, xtmp);
  auto F = NFunction(xtmp);

  auto g0 = 0.5 * norm2<T, dim>(F0);
  auto g1 = 0.5 * norm2<T, dim>(F);

  double lambda = 1.0;

  VectorMatrix_Multiply<T, dim, dim>(F0, DF, dF0);
  auto gp0 = scalar_Product<T, dim>(dF0, p);
  bool converged = check_convergence_LS(g0, g1, gp0);

  if (converged)
    return;

  // Reduce stepsize by doing a quadratic fit

  lambda = MAX(lambda_min, -0.5 * gp0 / (g1 - g0 - gp0));

  add_Vectors<T, dim>(x0, lambda, p, xtmp);
  F = NFunction(xtmp);

  auto gp1 = lambda * scalar_Product<T, dim>(dF0, p);

  auto g2 = 0.5 * norm2<T, dim>(F);
  converged = check_convergence_LS(g0, g2, gp1);

  if (converged) {
    ScalarVector_Multiply<T, dim>(lambda, p);
    return;
  }
  double lambda1 = 1.0;
  double lambda2 = lambda;

  for (int it = 0; it < iterations; it++) {
    auto a = g1 - gp0 * lambda1 - g0;
    auto b = g2 - gp0 * lambda2 - g0;

    auto a_tmp = a / (lambda1 * lambda1) - b / (lambda2 * lambda2);
    b = -lambda2 * a / (lambda1 * lambda1) + b * lambda1 / (lambda2 * lambda2);
    a = a_tmp / (lambda1 - lambda2);
    b /= (lambda1 - lambda2);
    if (fabs(a) < 1.0e-32) {
      auto tmp = -(b + std::copysign(1.0, b) * sqrt(b * b - 3.0 * a * gp0));
      lambda = (b < 0) ? tmp / (3.0 * a) : gp0 / tmp;
    } else {
      lambda = -0.5 * gp0 / b;
    }
    lambda = MIN(lambda_max * lambda1, MAX(lambda_min * lambda1, lambda));
    // Check convergence criterion
    add_Vectors<T, dim>(x0, lambda, p, xtmp);
    F = NFunction(xtmp);

    auto gp2 = lambda * scalar_Product<T, dim>(dF0, p);
    g1 = g2;
    g2 = 0.5 * norm2<T, dim>(F);
    converged = check_convergence_LS(g0, g2, gp2);

    lambda1 = lambda2;
    lambda2 = lambda;

    if (converged) {
      ScalarVector_Multiply<T, dim>(lambda, p);
      return;
    }
  }
  return;
}

template <class T, int dim>
inline static bool check_NRaphson_convergence(vec_ptr<T, dim> &F,
                                              vec_ptr<T, dim> &x,
                                              vec_ptr<T, dim> &x0) {
  double norm = 0.0;
  for (int i = 0; i < dim; i++)
    if (norm < fabs(F[i]))
      norm = fabs(F[i]);
  if (norm < NR_tol) {
     //printf("Norm: %.15f \n",norm);
    return true;
  }
  // Check also relative convergence in x
  norm = 0.0;
  for (int i = 0; i < dim; i++) {
    double rel = (fabs(x[i] - x0[i])) / MAX(fabs(x[i]), 1.0);
    if (rel > norm)
      norm = rel;
  }
  if (norm < NR_rel_tol_x)
    return true;
  return false;
}

template <class T, int dim>
inline static void adjust_step_size(vec_ptr<T, dim> &x, const double &maxstep) {
  double stepsize = sqrt(scalar_Product<T, dim>(x, x));
  if (stepsize - maxstep > 0.0)
    ScalarVector_Multiply<T, dim>(maxstep / stepsize, x);
}

template <typename T, int dim>
inline static void Print_vars(vec_ptr<T, dim> U, vec_ptr<T, dim> F,
                              const int it) {
  std::cout << "**********************************" << std::endl;
  std::cout << "Iteration: " << it << std::endl;
  for (int i = 0; i < dim; i++)
    std::cout << "U[" << i << "]=" << U[i] << std::endl;
  for (int i = 0; i < dim; i++)
    std::cout << "F[" << i << "]=" << F[i] << std::endl;
}

template <class T, int dim, int iterations, typename F_t, typename DF_t>
static int NRaphson(vec_ptr<T, dim> &F, matrix_ptr<T, dim, dim> &DF,
                    vec_ptr<T, dim> &x, vec_ptr<T, dim> &p,
                    matrix_ptr<T, dim, dim> &LU, vec_ptr<T, dim> &LU_rhs,
                    vec_ptr<T, dim> &LU_dx, int_ptr<dim> &LU_index,
                    F_t &NFunction, DF_t &NJacobian) {
  /*
   * Newton Raphson main routine
   * The return value classify the various errors.
   * Convergence : 0
   * Failed to converge: 1
   * Inversion error in LU: 2
   * Spurious convergence to a min of 0.5*FF: 3
   */
  vec_ptr<T, dim> x0;
  for (int i = 0; i < dim; i++)
    x0[i] = 1.0e6 * x[i];

  F = NFunction(x);
  DF = NJacobian(NFunction, x);
  bool converged = check_NRaphson_convergence<T, dim>(F, x, x0);
  int polish = 0;
  double max_stepsize =
      NR_max_step_size * MAX(sqrt(scalar_Product<T, dim>(x, x)), dim);

  if (converged)
    return 0;

  for (int it = 0; it < iterations; it++) {
    ScalarVector_Multiply<T, dim>(-1.0, F);

#ifdef DEBUG
    Print_vars<T, dim>(x, F, it);
    for (int ii = 0; ii < dim; ++ii)
      for (int jj = 0; jj < dim; ++jj)
        std::cout << "DF[" << ii << "][" << jj << "]= " << DF[ii][jj]
                  << std::endl;
#endif

    int ierr = LUdecompose<T, dim>(DF, LU, LU_index);
    if (ierr != 0)
      return 2;
    LUsolve<T, dim>(LU, F, p, LU_index);
    LUpolish<T, dim, polish_iterations>(DF, LU, p, F, LU_rhs, LU_dx, LU_index);
    ScalarVector_Multiply<T, dim>(-1.0, F);
    adjust_step_size<T, dim>(x, max_stepsize);
#ifdef NDLINESEARCH
    linesearch<T, dim, ls_iterations, F_t>(DF, x, p, F, NFunction);
#endif
    enforce_physical_limits<T, dim>(x, p);

    copy_Vector<T, dim>(x, x0);
    add_toVector<T, dim>(x, p);

    F = NFunction(x);
    DF = NJacobian(NFunction, x);

    converged = check_NRaphson_convergence<T, dim>(F, x, x0);
    if (converged) {
#ifdef DEBUG
      std::cout << "NR iterations: " << it + 1 << std::endl;
      Print_vars<T, dim>(x, F, it);
#endif
      return 0;
    }
  }
  return 1;
}

template <typename T, int dim, typename F_t>
static inline matrix_ptr<T, dim, dim> NJacobian(F_t &Func, vec_ptr<T, dim> U) {
  constexpr double eps = 0.5e-11;

  matrix_ptr<T, dim, dim> DF;
  // Compute using finite differencing
  for (int i = 0; i < dim; ++i) {
    vec_ptr<T, dim> h;
    for (int k = 0; k < dim; k++) {
      h[k] = 0.0;
    }
    h[i] = (std::fabs(U[i]) < 2.0e-15) ? eps : eps * (U[i]);
    vec_ptr<T, dim> Uh;

    add_Vectors<T, dim>(U, h, Uh);
    // Apparently improves perfomance if not optimised away by compiler
    add_Vectors<T, dim>(Uh, -1.0, U, h);

    auto F = Func(U);
    auto Fh = Func(Uh);

    vec_ptr<T, dim> dFdU;
    add_Vectors<T, dim>(Fh, -1.0, F, dFdU);
    ScalarVector_Multiply<T, dim>(1.0 / h[i], dFdU);
    for (int j = 0; j < dim; ++j) {
      DF[j][i] = dFdU[j];
    }
  }

  return DF;
}

template <class T, int dim, typename F_t>
static vec_ptr<T, dim> Solve(vec_ptr<T, dim> &X0, F_t &Func, int &error) {

  matrix_ptr<T, dim, dim> LU, DF;

  int_ptr<dim> index;

  vec_ptr<T, dim> rhs, dx, F, x_sol, p;

  x_sol = X0;

  error = NRaphson<T, dim, 300, F_t>(F, DF, x_sol, p, LU, rhs, dx, index, Func,
                                     NJacobian<T, dim, F_t>);

  return x_sol;
}

template <class T, int dim, typename F_t, typename NJ_t>
static vec_ptr<T, dim> Solve(vec_ptr<T, dim> &X0, F_t &Func, NJ_t &NJacobian,
                             int &error) {

  matrix_ptr<T, dim, dim> LU, DF;

  int_ptr<dim> index;

  vec_ptr<T, dim> rhs, dx, F, x_sol, p;

  x_sol = X0;

  error = NRaphson<T, dim, 300, F_t>(F, DF, x_sol, p, LU, rhs, dx, index, Func,
                                     NJacobian);

  return x_sol;
}
}

#endif
