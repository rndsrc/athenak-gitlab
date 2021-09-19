/*
 * =====================================================================================
 *
 *       Filename:  spline.hh
 *
 *    Description:  Cubic spline interpolation
 *
 *        Version:  1.0
 *        Created:  01/05/2017 00:14:10
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Elias Roland Most (ERM), most@fias.uni-frankfurt.de
 *   Organization:  Goethe University Frankfurt
 *
 * =====================================================================================
 */

#ifndef spline_INC
#define spline_INC

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include "LUP_old.cc"

template <typename T, int num_vars>
class cubic_spline_t {
 public:
  template <int N>
  using vec = std::array<T, N>;

  using vec_ptr = std::unique_ptr<T[]>;

  int nvars = num_vars;

  T minval;
  T maxval;

 private:
  vec_ptr x;
  std::array<vec_ptr, num_vars> y;
  vec_ptr deltax;
  std::array<vec_ptr, num_vars> ypp;

  int num_points;

  inline double const &operator[](const int i) { return x[i]; }

  inline int size() { return num_points; }

  inline void compute_matrix(LUP_old::matrix_ptr<T> M) {
    for (int i = 0; i < num_points - 2; ++i) {
      for (int j = 0; j < num_points - 2; ++j) {
        M[i][j] = 0;
      }
    }

    for (int i = 0; i < num_points - 2; ++i) {
      if (i > 0) M[i][i - 1] = deltax[i] / 6.;

      M[i][i] = (x[i + 2] - x[i]) / 3.;
      if (i + 1 < num_points - 2) M[i][i + 1] = deltax[i + 1] / 6.;
    }
  };

  inline void compute_rhs(int &N, T rhs[]) {
    for (int i = 0; i < num_points - 2; ++i) {
      rhs[i] = (y[N][i + 2] - y[N][i + 1]) / deltax[i + 1] -
               (y[N][i + 1] - y[N][i]) / deltax[i];
    }
  }

  inline void compute_ypp() {
    // We need some legacy style to use the old LUP routine...
    // This should be upgraded to proper modern C++11
    const int dim = num_points - 2;
    T *__restrict__ A[dim];
    T *__restrict__ LU[dim];

    for (int i = 0; i < dim; i++) {
      A[i] = new T[dim];
      LU[i] = new T[dim];
    }

    compute_matrix(A);

    int index[num_points - 2];
    T x[num_points - 2], dx[num_points - 2], rhs[num_points - 2];

    auto ierr = LUP_old::LUdecompose<T>(dim, A, LU, index);
    assert(ierr == 0);

    T b[num_points - 2];
    for (int i = 0; i < num_vars; ++i) {
      compute_rhs(i, b);
      LUP_old::LUsolve<T>(dim, LU, b, x, index);

      LUP_old::LUpolish<T, 2>(dim, A, LU, x, b, rhs, dx, index);

      // Natural boundaries
      ypp[i][0] = ypp[i][num_points - 1] = 0;
      for (int index = 0; index < num_points - 2; ++index)
        ypp[i][index + 1] = x[index];
    }

    for (int i = 0; i < num_points - 2; i++) {
      delete A[i];
      delete LU[i];
    }
  }

  inline int find_index(const T &xin) const {
    int lower = 0;
    int upper = num_points - 1;
    // simple bisection should do it
    while (upper - lower > 1) {
      int tmp = lower + (upper - lower) / 2;
      if (xin < x[tmp])
        upper = tmp;
      else
        lower = tmp;
    }
    return lower;
  }

  inline std::array<T, 4> interpolation_coefficients(const T &xin,
                                                     int &index) const {
    index = find_index(xin);
    std::array<T, 4> coeffs;
    coeffs[0] = (x[index + 1] - xin) / deltax[index];
    coeffs[1] = (xin - x[index]) / deltax[index];
    coeffs[2] = (coeffs[0] * coeffs[0] - 1.) *
                ((x[index + 1] - xin) * deltax[index]) / 6.;
    coeffs[3] =
        (coeffs[1] * coeffs[1] - 1.) * ((xin - x[index]) * deltax[index]) / 6.;

    return coeffs;
  }

 public:
  template <int... vals>
  inline vec<sizeof...(vals)> interpolate(const T &xin) const {
    constexpr int N = sizeof...(vals);
    static_assert(N <= num_vars,
                  "Cannot interpolate more quantities than num_vars");

    int index;
    const auto coeffs = interpolation_coefficients(xin, index);

    vec<N> V{};

    int m = 0;
    int tmp[] = {
        (V[m] = coeffs[0] * y[vals][index] + coeffs[1] * y[vals][index + 1] +
                coeffs[2] * ypp[vals][index] + coeffs[3] * ypp[vals][index + 1],
         ++m)...};
    (void)tmp;
    return V;
  }

  template <typename... TArgs>
  inline vec<num_vars> interpolate(const T &xin, const TArgs &... vals) const {
    constexpr int N = sizeof...(TArgs);
    static_assert(N <= num_vars,
                  "Cannot interpolate more quantities than num_vars");

    int index;
    const auto coeffs = interpolation_coefficients(xin, index);

    vec<num_vars> V{};

    int m = 0;
    int tmp[] = {(V[vals] = coeffs[0] * y[vals][index] +
                            coeffs[1] * y[vals][index + 1] +
                            coeffs[2] * ypp[vals][index] +
                            coeffs[3] * ypp[vals][index + 1],
                  ++m)...};
    (void)tmp;
    return V;
  }

  template <typename... TArgs>
  inline vec<num_vars> interpolate_strict(const T &xin,
                                          const TArgs &... vals) const {
    constexpr int N = sizeof...(TArgs);
    static_assert(N <= num_vars,
                  "Cannot interpolate more quantities than num_vars");

    auto xinL = std::min(x[num_points - 1], std::max(x[0], xin));
    int index;
    const auto coeffs = interpolation_coefficients(xinL, index);

    vec<num_vars> V{};

    int m = 0;
    int tmp[] = {(V[vals] = coeffs[0] * y[vals][index] +
                            coeffs[1] * y[vals][index + 1] +
                            coeffs[2] * ypp[vals][index] +
                            coeffs[3] * ypp[vals][index + 1],
                  ++m)...};
    (void)tmp;
    return V;
  }

  inline vec<num_vars> interpolate_all(const T &xin) const {
    int index;
    const auto coeffs = interpolation_coefficients(xin, index);
    vec<num_vars> V{};

    for (int vals = 0; vals < num_vars; ++vals) {
      V[vals] = coeffs[0] * y[vals][index] + coeffs[1] * y[vals][index + 1] +
                coeffs[2] * ypp[vals][index] + coeffs[3] * ypp[vals][index + 1];
    }

    return V;
  }

  template <typename... TArgs>
  inline vec<num_vars> get_derivative(const T &xin,
                                      const TArgs &... vals) const {
    constexpr int N = sizeof...(vals);
    static_assert(N <= num_vars,
                  "Cannot interpolate more quantities than num_vars");

    const auto index = find_index(xin);

    const auto A = (x[index + 1] - xin) / deltax[index];
    const auto B = (xin - x[index]) / deltax[index];

    vec<num_vars> V{};
    int m = 0;
    int tmp[] = {(
        V[vals] = (y[vals][index + 1] - y[vals][index]) / (deltax[index]) -
                  (3. * A * A - 1.) / 6. * deltax[index] * ypp[vals][index] +
                  (3. * B * B - 1.) / 6. * deltax[index] * ypp[vals][index + 1],
        ++m)...};
    (void)tmp;
    return V;
  }

  template <typename... Args>
  cubic_spline_t(int __num_points, vec_ptr &&__x, Args &&... args)
      : x(std::move(__x)), num_points(__num_points) {
    static_assert(sizeof...(args) == num_vars,
                  "Need exactly num_vars arguments");
    int n = 0;
    int tmp[] = {(y[n] = std::forward<Args>(args),
                  ypp[n] = std::unique_ptr<T[]>(new T[num_points]), ++n)...};
    (void)tmp;

    deltax = std::unique_ptr<T[]>(new T[num_points - 1]);
    for (int i = 0; i < num_points - 1; ++i) {
      deltax[i] = x[i + 1] - x[i];
      assert(!std::isnan(deltax[i]));
      assert(deltax[i] > 0.);
    }

    minval = x[0];
    maxval = x[num_points-1];

    compute_ypp();
  }

  // Dummy constructor, can't really be used
  cubic_spline_t() = default;

  template <typename... Args>
  inline void replace_y(Args &&... args) {
    static_assert(sizeof...(args) == num_vars,
                  "Need exactly num_vars arguments");
    int n = 0;
    int tmp[] = {(y[n] = std::forward<Args>(args), ++n)...};
    (void)tmp;

    minval = x[0];
    maxval = x[num_points-1];

    compute_ypp();
  }
};

#endif /* ----- #ifndef spline_INC  ----- */
