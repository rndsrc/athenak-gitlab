/*
 * =====================================================================================
 *
 *       Filename:  linear_interp_ND.hh
 *
 *    Description:  Linear interpolation for arbitrary dimensions
 *
 *        Version:  1.1
 *        Created:  24/08/2017 10:00:00
 *       Revision:  none
 *       Compiler:  gcc,clang
 *
 *         Author:  Elias Roland Most (ERM), most@fias.uni-frankfurt.de
 *   Organization:  Goethe University Frankfurt
 *
 * =====================================================================================
 */

#ifndef linear_interp_ND_INC
#define linear_interp_ND_INC

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <algorithm>

template <typename T, size_t dim, int num_vars, bool uniform>
class lintp_ND_t {
 public:
  using vec_ptr_t = std::unique_ptr<T[]>;

  template <int N>
  using vec_t = std::array<T, N>;

  std::array<T, dim> idx, dx;
  int nvars = num_vars;
  std::array<size_t, dim> num_points;
  std::array<vec_ptr_t, dim> x;
  vec_ptr_t y;

 private:
  // Useful helper for checking conditions on template parameters
  // https://stackoverflow.com/questions/28253399/check-traits-for-all-variadic-template-arguments/28253503#28253503
  template <bool...>
  struct bool_pack;
  template <bool... bs>
  using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;

  //! Get nearest table index of input value xin
  inline __attribute__((always_inline)) size_t find_index(
      size_t const which_dim, T const &xin) const {
    if (uniform) {
      // shift xin by lowest value
      const auto xL = (xin - x[which_dim][0]);
      
      if(xL <= 0) return 0;

      auto index = static_cast<int>(xL * idx[which_dim]);

     if(index > num_points[which_dim] - 2) return num_points[which_dim] - 2; 

      return index;

    } else {
      int lower = 0;
      int upper = num_points[which_dim] - 1;
      // simple bisection through the table range
      while (upper - lower > 1) {
        int tmp = lower + (upper - lower) / 2;
        if (xin < x[which_dim][tmp])
          upper = tmp;
        else
          lower = tmp;
      }
      return lower;
    }
  };

  template <typename CI_t, size_t offset>
  struct compressed_indexr_t {
    template <size_t N = 0>
    static inline __attribute__((always_inline)) size_t value(
        std::array<size_t, dim> const &index,
        std::array<size_t, dim> const &num_points) {
      return offset + index[N] +
             num_points[N] * CI_t::template value<N + 1>(index, num_points);
    }
  };

  struct compressed_index_t {
    template <size_t N = 0>
    static inline __attribute__((always_inline)) size_t value(
        std::array<size_t, dim> const &index,
        std::array<size_t, dim> const &num_points) {
      return 0;
    }
  };

  static inline __attribute__((always_inline)) T const linterp1D(
      T const &A, T const &B, T const &xA, T const &xB, T const &x) {
    const auto lambda = (x - xA) / (xB - xA);  // *ideltax;
    // return  A*(1.-lambda) + B*lambda;
    return (B - A) * lambda + A;
  }

  // Unpack in dimension!
  template <typename F_t, size_t which_dim, size_t var, typename CI_t,
            typename X1, typename... Xt>
  struct recursive_interpolate_single {
    static inline __attribute__((always_inline)) T const linterp(
        F_t const &F, std::array<size_t, dim> const &index, Xt const &... xin,
        X1 const &x1) {
      auto const xA = F.x[which_dim][index[which_dim]];
      auto const xB = F.x[which_dim][index[which_dim] + 1];

      auto const A =
          recursive_interpolate_single<F_t, which_dim - 1, var,
                                       compressed_indexr_t<CI_t, 0>,
                                       Xt...>::linterp(F, index, xin...);
      auto const B =
          recursive_interpolate_single<F_t, which_dim - 1, var,
                                       compressed_indexr_t<CI_t, 1>,
                                       Xt...>::linterp(F, index, xin...);

      return linterp1D(A, B, xA, xB, x1);
    }
  };

  // Unpack in dimension!
  template <typename F_t, size_t var, typename CI_t, typename X1>
  struct recursive_interpolate_single<F_t, 0, var, CI_t, X1> {
    static inline __attribute__((always_inline)) T const linterp(
        F_t const &F, std::array<size_t, dim> const &index, X1 const &x1) {
      constexpr size_t which_dim = 0;

      auto const xA = F.x[which_dim][index[which_dim]];
      auto const xB = F.x[which_dim][index[which_dim] + 1];

      auto const A = F[var + num_vars * compressed_indexr_t<CI_t, 0>::value(
                                            index, F.num_points)];
      auto const B = F[var + num_vars * compressed_indexr_t<CI_t, 1>::value(
                                            index, F.num_points)];

      return linterp1D(A, B, xA, xB, x1);
    }
  };

 public:
  inline T const &operator[](const int i) const { return y[i]; }

  inline int size(size_t const which_dim) { return num_points[which_dim]; }

  inline T const &xmin(size_t which_dim) { return x[which_dim][0]; };

  template <size_t which_dim>
  inline T const &xmin() {
    return x[which_dim][0];
  };

  inline T const &xmax(size_t which_dim) {
    return x[which_dim][num_points[which_dim] - 1];
  };

  template <size_t which_dim>
  inline T const &xmax() {
    return x[which_dim][num_points[which_dim] - 1];
  };

  template <size_t... vars, typename... Xt>
  inline vec_t<sizeof...(vars)> interpolate(Xt const &... xin) const {
    static_assert(
        all_true<(std::is_convertible<typename std::remove_cv<Xt>::type,
                                      T>::value)...>::value,
        "Interpolation points must be compatible with the data type of the "
        "arrays.");
    static_assert(
        sizeof...(Xt) == dim,
        "You need to provide the correct number of interpolation points");
    static_assert(sizeof...(vars) <= num_vars,
                  "You are trying to interpolate more variables than this "
                  "container holds.");

    int dir = -1;
    auto const index =
        std::array<size_t, dim>{(++dir, find_index(dir, xin))...};

    return vec_t<sizeof...(vars)>{(
        recursive_interpolate_single<decltype(*this), dim - 1, vars,
                                     compressed_index_t,
                                     Xt...>::linterp(*this, index, xin...))...};
  };

  lintp_ND_t() = default;

  template <typename Yt, typename... Xt>
  lintp_ND_t(Yt &&__y, std::array<size_t, dim> &&__num_points, Xt &&... __x)
      : num_points(std::move(__num_points)), y(std::move(__y)) {
    int n = 0;
    int tmp[] = {(x[n] = std::forward<Xt>(__x), ++n)...};
    (void)tmp;

    for (int i = 0; i < dim; ++i) {
      dx[i] = x[i][1] - x[i][0];
      idx[i] = 1. / dx[i];
    };
  };
};

template <typename T, size_t dim, int num_vars>
using linear_interp_uniform_ND_t = lintp_ND_t<T, dim, num_vars, true>;

template <typename T, size_t dim, int num_vars>
using linear_interp_ND_t = lintp_ND_t<T, dim, num_vars, false>;

#endif /* ----- #ifndef linear_interp_ND_INC  ----- */
