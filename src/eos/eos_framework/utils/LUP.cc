#ifndef __HH__LUP
#define __HH__LUP

#include <array>
#include <cmath>

namespace LUP {

template <class T, size_t row, size_t column>
using matrix_ptr = std::array<std::array<T, column>, row>;
template <class T, size_t dim> using vec_ptr = std::array<T, dim>;
template <size_t dim> using int_ptr = vec_ptr<int, dim>;

static constexpr int polish_iterations = 3;
template <typename T, size_t dim, typename matrix_t = matrix_ptr<T, dim, dim>,
	  typename index_t = int_ptr<dim>>
int LUdecompose(matrix_t &A, matrix_t &LU, index_t &index) {

  const double eps = 1.0e-40;
  std::array<T, dim> largest_row_element;
  // initialise LU with A
  for (int i = 0; i < dim; i++) {
    index[i] = i;
    for (int j = 0; j < dim; j++)
      LU[i][j] = A[i][j];
  }
  // Compute largest element of each row
  // This is later used for the pivoting
  for (int i = 0; i < dim; i++) {
    largest_row_element[i] = 0.0;
    for (int j = 0; j < dim; j++) {
      if (fabs(A[i][j]) > largest_row_element[i])
        largest_row_element[i] = fabs(A[i][j]);
    }
  }
  // Now we need to iterate over all columns
  for (int k = 0; k < dim; k++) {
    int row_with_largest_element = k; // No swap
    bool swap = false;
    // Need to compute the largest element in the column
    // to decide whether to swap rows or not.
    T largest_column_element_weight = 0.0;
    for (int i = k; i < dim; i++) {
      T current_element_weight = fabs(LU[i][k]) / largest_row_element[i];
      if (current_element_weight > largest_column_element_weight) {
        largest_column_element_weight = current_element_weight;
        row_with_largest_element = i;
        swap = true;
      }
    }
    // Now we need to interchange rows to get the largest weighted row element
    // as pivot
    if (swap) {
      for (int j = 0; j < dim; j++) {
        // swap elementwise
        T tmp = LU[row_with_largest_element][j];
        LU[row_with_largest_element][j] = LU[k][j];
        LU[k][j] = tmp;
      }
      largest_row_element[row_with_largest_element] = largest_row_element[k];
    }
    // This helps preventing a division by zero.
    if (LU[k][k] == 0.0)
      LU[k][k] = eps;
    // Compute decomposition
    index[k] = row_with_largest_element;
    for (int i = k + 1; i < dim; i++) {
      LU[i][k] /= LU[k][k];
      for (int j = k + 1; j < dim; j++) {
        LU[i][j] -= LU[i][k] * LU[k][j];
      }
    }
  }
  return 0;
}
template <typename T, size_t dim, typename matrix_t = matrix_ptr<T, dim, dim>,
	  typename index_t = int_ptr<dim>, typename vector_t = vec_ptr<T, dim>>
inline int LUsolve(matrix_t &LU, vector_t &b,
                   vector_t &x, index_t &index) {
  for (int i = 0; i < dim; i++)
    x[i] = b[i];
  // solve U*y=b
  for (int i = 0; i < dim; i++) {
    if (i == 0) {
      T tmp = x[i];
      x[i] = x[index[i]];
      x[index[i]] = tmp;
      continue;
    }
    double tmp = x[index[i]];
    x[index[i]] = x[i];
    x[i] = tmp;
    for (int j = i - 1; j < i; j++) {
      x[i] -= LU[i][j] * x[j];
    }
  }
  // Solve Lx=y
  for (int i = dim - 1; i >= 0; i--) {
    for (int j = i + 1; j < dim; j++) {
      x[i] -= LU[i][j] * x[j];
    }
    x[i] /= LU[i][i];
  }
  return 0;
}

template <typename T, size_t dim, typename matrix_t = matrix_ptr<T, dim, dim>,
	  typename index_t = int_ptr<dim>, typename vector_t = vec_ptr<T, dim>>
inline void MatrixMultiply(matrix_t &A, vector_t &x,
                           vector_t &y) {
  for (int i = 0; i < dim; i++) {
    y[i] = 0.0;
    for (int j = 0; j < dim; j++)
      y[i] += A[i][j] * x[j];
  }
}
template <typename T, size_t dim, size_t iterations=polish_iterations,
	 typename matrix_t = matrix_ptr<T, dim, dim>, typename index_t = int_ptr<dim>, 
	 typename vector_t = vec_ptr<T, dim>>
int LUpolish(matrix_t &A, matrix_t &LU,
             vector_t &x, vector_t &b, vector_t &rhs,
             vector_t &dx, index_t &index) {
  for (int it = 0; it < iterations; it++) {
    MatrixMultiply<T, dim>(A, x, rhs);
    for (int i = 0; i < dim; i++)
      rhs[i] -= b[i];
    LUsolve<T, dim>(LU, rhs, dx, index);
    for (int i = 0; i < dim; i++)
      x[i] -= dx[i];
  }
  return 0;
}
}

#endif
