#pragma once
#include <Eigen/Dense>
#include <optional>
#include "genmatrix.hpp"

template <typename T>
T* solve_gauss(T** matrix_A, T* vector_b, std::size_t dim_matrix, std::size_t length_internal);
template <typename T>
T** copy_matrix(T** original, std::size_t dim_matrix);

#include "solverlib.tpp"