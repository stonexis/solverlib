#pragma once
#include <Eigen/Dense>
#include "genmatrix.hpp"
#include "matrix/matrix.hpp"

template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
T* solve_gauss(BandMatrix<T, NumUpDiag, SizeDiag> matrix_A, T* vector_b);

template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
T* solve_lu_decomposition_crout(BandMatrix<T, NumUpDiag, SizeDiag> matrix_A, T* vector_b);

#include "solverlib.tpp"