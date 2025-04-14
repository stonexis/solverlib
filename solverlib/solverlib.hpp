#pragma once
#include <Eigen/Dense>
#include <numeric>
#include <cmath>
#include "genmatrix.hpp"
#include "matrix.hpp"

template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
T* solve_gauss(std::unique_ptr<BandMatrix<T, NumUpDiag, SizeDiag>>& matrix_A, T* vector_b);

template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
T* solve_lu_decomposition_crout(std::unique_ptr<BandMatrix<T, NumUpDiag, SizeDiag>>& matrix_A, T* vector_b);

template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
T* solve_choletsky(std::unique_ptr<BandMatrix<T, NumUpDiag, SizeDiag>>& matrix_A, T* vector_b);

template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
T* solve_success_over_relax(
                        const BandMatrix<T, NumUpDiag, SizeDiag>& matrix_A, 
                        const T* vector_b,
                        T omega=1.5, 
                        T epsilon=1e-10, 
                        std::size_t k_max=100,
                        std::size_t except_stable_iter=5
                    );

template <typename T>
T* gen_start_init_vec(std::size_t dim_vector, std::size_t random_seed);

template<typename T>
T calc_2norm_vector(T* vec, std::size_t dim_vec){ return std::sqrt(std::inner_product(vec, vec + dim_vec, vec, T(0)));}

#include "solverlib.tpp"