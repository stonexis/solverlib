#pragma once
#include <cmath>
#include <numeric>
#include "matrix.hpp"
#include "genmatrix.hpp"

template<class Mat> struct ProxyGauss; //forward declaration
template<class Mat> struct ProxyLU;
template<class Mat> struct ProxyCholetsky;
template<class Mat> struct ProxySuccesRelax;

namespace Backend {

    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    T* solve_gauss(const BandMatrix<T, NumUpDiag, SizeDiag>& matrix_A, const T* vector_b);

    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    T* solve_lu_decomposition_crout(const BandMatrix<T, NumUpDiag, SizeDiag>& matrix_A, const T* vector_b);

    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    T* solve_choletsky(const BandMatrix<T, NumUpDiag, SizeDiag>& matrix_A, const T* vector_b);

    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    T* solve_success_over_relax(
                            const BandMatrix<T, NumUpDiag, SizeDiag>& matrix_A, 
                            const T* vector_b,
                            T omega=1.5, 
                            T epsilon=1e-10, 
                            std::size_t k_max=100,
                            std::size_t except_stable_iter=5,
                            bool good_conditions=true
                        );

    template <typename T>
    T* gen_start_init_vec(std::size_t dim_vector, std::size_t random_seed);

    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    T* band_matrix_with_vector_product(const BandMatrix<T, NumUpDiag, SizeDiag>& matrix, const T* vector);

    template<typename T>
    T calc_2norm_vector(T* vec, std::size_t dim_vec){ return std::sqrt(std::inner_product(vec, vec + dim_vec, vec, T(0)));}
}

//-------------- Proxy hub ---------------
template<class Mat>
struct MatrixTools {
    Mat const& A;

    explicit MatrixTools(Mat const& m_) : A(m_) {}

    auto Gauss() const { return ProxyGauss<Mat>{ A }; }
    auto LU() const { return ProxyLU<Mat>{ A }; }
    auto Choletsky() const { return ProxyCholetsky<Mat>{ A };}
    auto SuccesRelax() const { return ProxySuccesRelax<Mat>{ A };}
    
    
};

//-------------------------- Proxy calls ------------------
template<class Mat>
class ProxyGauss {
    Mat const& A;
public:
    explicit ProxyGauss(Mat const& m) : A(m) {}
    template<typename T>
    auto solve(const T* vector_b) const { return Backend::solve_gauss(A, vector_b);}
};

template<class Mat>
class ProxyLU {
    Mat const& A;
public:
    explicit ProxyLU(Mat const& m) : A(m) {}
    template<typename T>
    auto solve(const T* vector_b) const { return Backend::solve_lu_decomposition_crout(A, vector_b);}
};

template<class Mat>
class ProxyCholetsky {
    Mat const& A;
public:
    explicit ProxyCholetsky(Mat const& m) : A(m) {}
    template<typename T>
    auto solve(const T* vector_b) const { return Backend::solve_choletsky(A, vector_b);}
};

template<class Mat>
class ProxySuccesRelax {
    Mat const& A;
public:
    explicit ProxySuccesRelax(Mat const& m) : A(m) {}
    template<typename T>
    auto solve(const T* vector_b) const { return Backend::solve_success_over_relax(A, vector_b);}
};

#include "backimplement.tpp"