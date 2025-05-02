#pragma once
#include <cmath>
#include <numeric>
#include <algorithm>
#include <string_view>
#include "matrix.hpp"
#include "genmatrix.hpp"

template<class Mat> struct ProxyGauss; //forward declaration
template<class Mat> struct ProxyLU;
template<class Mat> struct ProxyCholetsky;
template<class Mat> struct ProxySuccesRelax;
template<class Mat> struct ProxyConjugateGrad;

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
                            std::size_t k_max=300,
                            std::size_t except_stable_iter=15,
                            bool good_conditions=true
                        );

    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    T* solve_conjugate_grad(
                        const BandMatrix<T, NumUpDiag, SizeDiag>& matrix_A, 
                        const T* vector_b, 
                        T epsilon=1e-10, 
                        std::size_t k_max=300);

    template <typename T>
    T* gen_start_init_vec(std::size_t dim_vector, std::size_t random_seed);

    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    void inplace_product(const BandMatrix<T, NumUpDiag, SizeDiag>& matrix, const T* vector, T* result);

    template<typename T>
    inline T calc_1_norm_vector(const T* vec, std::size_t dim_vec){ 
        return std::accumulate(
                            vec, vec + dim_vec, T(0),
                            [](T sum, T x) { return sum + std::abs(x); }
                        );
    }
        
    template<typename T>
    inline T calc_2_norm_vector(const T* vec, std::size_t dim_vec){ return std::sqrt(std::inner_product(vec, vec + dim_vec, vec, T(0)));}

    template<typename T>
    inline T calc_inf_norm_vector(const T* vec, std::size_t dim_vec){ 
        T max = T(0); 
        for(std::size_t i = 0; i < dim_vec; i++){
            if (std::abs(vec[i]) > max)
                max = std::abs(vec[i]);
        }
        return max;
    }

    template<typename T>
    inline T calc_1_norm_difference(const T* vec_a, const T* vec_b, std::size_t dim_vec){ 
        return std::transform_reduce(
                            vec_a, vec_a + dim_vec, //ForwardIt first, ForwardIt last
                            vec_b,
                            T(0), // init
                            std::plus<>(), // binary_op, применяется после unary(binary1)
                            [](T x, T y) { return std::abs(x-y); }  // unary op or binary1, применяется первой
                        );
    }

    template<typename T>
    inline T calc_2_norm_difference(const T* vec_a, const T* vec_b, std::size_t dim_vec){ 
        return std::sqrt(
                    std::transform_reduce(
                            vec_a, vec_a + dim_vec, //ForwardIt first, ForwardIt last
                            vec_b,
                            T(0), // init
                            std::plus<>(), // binary_op, применяется после unary(binary1)
                            [](T x, T y) { return std::pow((x-y), 2); }  // unary op or binary1, применяется первой
                        )
                    );
    }

    template<typename T>
    inline T calc_inf_norm_difference(const T* vec_a, const T* vec_b, std::size_t dim_vec){
        T max_diff = T(0); 
        for(std::size_t i = 0; i < dim_vec; i++){
            T diff = std::abs(vec_a[i] - vec_b[i]);
            if (diff > max_diff)
                max_diff = diff;
        }
        return max_diff;
    }

    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    T calc_1_norm_residual(const BandMatrix<T, NumUpDiag, SizeDiag>& matrix_A, const T* vector_x, const T* vector_b){
        T* Ax = new T[SizeDiag];
        inplace_product(matrix_A, vector_x, Ax);
        T norm = calc_1_norm_difference(Ax,vector_b, SizeDiag);
        delete [] Ax;
        return norm;
    }

    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    T calc_2_norm_residual(const BandMatrix<T, NumUpDiag, SizeDiag>& matrix_A, const T* vector_x, const T* vector_b){
        T* Ax = new T[SizeDiag];
        inplace_product(matrix_A, vector_x, Ax);
        T norm = calc_2_norm_difference(Ax,vector_b, SizeDiag);
        delete [] Ax;
        return norm;
    }

    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    T calc_inf_norm_residual(const BandMatrix<T, NumUpDiag, SizeDiag>& matrix_A, const T* vector_x, const T* vector_b){
        T* Ax = new T[SizeDiag];
        inplace_product(matrix_A, vector_x, Ax);
        T norm = calc_inf_norm_difference(Ax,vector_b, SizeDiag);
        delete [] Ax;
        return norm;
    }

    template<typename T>
    void print_table(const T* expected, const T* obtained, std::size_t dim, std::string_view method_name);

    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    void print_table_residual(
                        const BandMatrix<T, NumUpDiag, SizeDiag>& matrix_A, 
                        const T* vector_x, const T* vector_b, 
                        std::string_view method_name
                    );
    template<typename T>
    void print_triplet(std::string_view method_name, std::string_view label1, T v1, std::string_view label2, T v2, std::string_view label3, T v3);

    inline void print_line(std::size_t width) {
        std::cout << std::setfill('-') 
                  << std::setw(width) << "" 
                  << std::setfill(' ') 
                  << '\n';
    }

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
    auto ConjugateGrad() const { return ProxyConjugateGrad<Mat>{ A };}
    
    
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

template<class Mat>
class ProxyConjugateGrad {
    Mat const& A;
public:
    explicit ProxyConjugateGrad(Mat const& m) : A(m) {}
    template<typename T>
    auto solve(const T* vector_b) const { return Backend::solve_conjugate_grad(A, vector_b);}
};


#include "backimplement.tpp"