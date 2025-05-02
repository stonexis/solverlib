#pragma once
#include <memory>
#include <unordered_set>
#include "gtest/gtest.h"
#include "genmatrix.hpp"
#include "matrix.hpp"


template< 
    std::size_t K_, //Количество подотрезков интeрполяции
    std::size_t N_, //Количество узлов конечного элемента (Степень полинома на 1 меньше)
    std::size_t L_ //Количество внутренних "случайных" точек
    >
struct Params {
    static constexpr double A = 0.0;
    static constexpr double B = 10.0;
    static constexpr std::size_t K = K_;
    static constexpr std::size_t N = N_;
    static constexpr std::size_t L = L_;
    static constexpr std::size_t M = K_ * (N_ - 1) + 1; //Общее количество узлов сетки на [a,b], K*(N-1) + 1, в каждом элементе теряем 1 узел из за перекрытия, кроме первого элемента
    static constexpr double H = (B - A) / (M-1); //Шаг равномерной сетки
};


template<typename P>
class MatrixFixture : public ::testing::Test {
    protected:
        static constexpr double A = P::A;
        static constexpr double B = P::B;
        static constexpr std::size_t K = P::K;
        static constexpr std::size_t N = P::N;
        static constexpr std::size_t M = P::M;
        static constexpr std::size_t L = P::L;
        static constexpr double H = P::H;

        using BandType = BandMatrix<double, N - 1, M>;
        double* vector_b = nullptr;
        std::unique_ptr<BandType> matrix_A;

        Eigen::Matrix<double, M, M> A_eigen;
        Eigen::Matrix<double, M, 1> b_eigen;
        
        void SetUp() override {
            std::pair<double**, double*> data = gen_data(M, H, L,N, K, A, B);
            
            matrix_A = BandType::ConvertToBandFromSymBlock(std::get<0>(data));
            
            vector_b = std::get<1>(data);

            for(std::size_t i = 0; i < M; i++){
                b_eigen(i) = vector_b[i];
                for(std::size_t j = 0; j < M; j++)
                    A_eigen(i,j) = std::get<0>(data)[i][j];

            }
            //delete_2d_array(std::get<0>(data));
        }
    
        void TearDown() override {
            delete[] vector_b;
            vector_b = nullptr;
        }
    };

using AllParams = ::testing::Types<     
                            Params< 3, 5, 40 >, //  K,N,L
                            Params< 1, 3, 30 >,
                            Params< 9, 2, 10 >,
                            Params< 8, 6, 10 >,
                            Params< 2, 3, 9 >,
                            Params< 8, 6, 10 >,
                            Params< 4, 3, 20 > // Params< 8, 6, 10 >
                            >;
TYPED_TEST_SUITE(MatrixFixture, AllParams);