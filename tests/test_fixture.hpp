#pragma once
#include <memory>
#include "gtest/gtest.h"
#include "genmatrix.hpp"
#include "matrix.hpp"

class MatrixFixture : public ::testing::Test {
    protected:
        using BandType = BandMatrix<double, Task_const::N - 1, Task_const::M>;
        double* vector_b = nullptr;
        std::unique_ptr<BandType> matrix_A;

        Eigen::Matrix<double, Task_const::M, Task_const::M> A_eigen;
        Eigen::Matrix<double, Task_const::M, 1> b_eigen;
        
        void SetUp() override {
            std::pair<double**, double*> data = gen_data(Task_const::M, Task_const::H);
            
            auto mat_ptr = BandType::ConvertToBandFromSymBlock(std::get<0>(data));
            matrix_A = mat_ptr->clone();
            vector_b = std::get<1>(data);

            for(std::size_t i = 0; i < Task_const::M; i++){
                b_eigen(i) = vector_b[i];
                for(std::size_t j = 0; j < Task_const::M; j++)
                    A_eigen(i,j) = std::get<0>(data)[i][j];

            }
            
        }
    
        void TearDown() override {
            
            delete[] vector_b;
        }
    };
