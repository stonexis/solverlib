#pragma once
#include <memory>
#include "gtest/gtest.h"
#include "genmatrix.hpp"
#include "matrix.hpp"

class MatrixFixture : public ::testing::Test {
    protected:
        using BandType = BandMatrix<double, Task_const::N - 1, Task_const::M>;
        
        std::unique_ptr<BandType> matrix_A;
        
        double* vector_b = nullptr;
        double* test_result = nullptr;

        void SetUp() override {
            std::tuple<double**, double*, double*> data = gen_data(Task_const::M, Task_const::H);
            using BandType = BandMatrix<double, Task_const::N - 1, Task_const::M>;
            auto mat_ptr = BandType::ConvertToBandFromSymBlock(std::get<0>(data));
            matrix_A = mat_ptr->clone();
            vector_b = std::get<1>(data);
            test_result = std::get<2>(data);
        }
    
        void TearDown() override {
            
            delete[] vector_b;
            delete[] test_result;
        }
    };
