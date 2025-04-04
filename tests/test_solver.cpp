#include "gtest/gtest.h"
#include <Eigen/Dense>
#include "../solverlib.hpp"
#include "../genmatrix.hpp"

class MatrixFixture : public ::testing::Test {
    protected:
        std::size_t size_matrix = Task_const::M;
        double init_param = Task_const::H; //Параметр для определения типа gen_data
        double** matrix_A = nullptr;
        double* vector_b = nullptr;
        double* test_result = nullptr;

        void SetUp() override {
            std::tuple<double**, double*, double*> data = gen_data(size_matrix, init_param);
            std::tie(matrix_A, vector_b, test_result) = data;
        }
    
        void TearDown() override {
            for (std::size_t i = 0; i < size_matrix; ++i)
                delete[] matrix_A[i];
            delete[] matrix_A;
            delete[] vector_b;
            delete[] test_result;
        }
    };

TEST_F(MatrixFixture, SolvesCorrectly) {
    auto x = solve_gauss(matrix_A, vector_b, size_matrix, Task_const::N);

    double epsilon = 1e-9;
    for(std::size_t i = 0; i < size_matrix; i++)
    EXPECT_TRUE((x[i] - test_result[i]) < epsilon) << "solver() gave wrong result:\n"
                                              << "Expected: " << test_result[i] << "\n"
                                              << "Got     : " << x[i];
}
