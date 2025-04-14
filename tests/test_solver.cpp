#include <Eigen/Dense>
#include "matrix.hpp"
#include "solverlib.hpp"
#include "test_fixture.hpp"

TEST_F(MatrixFixture, SolvesGaussCorrectly) {

    double epsilon = 1e-9;
    auto x = solve_gauss(matrix_A, vector_b);

    for(std::size_t i = 0; i < Task_const::M; i++)
    EXPECT_TRUE((x[i] - test_result[i]) < epsilon) << "solverGauss() gave wrong result:\n"
                                              << "Expected: " << test_result[i] << "\n"
                                              << "Got     : " << x[i];
}

TEST_F(MatrixFixture, SolvesLUCroutCorrectly) {

    double epsilon = 1e-9;
    auto x = solve_lu_decomposition_crout(matrix_A, vector_b);

    for(std::size_t i = 0; i < Task_const::M; i++)
    EXPECT_TRUE((x[i] - test_result[i]) < epsilon) << "solverLU() gave wrong result:\n"
                                              << "Expected: " << test_result[i] << "\n"
                                              << "Got     : " << x[i];
}

TEST_F(MatrixFixture, SolvesCholetskyCorrectly) {

    double epsilon = 1e-9;
    auto x = solve_choletsky(matrix_A, vector_b);

    for(std::size_t i = 0; i < Task_const::M; i++)
    EXPECT_TRUE((x[i] - test_result[i]) < epsilon) << "solverLU() gave wrong result:\n"
                                              << "Expected: " << test_result[i] << "\n"
                                              << "Got     : " << x[i];
}

TEST_F(MatrixFixture, SolvesOverRelaxCorrectly) {

    double epsilon = 1e-9;
    auto x = solve_success_over_relax(*matrix_A, vector_b);

    for(std::size_t i = 0; i < Task_const::M; i++)
    EXPECT_TRUE((x[i] - test_result[i]) < epsilon) << "solverLU() gave wrong result:\n"
                                              << "Expected: " << test_result[i] << "\n"
                                              << "Got     : " << x[i];
}