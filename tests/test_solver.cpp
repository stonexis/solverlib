#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Dense>
#include "matrix.hpp"
#include "solverlib.hpp"
#include "test_fixture.hpp"

TEST_F(MatrixFixture, SolvesGaussCorrectly) {

    double epsilon = 1e-9;
    auto x = solve_gauss(matrix_A, vector_b);
    auto x_eigen = A_eigen.lu().solve(b_eigen);
    for(std::size_t i = 0; i < Task_const::M; i++)
    EXPECT_TRUE((x[i] - x_eigen(i)) < epsilon) << "solverGauss() gave wrong result:\n"
                                              << "Expected: " << x_eigen(i) << "\n"
                                              << "Got     : " << x[i];
}

TEST_F(MatrixFixture, SolvesLUCroutCorrectly) {

    double epsilon = 1e-9;
    auto x = solve_lu_decomposition_crout(matrix_A, vector_b);
    auto x_eigen = A_eigen.lu().solve(b_eigen);
    for(std::size_t i = 0; i < Task_const::M; i++)
    EXPECT_TRUE((x[i] - x_eigen(i)) < epsilon) << "solverLU() gave wrong result:\n"
                                              << "Expected: " << x_eigen(i) << "\n"
                                              << "Got     : " << x[i];
}

TEST_F(MatrixFixture, SolvesCholetskyCorrectly) {
    double epsilon = 1e-9;
    auto x = solve_choletsky(matrix_A, vector_b);

    Eigen::LLT<Eigen::Matrix<double, Task_const::M, Task_const::M>> llt;
    llt.compute(A_eigen);
    auto x_eigen = llt.solve(b_eigen);
    for(std::size_t i = 0; i < Task_const::M; i++)
    EXPECT_TRUE((x[i] - x_eigen(i)) < epsilon) << "solverCholetsky() gave wrong result:\n"
                                              << "Expected: " << x_eigen(i) << "\n"
                                              << "Got     : " << x[i];
}

TEST_F(MatrixFixture, SolvesOverRelaxCorrectly) {

    double epsilon = 1e-9;
    auto x = solve_success_over_relax(*matrix_A, vector_b);
    Eigen::ConjugateGradient<Eigen::Matrix<double, Task_const::M, Task_const::M>, Eigen::Lower | Eigen::Upper> cg; // Eigen::Lower | Eigen::Upper Указывает, что A рассматривается как матрица общего вида, без ограничения на нижнюю или верхнюю треугольную форму
    // Предобрабатываем матрицу A
    cg.compute(A_eigen);
    // Решаем систему уравнений
    auto x_eigen = cg.solve(b_eigen);
    for(std::size_t i = 0; i < Task_const::M; i++)
    EXPECT_TRUE((x[i] - x_eigen [i]) < epsilon) << "solverSuccessOverRelax() gave wrong result:\n"
                                              << "Expected: " << x_eigen[i] << "\n"
                                              << "Got     : " << x[i];
}