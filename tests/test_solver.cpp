#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Dense>
#include "matrix.hpp"
#include "solverlib.hpp"
#include "test_fixture.hpp"

TEST_F(MatrixFixture, SolvesGaussCorrectly) {

    double epsilon = 1e-9;
    auto x = (*matrix_A)->Gauss().solve(vector_b);
    auto x_eigen = A_eigen.lu().solve(b_eigen);
    for(std::size_t i = 0; i < Task_const::M; i++)
    EXPECT_TRUE((x[i] - x_eigen(i)) < epsilon) << "solverGauss() gave wrong result:\n"
                                              << "Expected: " << x_eigen(i) << "\n"
                                              << "Got     : " << x[i];
}

TEST_F(MatrixFixture, SolvesLUCroutCorrectly) {

    double epsilon = 1e-9;
    auto x = (*matrix_A)->LU().solve(vector_b);
    auto x_eigen = A_eigen.lu().solve(b_eigen);
    for(std::size_t i = 0; i < Task_const::M; i++)
    EXPECT_TRUE((x[i] - x_eigen(i)) < epsilon) << "solverLU() gave wrong result:\n"
                                              << "Expected: " << x_eigen(i) << "\n"
                                              << "Got     : " << x[i];
}

TEST_F(MatrixFixture, SolvesCholetskyCorrectly) {
    double epsilon = 1e-9;
    auto x = (*matrix_A)->Choletsky().solve(vector_b);

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
    auto x = (*matrix_A)->SuccesRelax().solve(vector_b);
    
    Eigen::MatrixXd A_dyn = A_eigen;   
    Eigen::VectorXd b_dyn = b_eigen;      

    Eigen::ConjugateGradient<
    Eigen::MatrixXd,              // dynamic Matrix type
    Eigen::Lower|Eigen::Upper     // general symmetric but not triangular
    > cg;

    cg.compute(A_dyn);


    Eigen::VectorXd x_eigen = cg.solve(b_dyn);
    for(std::size_t i = 0; i < Task_const::M; i++)
    EXPECT_TRUE((x[i] - x_eigen(i)) < epsilon) << "solverSuccessOverRelax() gave wrong result:\n"
                                              << "Expected: " << x_eigen(i) << "\n"
                                              << "Got     : " << x[i];
}

TEST_F(MatrixFixture, SolvesConjugateGradCorrectly) {

    double epsilon = 1e-9;
    auto x = (*matrix_A)->ConjugateGrad().solve(vector_b);
    
    Eigen::MatrixXd A_dyn = A_eigen;   
    Eigen::VectorXd b_dyn = b_eigen;      

    Eigen::ConjugateGradient<
    Eigen::MatrixXd,              // dynamic Matrix type
    Eigen::Lower|Eigen::Upper     // general symmetric but not triangular
    > cg;

    cg.compute(A_dyn);


    Eigen::VectorXd x_eigen = cg.solve(b_dyn);
    for(std::size_t i = 0; i < Task_const::M; i++)
    EXPECT_TRUE((x[i] - x_eigen(i)) < epsilon) << "solverSuccessOverRelax() gave wrong result:\n"
                                              << "Expected: " << x_eigen(i) << "\n"
                                              << "Got     : " << x[i];
}