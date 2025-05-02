#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Dense>
#include "matrix.hpp"
#include "solverlib.hpp"
#include "test_fixture.hpp"

TYPED_TEST(MatrixFixture, SolvesGaussCorrectly) {

    double epsilon = 1e-9;
    auto& mat = *this->matrix_A;
    auto x = mat->Gauss().solve(this->vector_b);
    auto x_eigen = this->A_eigen.lu().solve(this->b_eigen);
    for(std::size_t i = 0; i < TestFixture::M; i++)
    EXPECT_TRUE((x[i] - x_eigen(i)) < epsilon) << "solverGauss() gave wrong result:\n"
                                              << "Expected: " << x_eigen(i) << "\n"
                                              << "Got     : " << x[i];
}

TYPED_TEST(MatrixFixture, SolvesLUCroutCorrectly) {

    double epsilon = 1e-9;
    auto& mat = *this->matrix_A;
    auto x = mat->LU().solve(this->vector_b);
    auto x_eigen = this->A_eigen.lu().solve(this->b_eigen);
    for(std::size_t i = 0; i < TestFixture::M; i++)
    EXPECT_TRUE((x[i] - x_eigen(i)) < epsilon) << "solverLU() gave wrong result:\n"
                                              << "Expected: " << x_eigen(i) << "\n"
                                              << "Got     : " << x[i];
}

TYPED_TEST(MatrixFixture, SolvesCholetskyCorrectly) {
    double epsilon = 1e-9;
    auto& mat = *this->matrix_A;
    auto x = mat->Choletsky().solve(this->vector_b);

    Eigen::LLT<Eigen::Matrix<double, TestFixture::M, TestFixture::M>> llt;
    llt.compute(this->A_eigen);
    auto x_eigen = llt.solve(this->b_eigen);
    for(std::size_t i = 0; i < TestFixture::M; i++)
    EXPECT_TRUE((x[i] - x_eigen(i)) < epsilon) << "solverCholetsky() gave wrong result:\n"
                                              << "Expected: " << x_eigen(i) << "\n"
                                              << "Got     : " << x[i];
}

TYPED_TEST(MatrixFixture, SolvesOverRelaxCorrectly) {

    double epsilon = 1e-9;
    auto& mat = *this->matrix_A;
    auto x = mat->SuccesRelax().solve(this->vector_b);
    
    Eigen::MatrixXd A_dyn = this->A_eigen;   
    Eigen::VectorXd b_dyn = this->b_eigen;      

    Eigen::ConjugateGradient<
    Eigen::MatrixXd,              // dynamic Matrix type
    Eigen::Lower|Eigen::Upper     // general symmetric but not triangular
    > cg;

    cg.compute(A_dyn);


    Eigen::VectorXd x_eigen = cg.solve(b_dyn);
    for(std::size_t i = 0; i < TestFixture::M; i++)
    EXPECT_TRUE((x[i] - x_eigen(i)) < epsilon) << "solverSuccessOverRelax() gave wrong result:\n"
                                              << "Expected: " << x_eigen(i) << "\n"
                                              << "Got     : " << x[i];
}

TYPED_TEST(MatrixFixture, SolvesConjugateGradCorrectly) {

    double epsilon = 1e-9;
    auto& mat = *this->matrix_A;
    auto x = mat->ConjugateGrad().solve(this->vector_b);
    
    Eigen::MatrixXd A_dyn = this->A_eigen;   
    Eigen::VectorXd b_dyn = this->b_eigen;      

    Eigen::ConjugateGradient<
    Eigen::MatrixXd,              // dynamic Matrix type
    Eigen::Lower|Eigen::Upper     // general symmetric but not triangular
    > cg;

    cg.compute(A_dyn);


    Eigen::VectorXd x_eigen = cg.solve(b_dyn);
    for(std::size_t i = 0; i < TestFixture::M; i++)
    EXPECT_TRUE((x[i] - x_eigen(i)) < epsilon) << "solverCG() gave wrong result:\n"
                                              << "Expected: " << x_eigen(i) << "\n"
                                              << "Got     : " << x[i];
}