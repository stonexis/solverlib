#include <Eigen/Dense>
#include <Eigen/Core>
#include "solverlib.hpp"
#include "test_fixture.hpp"

TYPED_TEST(MatrixFixture, MatrixProductCorrect) {
    double* res = new double[TestFixture::M];
    auto& mat = *this->matrix_A;
    Backend::inplace_product(mat, this->vector_b, res);
    auto res_eigen = this->A_eigen * this->b_eigen;
    
    double epsilon = 1e-9;
    for(std::size_t i = 0; i < TestFixture::M; i++)
    EXPECT_TRUE((res[i] - res_eigen[i]) < epsilon) << "product gave wrong result:\n"
                                              << "Expected: " << res[i] << "\n"
                                              << "Got     : " << res_eigen[i];
}

TYPED_TEST(MatrixFixture, NormsCorrect) {
    auto& mat = *this->matrix_A;
    auto x = mat->ConjugateGrad().solve(this->vector_b);

    Eigen::MatrixXd A_dyn = this->A_eigen;   
    Eigen::VectorXd b_dyn = this->b_eigen;      
    Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower|Eigen::Upper> cg;
    cg.compute(A_dyn);

    Eigen::VectorXd x_eigen = cg.solve(b_dyn);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> x_col(x, TestFixture::M);


    double L1_eigen = this->b_eigen.template lpNorm<1>();
    double L2_eigen = this->b_eigen.template lpNorm<2>();
    double Linf_eigen = this->b_eigen.template lpNorm<Eigen::Infinity>();

    double L1 = Backend::calc_1_norm_vector(this->vector_b, TestFixture::M);
    double L2 = Backend::calc_2_norm_vector(this->vector_b, TestFixture::M);
    double Linf = Backend::calc_inf_norm_vector(this->vector_b, TestFixture::M);

    double L1_diff_eigen = (x_col - x_eigen).lpNorm<1>();
    double L2_diff_eigen = (x_col - x_eigen).lpNorm<2>();
    double Linf_diff_eigen = (x_col - x_eigen).lpNorm<Eigen::Infinity>();

    double* ptr_x = x_col.data();
    double* ptr_x_eigen = x_eigen.data();

    double L1_diff = Backend::calc_1_norm_difference(ptr_x, ptr_x_eigen, TestFixture::M);
    double L2_diff = Backend::calc_2_norm_difference(ptr_x, ptr_x_eigen, TestFixture::M);
    double Linf_diff = Backend::calc_inf_norm_difference(ptr_x, ptr_x_eigen, TestFixture::M);
    double epsilon = 1e-15;

    EXPECT_TRUE((L1_eigen - L1) < epsilon) << "gave wrong result:\n" << "Expected: " << L1_eigen << "\n" << "Got: " << L1;
    EXPECT_TRUE((L2_eigen - L2) < epsilon) << "gave wrong result:\n" << "Expected: " << L2_eigen << "\n" << "Got: " << L2;
    EXPECT_TRUE((Linf_eigen - Linf) < epsilon) << "gave wrong result:\n" << "Expected: " << Linf_eigen << "\n" << "Got: " << Linf;

    EXPECT_TRUE((L1_diff_eigen - L1_diff) < epsilon) << "gave wrong result:\n" << "Expected: " << L1_diff_eigen << "\n" << "Got: " << L1_diff;
    EXPECT_TRUE((L2_diff_eigen - L2_diff) < epsilon) << "gave wrong result:\n" << "Expected: " << L2_diff_eigen << "\n" << "Got: " << L2_diff;
    EXPECT_TRUE((Linf_diff_eigen - Linf_diff) < epsilon) << "gave wrong result:\n" << "Expected: " << Linf_diff_eigen << "\n" << "Got: " << Linf_diff;
}