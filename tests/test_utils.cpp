#include <Eigen/Dense>
#include "solverlib.hpp"
#include "test_fixture.hpp"

TEST_F(MatrixFixture, MatrixProductCorrect) {
    auto res = Backend::band_matrix_with_vector_product(*matrix_A, vector_b);
    auto res_eigen = A_eigen * b_eigen;
    
    double epsilon = 1e-9;
    for(std::size_t i = 0; i < Task_const::M; i++)
    EXPECT_TRUE((res[i] - res_eigen[i]) < epsilon) << "product gave wrong result:\n"
                                              << "Expected: " << res[i] << "\n"
                                              << "Got     : " << res_eigen[i];
}