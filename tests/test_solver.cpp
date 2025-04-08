#include <Eigen/Dense>
#include "../matrix/matrix.hpp"
#include "../solverlib.hpp"
#include "test_fixture.hpp"

TEST_F(MatrixFixture, SolvesCorrectly) {

    double epsilon = 1e-9;
    auto x = solve_gauss(*matrix_A, vector_b);

    for(std::size_t i = 0; i < Task_const::M; i++)
    EXPECT_TRUE((x[i] - test_result[i]) < epsilon) << "solver() gave wrong result:\n"
                                              << "Expected: " << test_result[i] << "\n"
                                              << "Got     : " << x[i];
}
