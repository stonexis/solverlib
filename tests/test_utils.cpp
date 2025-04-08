#include <Eigen/Dense>
#include "../solverlib.hpp"
#include "test_fixture.hpp"

// TEST_F(MatrixFixture, DISABLED_MyTest) {
//     auto x = solve_gauss(matrix_A, vector_b);

//     double epsilon = 1e-9;
//     for(std::size_t i = 0; i < size_matrix; i++)
//     EXPECT_TRUE((x[i] - test_result[i]) < epsilon) << "solver() gave wrong result:\n"
//                                               << "Expected: " << test_result[i] << "\n"
//                                               << "Got     : " << x[i];
// }