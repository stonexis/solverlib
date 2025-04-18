#include <Eigen/Dense>
#include "solverlib.hpp"
#include "test_fixture.hpp"

// TEST_F(MatrixFixture, MatrixProductCorrect) {
//     auto x = band_matrix_with_vector_product(*matrix_A, vector_b);
//     // Создаем объект Eigen::Matrix
//     Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A(dim_matrix, dim_matrix);
//     // Копируем данные из matrix_A в матрицу A
//     for (std::size_t i = 0; i < dim_matrix; i++)
//         for (std::size_t j = 0; j < dim_matrix; j++)
//             A(i, j) = matrix_A[i][j];
//     // Создаем объект Eigen::Vector на основе массива vector_b
//     // Метод Map преобразует непрерывный блок памяти в Eigen::Vector 
//     Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> b(vector_b, dim_matrix); //(Копирования не происходит, работаем с исходным массивов напрямую)

//     double epsilon = 1e-9;
//     for(std::size_t i = 0; i < Task_const::M; i++)
//     EXPECT_TRUE((x[i] - test_result[i]) < epsilon) << "solver() gave wrong result:\n"
//                                               << "Expected: " << test_result[i] << "\n"
//                                               << "Got     : " << x[i];
// }