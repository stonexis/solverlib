#include <iostream>
#include <memory>
#include "matrix/matrix.hpp"
#include "genmatrix.hpp"
#include "solverlib.hpp"

int main() {

    std::tuple<double**, double*, double*> data = gen_data(Task_const::M, Task_const::H);
    auto [matrix_A, vector_b, test_result] = data;
    //auto ptr = std::unique_ptr<SymBandMatrix>(new SymBandMatrix(Convert(...)));
    using BandType = BandMatrix<double, Task_const::N - 1, Task_const::M>;
    auto mat_ptr = BandType::ConvertToBandFromSymBlock(matrix_A);
    PrintRawMatrix(matrix_A, Task_const::M, Task_const::M);
    auto result = solve_lu_decomposition_crout(*mat_ptr, vector_b);
    //auto result2 = solve_gauss(*mat_ptr, vector_b);
    //mat_ptr->PrintBandMatrix();
    //mat_ptr->PrintBandMatrixByLines();
    // for(std::size_t i = 0; i < Task_const::M; i++)
    //      std::cout << result[i] << " ";
    // std::cout << "\n" << "---------------" << "\n";
    // for(std::size_t i = 0; i < Task_const::M; i++)
    //      std::cout << result2[i] << " ";
    // for(std::size_t i = 0; i < Task_const::M; i++){
    //     std::cout << "\n";
    //     for(std::size_t j = 0; j < Task_const::M; j++)
    //         std::cout << matrix_A[i][j] << " ";

    // }
    return 0;
}

