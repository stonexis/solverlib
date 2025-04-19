#include <iostream>
#include <memory>
#include "genmatrix.hpp"
#include "solverlib.hpp"

int main() {

    auto [matrix_A, vector_b] = gen_data(Task_const::M, Task_const::H);
    //auto ptr = std::unique_ptr<SymBandMatrix>(new SymBandMatrix(Convert(...)));
    using BandType = BandMatrix<double, Task_const::N - 1, Task_const::M>;
    auto mat_ptr = BandType::ConvertToBandFromSymBlock(matrix_A);
    auto& mat = *mat_ptr;
    auto result_gauss = mat->Gauss().solve(vector_b);
    auto result_lu = mat->LU().solve(vector_b);
    auto result_choletsky = mat->Choletsky().solve(vector_b);
    auto result_relax = mat->SuccesRelax().solve(vector_b);
    
    //auto result = band_matrix_with_vector_product(*mat_ptr, vector_b);
    // for(std::size_t i = 0; i < Task_const::M; i++)
    //     std::cout << result[i] << " ";
    // std::cout << "\n";
    // for(std::size_t i = 0; i < Task_const::M; i++)
    //     std::cout << vector_b[i] << " ";
    // std::cout<< "\n";
    //mat_ptr->PrintBandMatrix();
    //mat_ptr->PrintBandMatrixByLines();
//     for(std::size_t i = 0; i < Task_const::M; i++)
//          std::cout << result_gauss[i] << " ";
//     std::cout << "\n" << "---------------" << "\n";
//     for(std::size_t i = 0; i < Task_const::M; i++)
//          std::cout << result_relax[i] << " ";
//     std::cout<< "\n";
    //PrintRawMatrix(matrix_A, Task_const::M, Task_const::M);

    
    return 0;
}

