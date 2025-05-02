#include <iostream>
#include <memory>
#include "genmatrix.hpp"
#include "solverlib.hpp"

int main() {

    auto [matrix_A, vector_b] = gen_data(Task_const::M, Task_const::H);
    using BandType = BandMatrix<double, Task_const::N - 1, Task_const::M>;
    auto mat_ptr = BandType::ConvertToBandFromSymBlock(matrix_A);
    auto& mat = *mat_ptr;

    auto result_gauss = mat->Gauss().solve(vector_b);
    auto result_lu = mat->LU().solve(vector_b);
    auto result_choletsky = mat->Choletsky().solve(vector_b);
    auto result_relax = mat->SuccesRelax().solve(vector_b);
    auto result_conjugate = mat->ConjugateGrad().solve(vector_b);

    Eigen::Matrix<double, Task_const::M, Task_const::M> A_eigen;
    Eigen::Matrix<double, Task_const::M, 1> b_eigen;

    for(std::size_t i = 0; i < Task_const::M; i++){
        b_eigen(i) = vector_b[i];
        for(std::size_t j = 0; j < Task_const::M; j++)
            A_eigen(i,j) = matrix_A[i][j];
    }

    Eigen::MatrixXd A_dyn = A_eigen;   
    Eigen::VectorXd b_dyn = b_eigen;      

    Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower|Eigen::Upper> cg;
    cg.compute(A_dyn);

    Eigen::VectorXd x_eigen = cg.solve(b_dyn);
    auto* x_eigen_ptr = x_eigen.data();
    //Таблица расхождений в решениях с эталоном - эйген cg
    Backend::print_table(x_eigen_ptr, result_gauss, Task_const::M, "Gauss");
    Backend::print_table(x_eigen_ptr, result_lu, Task_const::M, "LU");
    Backend::print_table(x_eigen_ptr, result_choletsky, Task_const::M, "Choletsky");
    Backend::print_table(x_eigen_ptr, result_relax, Task_const::M, "SOR");
    Backend::print_table(x_eigen_ptr, result_conjugate, Task_const::M, "CG");
    //Таблица невязок
    Backend::print_table_residual(mat, result_gauss, vector_b, "Gauss");
    Backend::print_table_residual(mat, result_lu, vector_b, "LU");
    Backend::print_table_residual(mat, result_choletsky, vector_b, "Choletsky");
    Backend::print_table_residual(mat, result_relax, vector_b, "SOR");
    Backend::print_table_residual(mat, result_conjugate, vector_b, "CG");

    delete[] result_gauss;
    delete[] result_lu;
    delete[] result_choletsky;
    delete[] result_relax;
    delete[] result_conjugate;
    delete[] vector_b;
    
    delete_2d_array(matrix_A, Task_const::M);

    return 0;
}

