#include <Eigen/Dense>
#include "matrix/matrix.hpp"

template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
T* solve_gauss(BandMatrix<T, NumUpDiag, SizeDiag> matrix_A, T* vector_b){
    std::size_t dim_matrix = SizeDiag;

    T* copy_vector_b = new T[dim_matrix];
    std::copy(vector_b, vector_b + dim_matrix, copy_vector_b);
    T* solution = new T[dim_matrix]{};
    
    //Прямой ход гаусса
    for (std::size_t i = 0; i < dim_matrix - 1; i++){
        std::size_t end_idx = std::min(NumUpDiag + 1, dim_matrix - i); // 1) Из-за ленточной структуры матрицы не нужно спускаться от диагонали ниже, чем на длину блока(ширину диагонали)
                                                                        // 2) Границу ограничиваем оставшимися строками
        for (std::size_t und_i = i + 1; und_i < i + end_idx; und_i++){
            T leading_elem = matrix_A(i , i);
            if (std::abs(leading_elem) < std::numeric_limits<T>::epsilon()) //Нормализация на случай если ведущий элемент нулевой
                leading_elem += std::numeric_limits<T>::epsilon() * 1e-5;

            T multiplier = matrix_A(und_i, i) / leading_elem;

            for (std::size_t j = i; j < i + end_idx; j++) // //Из-за ленточной структуры матрицы не нужно идти от диагонали правее, чем на длину блока(длину диагонали)
                matrix_A(und_i, j) -= multiplier * matrix_A(i, j);
            
            copy_vector_b[und_i] -= multiplier * copy_vector_b[i];
        }
    }
    
    // Обратный ход гаусса
    solution[dim_matrix - 1] = copy_vector_b[dim_matrix - 1] / matrix_A(dim_matrix - 1, dim_matrix - 1);//Стартовая итерация с последней строки
    for (int i = dim_matrix - 2; i >= 0; i--) {
        std::size_t end_idx = std::min(NumUpDiag + 1, dim_matrix - i);
        T tmp_sum = T(0);
        
        for (std::size_t j = i + 1; j < i + end_idx; j++) //Из-за ленточной структуры матрицы не нужно идти от диагонали правее, чем на длину блока(длину диагонали)
            tmp_sum += matrix_A(i, j) * solution[j];
        solution[i] = (copy_vector_b[i] - tmp_sum) / matrix_A(i,i);
    }

    delete[] copy_vector_b;

    return solution;
}

template <typename T>
T** copy_matrix(T** original, std::size_t dim_matrix){
    
    T** copy_matrix = new T*[dim_matrix];
    for(std::size_t i = 0; i < dim_matrix; i++){
        copy_matrix[i] = new T[dim_matrix];
        std::copy(original[i], original[i] + dim_matrix, copy_matrix[i]);
    }
    return copy_matrix;
}

namespace SuppUtils {
    template<typename T>
    T** copy_matrix(T** original, std::size_t dim_matrix){
    
        T** copy_matrix = new T*[dim_matrix];
        for(std::size_t i = 0; i < dim_matrix; i++){
            copy_matrix[i] = new T[dim_matrix];
            std::copy(original[i], original[i] + dim_matrix, copy_matrix[i]);
        }
        return copy_matrix;
    }




};