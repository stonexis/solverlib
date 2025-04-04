#include <optional>
#include <Eigen/Dense>

template <typename T>
T* solve_gauss(T** matrix_A, T* vector_b, std::size_t dim_matrix, std::size_t length_internal){
    if (matrix_A == nullptr || vector_b == nullptr) throw std::invalid_argument("Input arrays cannot be null");
    if (dim_matrix == 0) throw std::invalid_argument("Incorrect lengths");
    
    T** copy_matrix_A = copy_matrix(matrix_A, dim_matrix);
    T* copy_vector_b = new T[dim_matrix];
    std::copy(vector_b, vector_b + dim_matrix, copy_vector_b);
    T* solution = new T[dim_matrix]{};
    
    //Прямой ход гаусса
    for (std::size_t i = 0; i < dim_matrix - 1; i++){
        std::size_t end_idx = std::min(length_internal, dim_matrix - i); // 1) Из-за ленточной структуры матрицы не нужно спускаться от диагонали ниже, чем на длину блока(длину диагонали)
                                                                        // 2) Границу ограничиваем оставшимися строками
        for (std::size_t und_i = i + 1; und_i < i + end_idx; und_i++){
            T leading_elem = copy_matrix_A[i][i];
            if (std::abs(leading_elem) < std::numeric_limits<T>::epsilon()) //Нормализация на случай если ведущий элемент нулевой
                leading_elem += std::numeric_limits<T>::epsilon() * 1e-5;

            T multiplier = copy_matrix_A[und_i][i] / leading_elem;

            for (std::size_t j = i; j < i + end_idx; j++) // //Из-за ленточной структуры матрицы не нужно идти от диагонали правее, чем на длину блока(длину диагонали)
                copy_matrix_A[und_i][j] -= multiplier * copy_matrix_A[i][j];
            
            copy_vector_b[und_i] -= multiplier * copy_vector_b[i];
        }
    }
    // Обратный ход гаусса
    solution[dim_matrix - 1] = copy_vector_b[dim_matrix - 1] / copy_matrix_A[dim_matrix - 1][dim_matrix - 1];//Стартовая итерация с последней строки
    for (int i = dim_matrix - 2; i >= 0; i--) {
        std::size_t end_idx = std::min(length_internal, dim_matrix - i);
        T tmp_sum = T(0);
        
        for (std::size_t j = i + 1; j < i + end_idx; j++) //Из-за ленточной структуры матрицы не нужно идти от диагонали правее, чем на длину блока(длину диагонали)
            tmp_sum += copy_matrix_A[i][j] * solution[j];
        solution[i] = (copy_vector_b[i] - tmp_sum) / copy_matrix_A[i][i];
    }

    delete_2d_array(copy_matrix_A, dim_matrix);
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