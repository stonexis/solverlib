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

template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
T* solve_lu_decomposition_crout(BandMatrix<T, NumUpDiag, SizeDiag> matrix_A, T* vector_b){
    std::size_t dim_matrix = SizeDiag;

    T* copy_vector_b = new T[dim_matrix];
    std::copy(vector_b, vector_b + dim_matrix, copy_vector_b);
    T* solution = new T[dim_matrix]{};

    //In-place реализация, L и U записываются поверх A. Богачев стр 20
    //В форме краута LU разложения диагональ исходной матрицы лежит в L, а диагональ U = 1 (единичная)
    //Первый столбец L равен первому столбцу A, поскольку inplace реализация, столбец остается без изменений
    for(std::size_t k = 1; k < NumUpDiag + 1; k++) //В случае с неленточной матрицей граница цикла = dim_matrix
        matrix_A(0,k) /= matrix_A(0,0); //Вычисляем элементы первой строки матрицы U

    auto col_calc = [&](std::size_t i, std::size_t k){ //Вычисление l_ik
        T sum = T(0);
        std::size_t start_id = std::max(static_cast<int>(k - NumUpDiag), 0);
        std::size_t start_idx = std::max(static_cast<int>(i - NumUpDiag), static_cast<int>(start_id));
        for(std::size_t j = start_idx; j < k; j++){
            T first = matrix_A(i,j);
            T second = matrix_A(j,k);
            sum += matrix_A(i,j) * matrix_A(j, k);
        }
        matrix_A(i, k) -= sum;
    };
    auto row_calc = [&](std::size_t i, std::size_t k){ //Вычисление u_ik
        T sum = T(0);
        std::size_t start_idx = std::max(static_cast<int>(k - NumUpDiag), 0);
        for(std::size_t j = start_idx; j < i; j++){
            sum += matrix_A(i, j) * matrix_A(j , k);
        }
        std::cout << "[" << matrix_A(i,i) << "]" << '\n';
        matrix_A(i, k) = (matrix_A(i, k) - sum) / matrix_A(i, i);
    };

    for(std::size_t i = 1; i < dim_matrix; i++){
        std::size_t start_idx = std::max(static_cast<int>(i - NumUpDiag), 1); //Пока находимся в первом блоке, стартуем от 1, после отходим от i на ширину диагонали
        for(std::size_t k = start_idx; k <= i; k++)
            col_calc(i, k);
            
        std::size_t end_idx = std::min(NumUpDiag + 1, dim_matrix - i);// Пока не подошли к последнему блоку, итерироваться по столбцам достаточно только до i + ширина ленты
        for(std::size_t k = i + 1; k < i + end_idx; k++)
            row_calc(i, k);
    }
    matrix_A.PrintBandMatrix();
    matrix_A.PrintBandMatrixByLines();
    
    //Прямой ход
    T* y = new T[dim_matrix]{};
    for (std::size_t i = 0; i < dim_matrix; i++) {
        T sum = 0;
        std::size_t start_idx = std::max(static_cast<int>(i - NumUpDiag), 0); //Пока находимся в первом блоке, стартуем от 0, после отходим от i на ширину диагонали
        for (std::size_t j = start_idx; j < i; j++)
            sum += matrix_A(i,j) * y[j];  // L_{i,j} * y_j
        
        y[i] = (copy_vector_b[i] - sum) / matrix_A(i,i);
    }

    // Обратный ход
    solution[dim_matrix - 1] = y[dim_matrix - 1]; //Стартовая итерация с последней строки, Так как U_{i,i} = 1 делить не нужно
    for (int i = dim_matrix - 2; i >= 0; i--) {
        std::size_t end_idx = std::min(NumUpDiag + 1, dim_matrix - i);
        T tmp_sum = T(0);
        
        for (std::size_t j = i + 1; j < i + end_idx; j++) //Из-за ленточной структуры матрицы не нужно идти от диагонали правее, чем на длину блока(длину диагонали)
            tmp_sum += matrix_A(i,j) * solution[j]; // U_{i,j} * x_j
        solution[i] = y[i] - tmp_sum; // Так как U_{i,i} = 1 делить не нужно
    }
    
    delete[] copy_vector_b;
    delete[] y;

    return solution;
}