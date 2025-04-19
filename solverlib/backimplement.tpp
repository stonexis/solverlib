#pragma once
#include <cmath>
#include <random>
#include <algorithm>
#include <Eigen/Dense>
#include "matrix.hpp"

namespace Backend {

    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    T* solve_gauss(const BandMatrix<T, NumUpDiag, SizeDiag>& matrix_A, const T* vector_b){
        if (vector_b == nullptr) throw std::invalid_argument("Vector_b is null");

        std::size_t dim_matrix = SizeDiag;
        auto copy_matrix_A = std::make_unique<BandMatrix<T, NumUpDiag, SizeDiag>>(matrix_A);

        T* copy_vector_b = new T[dim_matrix];
        std::copy(vector_b, vector_b + dim_matrix, copy_vector_b);
        T* solution = new T[dim_matrix]{};
        
        //Прямой ход гаусса
        for (std::size_t i = 0; i < dim_matrix - 1; i++){
            std::size_t end_idx = std::min(NumUpDiag + 1, dim_matrix - i); // 1) Из-за ленточной структуры матрицы не нужно спускаться от диагонали ниже, чем на длину блока(ширину диагонали)
                                                                            // 2) Границу ограничиваем оставшимися строками
            for (std::size_t und_i = i + 1; und_i < i + end_idx; und_i++){
                T leading_elem = (*copy_matrix_A)(i , i);
                if (std::abs(leading_elem) < std::numeric_limits<T>::epsilon()) //Нормализация на случай если ведущий элемент нулевой
                    leading_elem += std::numeric_limits<T>::epsilon() * 1e-5;

                T multiplier = (*copy_matrix_A)(und_i, i) / leading_elem;

                for (std::size_t j = i; j < i + end_idx; j++) //Из-за ленточной структуры матрицы не нужно идти от диагонали правее, чем на длину блока(длину диагонали)
                    (*copy_matrix_A)(und_i, j) -= multiplier * (*copy_matrix_A)(i, j);
                
                copy_vector_b[und_i] -= multiplier * copy_vector_b[i];
            }
        }
        
        // Обратный ход гаусса
        solution[dim_matrix - 1] = copy_vector_b[dim_matrix - 1] / (*copy_matrix_A)(dim_matrix - 1, dim_matrix - 1);//Стартовая итерация с последней строки
        for (int i = dim_matrix - 2; i >= 0; i--) {
            std::size_t end_idx = std::min(NumUpDiag + 1, dim_matrix - i);
            T tmp_sum = T(0);
            
            for (std::size_t j = i + 1; j < i + end_idx; j++) //Из-за ленточной структуры матрицы не нужно идти от диагонали правее, чем на длину блока(длину диагонали)
                tmp_sum += (*copy_matrix_A)(i, j) * solution[j];
            solution[i] = (copy_vector_b[i] - tmp_sum) / (*copy_matrix_A)(i,i);
        }

        delete[] copy_vector_b;

        return solution;
    }

    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    T* solve_lu_decomposition_crout(const BandMatrix<T, NumUpDiag, SizeDiag>& matrix_A, const T* vector_b){
        if (vector_b == nullptr) throw std::invalid_argument("Vector_b is null");

        std::size_t dim_matrix = SizeDiag;

        auto copy_matrix_A = std::make_unique<BandMatrix<T, NumUpDiag, SizeDiag>>(matrix_A);

        T* copy_vector_b = new T[dim_matrix];
        std::copy(vector_b, vector_b + dim_matrix, copy_vector_b);
        T* solution = new T[dim_matrix]{};

        //In-place реализация, L и U записываются поверх A. Богачев стр 20
        //В форме краута LU разложения диагональ исходной матрицы лежит в L, а диагональ U = 1 (единичная)
        //Первый столбец L равен первому столбцу A, поскольку inplace реализация, столбец остается без изменений
        for(std::size_t k = 1; k < NumUpDiag + 1; k++) //В случае с неленточной матрицей граница цикла = dim_matrix
            (*copy_matrix_A)(0,k) /= (*copy_matrix_A)(0,0); //Вычисляем элементы первой строки матрицы U

        auto col_calc = [&](std::size_t i, std::size_t k){ //Вычисление l_ik
            T sum = T(0);
            std::size_t start_id = std::max(static_cast<int>(k - NumUpDiag), 0);
            std::size_t start_idx = std::max(static_cast<int>(i - NumUpDiag), static_cast<int>(start_id));
            for(std::size_t j = start_idx; j < k; j++)
                sum += (*copy_matrix_A)(i,j) * (*copy_matrix_A)(j, k);   
            (*copy_matrix_A)(i, k) -= sum;
        };
        auto row_calc = [&](std::size_t i, std::size_t k){ //Вычисление u_ik
            T sum = T(0);
            std::size_t start_idx = std::max(static_cast<int>(k - NumUpDiag), 0);
            for(std::size_t j = start_idx; j < i; j++)
                sum += (*copy_matrix_A)(i, j) * (*copy_matrix_A)(j , k);
            (*copy_matrix_A)(i, k) = ((*copy_matrix_A)(i, k) - sum) / (*copy_matrix_A)(i, i);
        };

        for(std::size_t i = 1; i < dim_matrix; i++){
            std::size_t start_idx = std::max(static_cast<int>(i - NumUpDiag), 1); //Пока находимся в первом блоке, стартуем от 1, после отходим от i на ширину диагонали
            for(std::size_t k = start_idx; k <= i; k++)
                col_calc(i, k);
                
            std::size_t end_idx = std::min(NumUpDiag + 1, dim_matrix - i);// Пока не подошли к последнему блоку, итерироваться по столбцам достаточно только до i + ширина ленты
            for(std::size_t k = i + 1; k < i + end_idx; k++)
                row_calc(i, k);
        }
        
        //Прямой ход
        T* y = new T[dim_matrix]{};
        for (std::size_t i = 0; i < dim_matrix; i++) {
            T sum = 0;
            std::size_t start_idx = std::max(static_cast<int>(i - NumUpDiag), 0); //Пока находимся в первом блоке, стартуем от 0, после отходим от i на ширину диагонали
            for (std::size_t j = start_idx; j < i; j++)
                sum += (*copy_matrix_A)(i,j) * y[j];  // L_{i,j} * y_j
            
            y[i] = (copy_vector_b[i] - sum) / (*copy_matrix_A)(i,i);
        }

        // Обратный ход
        solution[dim_matrix - 1] = y[dim_matrix - 1]; //Стартовая итерация с последней строки, Так как U_{i,i} = 1 делить не нужно
        for (int i = dim_matrix - 2; i >= 0; i--) {
            std::size_t end_idx = std::min(NumUpDiag + 1, dim_matrix - i);
            T tmp_sum = T(0);
            
            for (std::size_t j = i + 1; j < i + end_idx; j++) //Из-за ленточной структуры матрицы не нужно идти от диагонали правее, чем на длину блока(длину диагонали)
                tmp_sum += (*copy_matrix_A)(i,j) * solution[j]; // U_{i,j} * x_j
            solution[i] = y[i] - tmp_sum; // Так как U_{i,i} = 1 делить не нужно
        }
        
        delete[] copy_vector_b;
        delete[] y;

        return solution;
    }

    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    T* solve_choletsky(const BandMatrix<T, NumUpDiag, SizeDiag>& matrix_A, const T* vector_b){
        if (vector_b == nullptr) throw std::invalid_argument("Vector_b is null");

        std::size_t dim_matrix = SizeDiag;

        auto copy_matrix_A = std::make_unique<BandMatrix<T, NumUpDiag, SizeDiag>>(matrix_A);

        T* copy_vector_b = new T[dim_matrix];
        std::copy(vector_b, vector_b + dim_matrix, copy_vector_b);
        T* solution = new T[dim_matrix]{};

        auto sum_calc_ii = [&](std::size_t i){ //ссылка статична, захвачена до входа в цикл, однако данные в ней изменяются
            T sum = T(0);
            std::size_t start_idx = std::max(static_cast<int>(i - NumUpDiag), 0);
            for(std::size_t k = start_idx; k < i; k++) //sum_k=1 i-1
                sum += (*copy_matrix_A)(k,i) * (*copy_matrix_A)(k,i); //(r_ki)^2
            return sum;
        };
        auto sum_calc_ij = [&](std::size_t i, size_t j){
            T sum = T(0);
            std::size_t start_id = std::max(static_cast<int>(i - NumUpDiag), 0);
            std::size_t start_idx = std::max(static_cast<int>(start_id), static_cast<int>(j - NumUpDiag));
            for(std::size_t k = start_idx; k < i; k++)
                sum += (*copy_matrix_A)(k,i) * (*copy_matrix_A)(k,j);
            
            return sum;
        };

        for(std::size_t i = 0; i < dim_matrix; i++){
            (*copy_matrix_A)(i,i) = std::sqrt((*copy_matrix_A)(i,i) - sum_calc_ii(i));
            std::size_t end_idx = std::min(NumUpDiag + 1, dim_matrix - i);
                for(std::size_t j = i + 1; j < i + end_idx; j++)
                    (*copy_matrix_A)(i,j) = ((*copy_matrix_A)(i,j) - sum_calc_ij(i,j)) / (*copy_matrix_A)(i,i);
        }

        //Прямой ход
        T* y = new T[dim_matrix]{};
        for (std::size_t i = 0; i < dim_matrix; i++) {
            T sum = 0;
            std::size_t start_idx = std::max(static_cast<int>(i - NumUpDiag), 0); //Пока находимся в первом блоке, стартуем от 0, после отходим от i на ширину диагонали
            for (std::size_t j = start_idx; j < i; j++)
                sum += (*copy_matrix_A)(j,i) * y[j];  // R*_{i,j} * y_j
            
            y[i] = (copy_vector_b[i] - sum) / (*copy_matrix_A)(i,i);
        }

        // Обратный ход
        solution[dim_matrix - 1] = y[dim_matrix - 1] / (*copy_matrix_A)(dim_matrix - 1, dim_matrix - 1); //Стартовая итерация с последней строки
        for (int i = dim_matrix - 2; i >= 0; i--) {
            std::size_t end_idx = std::min(NumUpDiag + 1, dim_matrix - i);
            T tmp_sum = T(0);
            
            for (std::size_t j = i + 1; j < i + end_idx; j++) //Из-за ленточной структуры матрицы не нужно идти от диагонали правее, чем на длину блока(длину диагонали)
                tmp_sum += (*copy_matrix_A)(i,j) * solution[j]; // R_{i,j} * x_j
            solution[i] = (y[i] - tmp_sum) / (*copy_matrix_A)(i,i);
        }
        
        delete[] copy_vector_b;
        delete[] y;

        return solution;
    }

    template <typename T>
    T* gen_start_init_vec(std::size_t dim_vector, std::size_t random_seed){
        T* init_vec = new T[dim_vector];
        std::mt19937 gen(random_seed);
        std::normal_distribution<T> distrib(0, 1);
        for(std::size_t i = 0; i < dim_vector; i++)
            init_vec[i] = distrib(gen);

        return init_vec;
    }

    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    T* solve_success_over_relax(
                            const BandMatrix<T, NumUpDiag, SizeDiag>& matrix_A, 
                            const T* vector_b,
                            T omega, 
                            T epsilon, 
                            std::size_t k_max,
                            std::size_t except_stable_iter,
                            bool good_conditions //Матрица симметрична, положительно определенная, с диагональным преобладанием или w<=1
                        ){
        if (vector_b == nullptr) throw std::invalid_argument("Vector_b is null");

        std::size_t dim_matrix = SizeDiag;
        auto vector_x = gen_start_init_vec<T>(dim_matrix, 42); // Генерация 2-х стартовых векторов, начальный (k) и пустой (k+1)

        std::size_t num_iter_where_r_small = 0;
        for(std::size_t k_iteration = 0; k_iteration < k_max; k_iteration++){
            T max_diff = T(0); // Для критерия остановки процесса
            for(std::size_t i = 0; i < dim_matrix; i++){ //Нижняя сумма (новые значения)
                T down_sum = T(0);
                std::size_t start_idx = std::max(static_cast<int>(i - NumUpDiag), 0);
                for(std::size_t j = start_idx; j < i; j++)
                    down_sum += matrix_A(i,j) * vector_x[j];

                T up_sum = T(0);
                std::size_t end_idx = std::min(NumUpDiag + 1, dim_matrix - i);
                for(std::size_t j = i + 1; j < i + end_idx; j++) //Верхняя сумма (старые значения)
                    up_sum += matrix_A(i,j) * vector_x[j];

                T old_val = vector_x[i];
                vector_x[i] = (1 - omega) * vector_x[i] + (omega / matrix_A(i,i) ) * ( (vector_b[i] - down_sum - up_sum ));
                T diff = std::abs(vector_x[i] - old_val); //Значение разности x_k+1 x_k, формируется для каждой компоненты
                if (diff > max_diff)//Считаем L_inf по всем компонентам 
                    max_diff = diff;
            }
            if (max_diff < epsilon) //L_inf норма разности приближений
                num_iter_where_r_small++;
            else 
                num_iter_where_r_small = 0;
            if (good_conditions == true && num_iter_where_r_small == except_stable_iter) 
                break; //Метод монотонный, признак сходимости
        }

        return vector_x;
    }
    /// ||Ax_k - b||
    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    T norm_residual(const BandMatrix<T, NumUpDiag, SizeDiag>& matrix_A, const T* vector_b, const T* x_k){




    }
    template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    T* band_matrix_with_vector_product(const BandMatrix<T, NumUpDiag, SizeDiag>& matrix, const T* vector){
        std::size_t dim_matrix = SizeDiag;
        T* result = new T[dim_matrix];
        for(std::size_t i = 0; i < dim_matrix; i++){
            std::size_t start_idx = std::max(static_cast<int>(i - NumUpDiag), 0);
            std::size_t end_idx = i + std::min(NumUpDiag + 1, dim_matrix - i);
            T sum = T(0);
            for(std::size_t j = start_idx; j < end_idx; j++){
                sum += matrix(i,j) * vector[j];
            }
            result[i] = sum;            
        }
        return result;
    }


    // template <typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
    // T* solve_conjugate_grad(
    //                         const BandMatrix<T, NumUpDiag, SizeDiag>& matrix_A, 
    //                         const T* vector_b,
    //                         T omega, 
    //                         T epsilon, 
    //                         std::size_t k_max,
    //                         std::size_t except_stable_iter
    //                     ){





    // }
}