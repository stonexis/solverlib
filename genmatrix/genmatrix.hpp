#pragma once
#include <string>
#include <map>
#include "matrix.hpp"

/**
 * @namespace Task_const
 * @brief Содержит константы и параметры задачи.
 */
namespace Task_const {
    /// Редактируемые параметры
    inline constexpr double A = 0.0; ///Концы отрезка
    inline constexpr double B = 10.0; ///Концы отрезка
    inline constexpr std::size_t K = 3; ///Количество подотрезков интeрполяции
    inline constexpr std::size_t N = 4; ///Количество узлов конечного элемента (Степень полинома на 1 меньше)
    inline constexpr std::size_t L = 8; ///Количество внутренних "случайных" точек
    inline constexpr std::size_t M_viz = 300; ///Количество точек равномерной сетки для отображения графика

    /// Нередактируемые параметры 
    inline constexpr std::size_t M = K * (N - 1) + 1; ///Общее количество узлов сетки на [a,b], K*(N-1) + 1, в каждом элементе теряем 1 узел из за перекрытия, кроме первого элемента
    inline const double H = std::abs(B-A)/(M-1); ///Шаг равномерной сетки
    inline const double STEP_H_100 = H / 100; /// Шаг сетки h/100
    inline const double STEP_M_viz = std::abs(B-A)/(M_viz - 1); /// Шаг сетки M_viz 
}

template <typename T> 
T* gen_func_arr(const T *array_x, const std::size_t length);

template <typename T> 
T** gen_func_2d_arr(T **array_2d_x, const std::size_t length_internal, const std::size_t length_external=Task_const::K);

template <typename T>
void delete_2d_array(T**& array, std::size_t length_external=Task_const::K);

template <typename T>
void write_to_file_arr_2d(
                std::ofstream &out,
                T **array,
                const std::string &name_array, 
                const std::size_t length_internal,
                const std::size_t length_external=Task_const::K
                );
template <typename T>
void write_data_to_file(
            T **array_nodes,
            T **array_f_nodes, 
            T **array_x, 
            T **array_fx, 
            T **array_approximation,  
            const std::size_t length_internal_values, 
            const std::size_t length_internal_nodes=Task_const::N,
            const std::size_t length_external=Task_const::K
            );
template <typename T> 
const T* gen_uniform_grid(
                const T step=Task_const::H,
                const std::size_t count_nodes=Task_const::M, 
                const T a=Task_const::A, 
                const T b=Task_const::B
                );
template <typename T>
T** gen_2d_arr_uniform(
            const T *array_global,
            const T step=Task_const::H, 
            const std::size_t length_internal=Task_const::N, 
            const std::size_t length_external=Task_const::K, 
            const T a=Task_const::A, 
            const T b=Task_const::B 
            );
template <typename T>
T* gen_random_arr_in_local(
                const T *arr_old, 
                const std::size_t length_old=Task_const::N, 
                const std::size_t count_random_points=Task_const::L
                );
template <typename T>
T** gen_random_2d_arr_in_local(
                    T **arr_old,
                    const std::size_t length_internal=Task_const::N,
                    const std::size_t length_external=Task_const::K,
                    const std::size_t count_random_points=Task_const::L
                    );
template <typename T>
T* gen_uniform_arr_in_local(
                    bool content_orig_mesh, 
                    const T *arr_old, 
                    const std::size_t length_old, 
                    std::size_t& length_out, 
                    const T step
                    );
template <typename T>
T** gen_2d_uniform_between_nodes(
                    bool content_orig_mesh,
                    T **array_nodes, 
                    std::size_t& length_new, 
                    const T step, 
                    const std::size_t length_internal=Task_const::N, 
                    const std::size_t length_external=Task_const::K
                    );
template <typename T>
T* denominator_fun(const T *array_nodes_x, const std::size_t count_nodes_points=Task_const::N);
template <typename T>
T** gen_lagrange_basis_arr_local(
                    const T* array_nodes, 
                    const T* array_random, 
                    const std::size_t count_nodes_points=Task_const::N, 
                    const std::size_t count_eval_points=Task_const::L
                    );
template <typename T>
T** gen_local_matrix(
                const T* array_nodes, 
                const T* array_random,
                const std::size_t count_nodes_points=Task_const::N, 
                const std::size_t count_random_points=Task_const::L
                );
template <typename T>
T** gen_global_matrix(
                T** array_nodes,
                T** array_random,
                const std::size_t length_internal_nodes=Task_const::N,
                const std::size_t length_external=Task_const::K,
                const std::size_t length_internal_random=Task_const::L,
                const std::size_t dim_matrix=Task_const::M
                );
template <typename T>
T* gen_local_vector_b(
                const T* array_nodes, 
                const T* array_random,
                const T* array_func,
                const std::size_t count_nodes_points=Task_const::N,
                const std::size_t count_random_points=Task_const::L
                );
template <typename T>
T* gen_global_vector_b(
                T** array_nodes, 
                T** array_random, 
                T** array_func,
                const std::size_t length_internal_nodes=Task_const::N,
                const std::size_t length_external=Task_const::K,
                const std::size_t length_internal_random=Task_const::L,
                const std::size_t dim_matrix=Task_const::M
                );
template <typename T>
T** solve_system(
            T** matrix_A,
            T* vector_b,
            const std::size_t dim_matrix=Task_const::M,
            const std::size_t length_internal=Task_const::N,
            const std::size_t length_external=Task_const::K
            );
template <typename T>
T* solve_system_for_gen(
            T** matrix_A,
            T* vector_b,
            const std::size_t dim_matrix=Task_const::M,
            const std::size_t length_internal=Task_const::N,
            const std::size_t length_external=Task_const::K
            );
template <typename T>
T* best_approximation_local(
                    const T* local_coefficients, 
                    const T* array_nodes, 
                    const T* eval_points, 
                    const std::size_t count_eval_points,
                    const std::size_t count_nodes_points=Task_const::N
                    );
template <typename T>
T** best_approximation_global(
                    T **coefficients, 
                    T **array_nodes, 
                    T **eval_points, 
                    const std::size_t count_eval_points,
                    const std::size_t length_internal=Task_const::N,
                    const std::size_t length_external=Task_const::K
                    );
template <typename T>
std::pair< std::map<std::string, T>, std::map<std::string, T> > calculate_errors(
                                                                        T** points_fx, 
                                                                        T** points_lx, 
                                                                        const std::size_t length_internal, 
                                                                        const std::size_t length_external=Task_const::K
                                                                        );
template <typename T>
void print_error_table(
                const std::pair<std::map<std::string, T>, std::map<std::string, T>> errors_random,
                const std::pair<std::map<std::string, T>, std::map<std::string, T>> errors_h_100
                );

template <typename T>
std::tuple<T**, T*, T*> gen_data(std::size_t size_matrix=Task_const::M, T step=Task_const::H, std::size_t count_rnd_points = Task_const::L){
    
    auto* array_global_nodes = gen_uniform_grid(step, size_matrix); // Генерируем глобальную сетку
    auto** array_2d_nodes = gen_2d_arr_uniform(array_global_nodes); // Разбиваем глобальную сетку на подсетки (с перекрытием в 1 узел)
    auto** array_2d_random = gen_random_2d_arr_in_local(array_2d_nodes); // Генерируем массив случайных точек на каждом КЭ, (В нем содержатся только случаные точки, без узлов)
    auto** array_2d_func_in_rand = gen_func_2d_arr(array_2d_random, Task_const::L); // Применяем заданную функцию к массиву случаных точек
    auto** array_2d_func_in_nodes = gen_func_2d_arr(array_2d_nodes,Task_const::N);
    
    // Создание и решение слау
    auto** global_matrix = gen_global_matrix(array_2d_nodes, array_2d_random); // Собираем глобальную матрицу для решения слау
    auto* global_b_vector = gen_global_vector_b(array_2d_nodes, array_2d_random, array_2d_func_in_rand); // Собираем глобальную правую часть для решения слау
    auto* result = solve_system_for_gen(global_matrix, global_b_vector);
    

    return std::make_tuple(global_matrix, global_b_vector, result);
}
#include "genmatrix.tpp"