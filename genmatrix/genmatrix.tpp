/** 
 * @file approxmod.tpp
 * @brief Реализация шаблонных функций
 */
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <utility>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

/**
 * @brief Функция для генерации массива значений заданной функции на отрезке
 * @tparam T Тип данных (float, double, long double).
 * @param array_x Массив, для которого генерируются значения функции (Массив случайных точек).
 * @param length Длина массива точек.
 * @return T* Указатель на массив значений функции.
 * @note Имеет размер length.
 */
template <typename T> 
T* gen_func_arr(const T *array_x, const std::size_t length) {
    if (array_x == nullptr) throw std::invalid_argument("array_x is null");
    if (length <= 0) throw std::invalid_argument("Invalid length values");

    T* arr_func = new T[length]{};
    for (std::size_t i = 0; i < length; i++){
        arr_func[i] = sin(array_x[i]); //заданная функция
    }
    return arr_func;
}
/**
 * @brief Функция для создания двумерного массива значений.
 * @tparam T Тип данных (float, double, long double).
 * @param array_2d_x Двумерный массив, для которого вычисляются значения функции.
 * @param length_internal Количетсво точек на каждом подотрезке внутри двумерного массива.
 * @param length_external Количество подтрезков (По умолчанию Task_const::K).
 * @return Указатель на двумерный массив значений функции. Первая размерность - индекс КЭ, вторая размерность - индекс значения функции в точке.
 * @note Внешний массив имеет размер length_external (По умолчанию Task_const::K). Внутренний — length_internal.
 */
template <typename T>
T** gen_func_2d_arr(T **array_2d_x, const std::size_t length_internal, const std::size_t length_external){
    if (array_2d_x == nullptr) throw std::invalid_argument("array_2d_x is null");
    if (length_external <= 0) throw std::invalid_argument("Invalid length_external values");

    T** arr_2d_func = new T*[length_external]{}; 
    for (std::size_t i = 0; i < length_external; i++)
        arr_2d_func[i] = gen_func_arr(array_2d_x[i], length_internal);
    return arr_2d_func;
}
/**
 * @brief Функция для очистки памяти двумерного массива.
 * @tparam T Тип данных (float, double, long double).
 * @param array Двумерный массив для очистки.
 * @param length_external Внешняя размерность массива.(По умолчанию Task_const::K)
 */
template <typename T>
void delete_2d_array(T**& array, std::size_t length_external){
    if (array == nullptr) return;
    if (length_external == 0) return;
    for (std::size_t i = 0; i < length_external; i++)
        delete[] array[i];
    delete[] array;
    array = nullptr;
}
/**
 * @brief Функция для генерации равномерной сетки на отрезке [a,b]
 * @tparam T Тип данных (float, double, long double).
 * @param step - Шаг равномерной сетки. (По умолчанию Task_conts::H)
 * @param count_nodes - Количество узлов сетки. (По умолчанию Task_const::M)
 * @param a Начало отрезка. (По умолчанию Task_const::A)
 * @param b Конец отрезка. (По умолчанию Task_const::B)
 * @return T* Указатель на массив равномерной сетки.
 * @note Массив имеет размер count_nodes. (По умолчанию Task_const::M)
 */
template <typename T>
const T* gen_uniform_grid(const T step, const std::size_t count_nodes, const T a, const T b) {
    if (count_nodes <= 0) throw std::invalid_argument("Invalid count_nodes values");
    if ((b - a) < std::numeric_limits<T>::epsilon()) throw std::invalid_argument("Invalid a, b values");
    T* array = new T[count_nodes]{}; 
    for (std::size_t i = 0; i < count_nodes; i++) 
        array[i] = a + step * i; // Заполняем значения, включая последний узел, равный b
    if (array[count_nodes - 1] != b)
        array[count_nodes - 1] = b;
    return array;
}
/**
 * @brief Функция для генерации более мелкой, равномерной сетки, внутри существующей
 * @tparam T Тип данных (float, double, long double).
 * @param content_orig_mesh Должен ли новый массив содержать в себе исходную сетку? (True/False)
 * @param arr_old Интервал, внутри которого строится мелкая сетка
 * @param length_old Размер массива, внутри которого строится сетка 
 * @param[out] length_out Длинна нового массива (Заполняемый параметр)
 * @param step Шаг новой сетки
 * @return T* Указатель на массив новой равномерной сетки.
 * @note Массив имеет размер length_new. (Массив не содержит значения исходной сетки, внутри которой строился)
 */
template <typename T>
T* gen_uniform_arr_in_local(bool content_orig_mesh, const T* arr_old, const std::size_t length_old, std::size_t& length_out, const T step) {
    if (arr_old == nullptr) throw std::invalid_argument("array_old is null");
    if (length_old < 2) throw std::invalid_argument("length_old must be at least 2");
    if (step < std::numeric_limits<T>::epsilon()) throw std::invalid_argument("Incorrect step");
    //if (step > std::abs(arr_old[length_old - 1] - arr_old[0])) throw std::invalid_argument("Incorrect step"); // Опциональная проверка(Если выключена, то добавление точек при выполнении условия не происходит)
    
    T interval_length = std::abs(arr_old[length_old - 1] - arr_old[0]); // Длина интервала
    //Округление вверх, поскольку нужно захватить весь интервал 
    std::size_t count_new_nodes = static_cast<std::size_t>(std::ceil(interval_length / step)) + 1; // Рассчитываем количество узлов с фиксированным шагом
    T* arr_new = nullptr;
    if (content_orig_mesh == false){
        arr_new = new T[count_new_nodes]{};
        T a = arr_old[0]; //Начало отсчета
        for (std::size_t i = 0; i < count_new_nodes; i++)
            arr_new[i] = a + step * i; // Заполняем массив равномерными узлами
        length_out = count_new_nodes;
    }
    else {
        arr_new = new T[length_old + count_new_nodes - 1]{}; // Учёт старых узлов и новых точек
        std::copy(arr_old, arr_old + length_old, arr_new); // Копируем старую сетку
        // Добавляем новые точки равномерно, с учётом step
        T a = arr_old[0];
        std::size_t insert_index = length_old; // Индекс для вставки новых точек
        for (std::size_t i = 1; i < count_new_nodes; i++) {
            T value = a + step * i;
            auto position = std::lower_bound(arr_new, arr_new + insert_index, value); // Найти позицию для вставки
            std::rotate(position, arr_new + insert_index, arr_new + insert_index + 1); // Сдвинуть элементы
            *position = value; // Вставить новую точку
            insert_index++;
        }
        length_out = count_new_nodes + length_old - 2; // Отнимаем 2 поскольку строили новую сетку, на основе старой, в которой есть 1я точка и последняя точка
    }
    return arr_new;
}
/**
 * @brief Функция для генерации более мелкой, равномерной сетки, внутри существующей
 * @tparam T Тип данных (float, double, long double).
 * @param content_orig_mesh Должен ли новый массив содержать в себе исходную сетку? (True/False)
 * @param array_nodes Двумерный массив узлов.
 * @param[out] length_new Длина нового массива (Заполняемый параметр)
 * @param step Шаг новой сетки
 * @param length_internal Количество узлов старой сетки на каждом КЭ. (По умолчанию Task_const::N)
 * @param length_external - Количество подотрезков. (По умолчанию Task_const::K)
 * @return T** Указатель на двумерный массив новой равномерной сетки.
 * @note Внешний массив имеет размер length_external (По умолчанию Task_const::K). Внутренний — length_new.
 *  (Массив не содержит значения исходной сетки, внутри которой строился)
 */
template <typename T>
T** gen_2d_uniform_between_nodes(
                            bool content_orig_mesh, 
                            T **array_nodes, 
                            std::size_t& length_new, 
                            const T step, 
                            const std::size_t length_internal, 
                            const std::size_t length_external
                            ){
    if (array_nodes == nullptr) throw std::invalid_argument("Input array cannot be null");
    if (step < std::numeric_limits<T>::epsilon()) throw std::invalid_argument("Incorrect step");
    if (length_internal <= 0 || length_external <= 0) throw std::invalid_argument("Size arrays cannot be 0");

    T** array_2d_uniform = new T*[length_external]{};
    for(std::size_t k = 0; k < length_external; k++)
            array_2d_uniform[k] = gen_uniform_arr_in_local(content_orig_mesh, array_nodes[k], length_internal, length_new, step); // Заполняем внутренние массивы
    return array_2d_uniform;
}
/**
 * @brief Функция для разбиения глобальной сетки на локальные подсетки, с учетом перекрытия в 1 элемент
 * @tparam T Тип данных (float, double, long double).
 * @param array_global Базовый массив глобальной равномерной сетки.
 * @param step - Шаг равномерной сетки. (По умолчанию Task_conts::H)
 * @param length_internal - Количество узлов на каждом КЭ. (По умолчанию Task_const::N)
 * @param length_external - Количество подотрезков. (По умолчанию Task_const::K)
 * @param a Начало отрезка. (По умолчанию Task_const::A)
 * @param b Конец отрезка. (По умолчанию Task_const::B)
 * @return T** Указатель на двумерный массив локальной равномерной сетки, первая размерность - индекс КЭ, вторая размерность - локальный индекс узла КЭ.
 * @note Внешний массив имеет размер length_external (По умолчанию Task_const::K). Внутренний — length_internal (По умолчанию Task_const::N).
 */
template <typename T>
T** gen_2d_arr_uniform(
                const T *array_global, 
                const T step, 
                const std::size_t length_internal, 
                const std::size_t length_external, 
                const T a, const T b
                ){
    if (array_global == nullptr) throw std::invalid_argument("array_global is null");
    if (length_external <= 0 || length_internal <= 0) throw std::invalid_argument("Invalid length_external values");

    T **array_2d_uniform = new T*[length_external]{}; //Выделяем память под массив массивов

    for (std::size_t i = 0; i < length_external; i++) {
        array_2d_uniform[i] = new T[length_internal]{}; //Выделяем память под каждый массив в массиве (Точки локального отрезка)
        for (std::size_t j = 0; j < length_internal; j++) {
            if (i > 0 && j == 0) //Если это первый элемент текущего блока, и это не первый блок, то берем последний элемент предыдущего блока
                array_2d_uniform[i][j] = array_2d_uniform[i-1][length_internal-1]; // последний элемент предыдущего блока
            else 												// иначе берем элемент из одномерного массива
                array_2d_uniform[i][j] = array_global[i * (length_internal - 1) + j];//чтобы смещение между блоками было на 1 элемент меньше, чем размер блока, чтобы обеспечить перекрытие  
            }
        }
    return array_2d_uniform;
}
/**
 * @brief Функция для создания рандомных точек внутри каждого элемента
 * @tparam T Тип данных (float, double, long double).
 * @param arr_old Массив, для которого генерируются внутренние случайные точки.
 * @param length_old Длина массива, в котором генерируются случайние точки. (По умолчанию Task_const::N)
 * @param count_random_points Количество генерируемых точек. (По умолчанию Task_const::L)
 * @return T* Указатель на массив случайных точек (Не содержит исходные точки, внутри которых генерировались случайные).
 * @note Массив имеет размер count_random_points (По умолчанию Task_const::L).
 */
template <typename T>
T* gen_random_arr_in_local(const T *arr_old, const std::size_t length_old, const std::size_t count_random_points){
    if (arr_old == nullptr) throw std::invalid_argument("array_old is null");
    if (count_random_points <= length_old) throw std::invalid_argument("Invalid count points values");

    T* arr_new = new T[count_random_points]{};
    
    //std::random_device rd; //инициализируем случайное зерно
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> distrib(arr_old[0], arr_old[length_old-1]);
    for (std::size_t i = 0; i < count_random_points; i++) {
        T value = distrib(gen);  //Генерируем новое значение
        auto position = std::lower_bound(arr_new, arr_new + i, value); //Вставляем значение в правильное место, чтобы сохранить массив отсортированным
        std::rotate(position, arr_new + i, arr_new + i + 1);//циклически сдвигаем от позиции posit все оставшиеся элементы заполненного массива на 1 вправо
        *position = value;
    }
   
    return arr_new;
}
/**
 * @brief Функция для создания двумерного массива случайных точек.
 * @tparam T Тип данных (float, double, long double).
 * @param arr_old Исходный двумерный массив, для которого генерируются внутренние случайные точки.
 * @param length_internal Длина исходного внутреннего массива (По умолчанию Task_const::N)
 * @param length_external Количество подотрезков (По умолчанию Task_const::K)
 * @param count_random_points Количество случайных точек внутри каждого подотрезка. (По умолчанию - Task_const::L)
 * @return T** Указатель на двумерный массив случайных точек. Первая размерность - индекс КЭ, вторая размерность - индекс случайной точки.
 * @note Внешний массив имеет размер length_external (По умолчанию - Task_const::K). Внутренний — count_random_points (По умолчанию - Task_const::L).
 * @note (Не содержит исходные точки, внутри которых генерировались случайные).
 */
template <typename T>
T** gen_random_2d_arr_in_local(T **arr_old, const std::size_t length_internal, const std::size_t length_external, const std::size_t count_random_points){
    if (arr_old == nullptr) throw std::invalid_argument("array_old is null");
    if (length_external <= 0) throw std::invalid_argument("incorrect length_external");

    T** arr_2d_in_local = new T*[length_external]{}; //Длина нового массива массивов остается прежней, поскольку мы уплотняем только внутренние массивы 
    for (std::size_t i = 0; i < length_external; i++)
        arr_2d_in_local[i] = gen_random_arr_in_local(arr_old[i], length_internal, count_random_points);

    return arr_2d_in_local;
}
/**
 * @brief Функция для вычисления и кеширования знаменателя.
 * @tparam T Тип данных (float, double, long double).
 * @param array_nodes_x Массив узлов конечного элемента.
 * @param count_nodes_points Количество узлов (По умолчанию Task_const::N)
 * @return T* Указатель на массив значений знаменателя.
 * @note Массив имеет размер count_nodes_points.
 */
template <typename T>
T* denominator_fun(const T *array_nodes_x, const std::size_t count_nodes_points) {
    T* denominator = new T[count_nodes_points]{};
    for(std::size_t i = 0; i < count_nodes_points; i++){
        T product = 1;
        for(std::size_t j = 0; j < count_nodes_points; j++){
             if (i != j){
                // Проверка на совпадение узлов
                if (std::abs(array_nodes_x[i] - array_nodes_x[j]) < std::numeric_limits<T>::epsilon()) {
                    //std::cout << array_nodes_x[i] << " " << array_nodes_x[j] << "\n";
                    delete[] denominator; // Очищаем выделенную память перед исключением
                    throw std::runtime_error("Duplicate nodes detected in array_nodes_x");
                }
                product *= array_nodes_x[i] - array_nodes_x[j];
             }
        }
        denominator[i] = product;
    }
    return denominator;
}
/**
 * @brief Функция для создания двумерного массива базисных функций для всех случайных точек на одном конечном элементе.
 * @tparam T Тип данных (float, double, long double).
 * @param array_nodes Массив узлов конечного элемента.
 * @param array_eval_points Массив точек, в которых вычисляем значение базисных функций.
 * @param count_nodes_points Количество узлов для построения базисных функций. (По умолчанию Task_const::N)
 * @param count_eval_points Количество вычисляемых точек, для построения базисных функций. (По умолчанию Task_const::L)
 * @return T** Указатель на двумерный массив значений базисных функций.
 * @note Внешний массив имеет размер count_eval_points (В базовом случае Task_const::L), Внутренний — count_nodes_points. (По умолчанию Task_const::N)
 */
template <typename T>
T** gen_lagrange_basis_arr_local(const T* array_nodes, const T* array_eval_points, const std::size_t count_nodes_points, const std::size_t count_eval_points) {
    if (array_nodes == nullptr || array_eval_points == nullptr) throw std::invalid_argument("Input arrays cannot be null");
    if (count_nodes_points == 0 || count_eval_points == 0) throw std::invalid_argument("Size arrays cannot be 0");
    T* denominator = nullptr;//static T* denominator = nullptr;
    if (!denominator) // Проверяем, был ли уже вычислен знаменатель
        denominator = denominator_fun(array_nodes, count_nodes_points); // Вычисляем знаменатель один раз
    T** lagrange_basis_array = new T*[count_eval_points]{}; // Массив для хранения значений всех базисных функций для всех случайных точек
    for (std::size_t l = 0; l < count_eval_points; l++) { 
        lagrange_basis_array[l] = new T[count_nodes_points]{}; // Выделяем память на внутренний массив размера N (Значения всех базисных функций КЭ для этой точки)
        T eval_point = array_eval_points[l];
        for (std::size_t i = 0; i < count_nodes_points; i++) {
            T product = 1.0;
            for (std::size_t j = 0; j < count_nodes_points; j++) {
                if (i != j) {
                    product *= (eval_point - array_nodes[j]);
                }
            }
            lagrange_basis_array[l][i] = product / denominator[i];
        }
    }
    return lagrange_basis_array; //Первая размерность - количество случайных точек, вторая размерность - все базисные функции для каждой случайной точки
}
/**
 * @brief Функция для создания локальной матрицы, состоящей из попарных скалярных произведений базисных функций a_ij = sum[1...L](phi(x_l)_i * phi(x_l)_j).
 * @tparam T Тип данных (float, double, long double).
 * @param array_nodes Массив узлов конечного элемента.
 * @param array_random Массив случайных точек для вычисления базисных функций.
 * @param count_nodes_points Количество узлов в конечном элементе (По умолчанию Task_const::N)
 * @param count_random_points Количество случайных точек. (По умолчанию Task_const::L)
 * @return T** Указатель на двумерный массив локальной матрицы.
 * @note Внешний массив имеет размер count_nodes_points (По умолчанию Task_const::N), внутренний — count_nodes_points (По умолчанию Task_const::N).
 */
template <typename T>
T** gen_local_matrix(const T* array_nodes, const T* array_random, const std::size_t count_nodes_points, const std::size_t count_random_points){
    if (count_nodes_points == 0 || count_random_points == 0) throw std::invalid_argument("size arrays cannot be null");
    T** lagrange_basis_array = gen_lagrange_basis_arr_local(array_nodes, array_random, count_nodes_points, count_random_points);
    T** local_matrix = new T*[count_nodes_points]{}; // Матрица всевозможных скалярных произведений базисных функций т.е. N*N
    for (std::size_t i = 0; i < count_nodes_points; i++){ // Проходим по всем строчкам матрицы
        local_matrix[i] = new T[count_nodes_points]{};
        for (std::size_t j = 0; j <= i; j++){ // Проходим по всем столбцам матрицы, она симметрична, поэтому достаточно вычислить только значения под диагональю(j<=i)
            T sum = 0.0;
            for (std::size_t l = 0; l < count_random_points; l++) // Цикл по всем случайным точкам, для вычисления скалярного произведения
                sum += lagrange_basis_array[l][i] * lagrange_basis_array[l][j]; // Скалярное произведение
            local_matrix[i][j] = sum;
            local_matrix[j][i] = sum;
        }
    }
    delete_2d_array(lagrange_basis_array, count_random_points);
    return local_matrix;
}
/**
 * @brief Функция для создания глобальной матрицы, состоящей из суммы локальных матриц, в соответсвующих позициях.
 * @tparam T Тип данных (float, double, long double).
 * @param array_nodes Двумерный массив узлов.
 * @param array_random Двумерный массив случайных точек для вычисления базисных функций.
 * @param length_internal_nodes Количество узлов в каждом КЭ (По умолчанию Task_const::N)
 * @param length_external Количество КЭ (По умолчанию Task_const::K)
 * @param length_internal_random Количество случайных точек в каждом КЭ (По умолчанию Task_const::L)
 * @param dim_matrix Размерность матрицы, общее количество узлов. (По умолчанию Task_const::M)
 * @return T** Указатель на двумерный массив глобальной матрицы.
 * @note Внешний массив имеет размер dim_matrix (По умолчанию Task_const::M), внутренний — dim_matrix (По умолчанию Task_const::M).
 */
template <typename T>
T** gen_global_matrix(
                T** array_nodes,
                T** array_random, 
                const std::size_t length_internal_nodes, 
                const std::size_t length_external,
                const std::size_t length_internal_random, 
                const std::size_t dim_matrix
                ){
    if (array_nodes == nullptr || array_random == nullptr) 
        throw std::invalid_argument("Input arrays cannot be null");
    if (length_internal_nodes == 0 || length_external == 0 || length_internal_random == 0 || dim_matrix == 0)
        throw std::invalid_argument("Incorrect lengths");

    T** global_matrix = new T*[dim_matrix]{};
    for (std::size_t i = 0; i < dim_matrix; i++)
        global_matrix[i] = new T[dim_matrix]{};
        
    for (std::size_t k = 0; k < length_external; k++){ // Цикл по всем конечным элементам (По всем локальным матрицам)
        T** local_matrix = gen_local_matrix(array_nodes[k], array_random[k], length_internal_nodes, length_internal_random);
        for(std::size_t i = 0; i < length_internal_nodes; i++){ // Цикл по строчкам локальной матрицы (Нас не интересуют значения за границами локальных подматриц, поскольку там скалярные произведения равны нулю) 
            for(std::size_t j = 0; j < length_internal_nodes; j++){ // Цикл по столбцам локальной матрицы
                std::size_t global_i = i + (length_internal_nodes - 1)*k; // Преобразование индексов из локальных в глобальные
                std::size_t global_j = j + (length_internal_nodes - 1)*k;
                global_matrix[global_i][global_j] += local_matrix[i][j]; // Сборка матрицы A_ij = sum[1...K](a_ij) = sum[1...K](sum[1...L](phi(x_l)_i * phi(x_l)_j))
            } 
        }
        delete_2d_array(local_matrix, length_internal_nodes);	// Очищаем память для следующей локальной матрицы
        local_matrix = nullptr;
    }
    return global_matrix;
}
/**
 * @brief Функция для локальной правой части, b_i = sum[1...L](f(x_l) * phi_i(x_l))
 * @tparam T Тип данных (float, double, long double).
 * @param array_nodes Массив узлов конечного элемента.
 * @param array_random Массив случайных точек для вычисления базисных функций.
 * @param array_func Массив значений заданной функции в случайных точках.
 * @param count_nodes_points Количество узлов в конечном элементе (По умолчанию Task_const::N)
 * @param count_random_points Количество случайных точек. (По умолчанию Task_const::L)
 * @return T* Указатель на массив локальной правой части.
 * @note Массив имеет размер Task_const:N.
 */
template <typename T>
T* gen_local_vector_b(
                const T* array_nodes, 
                const T* array_random, 
                const T* array_func, 
                const std::size_t count_nodes_points, 
                const std::size_t count_random_points
                ){
    if (count_nodes_points == 0 || count_random_points == 0)
        throw std::invalid_argument("Incorrect lengths");
    T *local_vector_b = new T[count_nodes_points]{};
    T** lagrange_basis_array = gen_lagrange_basis_arr_local(array_nodes, array_random, count_nodes_points, count_random_points);
    for (std::size_t i = 0; i < count_nodes_points; i++){
        T sum = 0.0;
        for (std::size_t l = 0; l < count_random_points; l++) //Проходим по всем случайным точкам в данном КЭ
            sum += array_func[l] * lagrange_basis_array[l][i]; // Скалярное произведение
        local_vector_b[i] = sum;
    }
    delete_2d_array(lagrange_basis_array, count_random_points);
    return local_vector_b;
}
/**
 * @brief Функция для глобальной правой части, B_i = sum[1...K](b_i) = sum[1...K](sum[1...L](f(x_l) * phi_i(x_l)))
 * @tparam T Тип данных (float, double, long double).
 * @param array_nodes Двумерный массив узлов.
 * @param array_random Двумерный массив случайных точек для вычисления базисных функций.
 * @param array_func Двумерный массив значений заданной функции в случайных точках
 * @param length_internal_nodes Количество узлов в каждом КЭ (По умолчанию Task_const::N)
 * @param length_external Количество КЭ (По умолчанию Task_const::K)
 * @param length_internal_random Количество случайных точек в каждом КЭ (По умолчанию Task_const::L)
 * @param dim_matrix Размерность вектор-столбца правой части, общее количество узлов. (По умолчанию Task_const::M)
 * @return T* Указатель на массив глобальной правой части.
 * @note Массив имеет размер dim_matrix (По умолчанию Task_const::M).
 */
template <typename T>
T* gen_global_vector_b(
                T** array_nodes, 
                T** array_random, 
                T** array_func,
                const std::size_t length_internal_nodes,
                const std::size_t length_external,
                const std::size_t length_internal_random,
                const std::size_t dim_matrix
                ) {
    if (array_nodes == nullptr || array_random == nullptr || array_func == nullptr) 
        throw std::invalid_argument("Input arrays cannot be null");
    if (length_internal_nodes == 0 || length_external == 0 || length_internal_random == 0 || dim_matrix == 0)
        throw std::invalid_argument("Incorrect lengths");

    T *global_vector = new T[dim_matrix]{};
    for(std::size_t k = 0; k < length_external; k++){
        T* local_vector = gen_local_vector_b(array_nodes[k], array_random[k], array_func[k], length_internal_nodes, length_internal_random);
        for(std::size_t i = 0; i < length_internal_nodes; i++){ // Цикл по строчкам локального вектора
            std::size_t global_i = i + (length_internal_nodes - 1)*k; // Преобразование индексов из локальных в глобальные
            global_vector[global_i] += local_vector[i];
        }
        delete[] local_vector; //Освобождаем место для следующего локального вектора
        local_vector = nullptr;
    }
    return global_vector;
}
/**
 * @brief Функция для решения слау.
 * @tparam T Тип данных (float, double, long double).
 * @param matrix_A Двумерный массив матрицы СЛАУ.
 * @param vector_b Массив правой части.
 * @param dim_matrix Размерность матрицы, общее количество узлов. (По умолчанию Task_const::M)
 * @param length_internal Количество узлов в каждом КЭ (По умолчанию Task_const::N)
 * @param length_external Количество КЭ (По умолчанию Task_const::K)
 * @return T** Указатель на двумерный массив коэффициентов.
 * @note Внешний массив имеет размер length_external (По умолчанию Task_const::K). Внутренний — length_internal (По умолчанию Task_const::N).
 */
template <typename T>
T** solve_system(T** matrix_A, T* vector_b, const std::size_t dim_matrix, const std::size_t length_internal, const std::size_t length_external){
    if (matrix_A == nullptr || vector_b == nullptr) throw std::invalid_argument("Input arrays cannot be null");
    if (length_external == 0 || length_internal == 0 || dim_matrix == 0) throw std::invalid_argument("Incorrect lengths");

    // Создаем объект Eigen::Matrix
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A(dim_matrix, dim_matrix);
    // Копируем данные из matrix_A в матрицу A
    for (std::size_t i = 0; i < dim_matrix; i++)
        for (std::size_t j = 0; j < dim_matrix; j++)
            A(i, j) = matrix_A[i][j];
    // Создаем объект Eigen::Vector на основе массива vector_b
    // Метод Map преобразует непрерывный блок памяти в Eigen::Vector 
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> b(vector_b, dim_matrix); //(Копирования не происходит, работаем с исходным массивов напрямую)
    // Создаем объект решателя метода сопряженных градиентов
    Eigen::ConjugateGradient<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Lower | Eigen::Upper> cg; // Eigen::Lower | Eigen::Upper Указывает, что A рассматривается как матрица общего вида, без ограничения на нижнюю или верхнюю треугольную форму
    // Предобрабатываем матрицу A
    cg.compute(A);
    // Решаем систему уравнений
    Eigen::Matrix<T, Eigen::Dynamic, 1> x = cg.solve(b);
    // Копируем результат в массив coefficients
    T** coefficients = new T*[length_external]{};
    for (std::size_t i = 0; i < length_external; i++){
        coefficients[i] = new T[length_internal]{};
        for(std::size_t j = 0; j < length_internal; j++)
            coefficients[i][j] = x[i*(length_internal - 1) + j]; // Переход от координат одномерных к двумерным
    }
    return coefficients;
}

template <typename T>
T* solve_system_for_gen(T** matrix_A, T* vector_b, const std::size_t dim_matrix, const std::size_t length_internal, const std::size_t length_external){
    if (matrix_A == nullptr || vector_b == nullptr) throw std::invalid_argument("Input arrays cannot be null");
    if (length_external == 0 || length_internal == 0 || dim_matrix == 0) throw std::invalid_argument("Incorrect lengths");

    // Создаем объект Eigen::Matrix
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A(dim_matrix, dim_matrix);
    // Копируем данные из matrix_A в матрицу A
    for (std::size_t i = 0; i < dim_matrix; i++)
        for (std::size_t j = 0; j < dim_matrix; j++)
            A(i, j) = matrix_A[i][j];
    // Создаем объект Eigen::Vector на основе массива vector_b
    // Метод Map преобразует непрерывный блок памяти в Eigen::Vector 
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> b(vector_b, dim_matrix); //(Копирования не происходит, работаем с исходным массивов напрямую)
    // Создаем объект решателя метода сопряженных градиентов
    Eigen::ConjugateGradient<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Lower | Eigen::Upper> cg; // Eigen::Lower | Eigen::Upper Указывает, что A рассматривается как матрица общего вида, без ограничения на нижнюю или верхнюю треугольную форму
    // Предобрабатываем матрицу A
    cg.compute(A);
    // Решаем систему уравнений
    Eigen::Matrix<T, Eigen::Dynamic, 1> x = cg.solve(b);
    // Копируем результат в массив coefficients
    T* coefficients = new T[x.size()];
    std::copy(x.data(), x.data() + x.size(), coefficients);
    return coefficients;
}


/**
 * @brief Функция для построения наилучшего приближения на локальном отрезке на основе локальных коэффициентов
 * @tparam T Тип данных (float, double, long double).
 * @param local_coefficients Массив локальных коэффициентов.
 * @param array_nodes Массив узлов конечного элемента.
 * @param eval_points Массив точек, в которых вычисляется наилучшее приближение.
 * @param count_eval_points Количество точек, в которых вычисляется наилучшее приближение.
 * @param count_nodes_points Количество узлов в конечном элементе (По умолчанию Task_const::N)
 * @return T* Указатель на массив локального наилучшего приближения.
 * @note Массив имеет размер count_eval_points.
 */
template <typename T>
T* best_approximation_local(
                    const T* local_coefficients, 
                    const T* array_nodes, 
                    const T* eval_points, 
                    const std::size_t count_eval_points,
                    const std::size_t count_nodes_points
                    ) {
    if (local_coefficients == nullptr || array_nodes == nullptr || eval_points == nullptr) 
        throw std::invalid_argument("Input arrays cannot be null");
    if (count_nodes_points == 0 || count_eval_points == 0) 
        throw std::invalid_argument("Size arrays cannot be 0");

    T* best_approximation_array = new T[count_eval_points]{};
    T** lagrange_basis_array = gen_lagrange_basis_arr_local(array_nodes, eval_points, count_nodes_points, count_eval_points);
    for (std::size_t i = 0; i < count_eval_points; i++){
        T sum = 0.0;
        for (std::size_t j = 0; j < count_nodes_points; j++)
            sum += local_coefficients[j] * lagrange_basis_array[i][j];
        best_approximation_array[i] = sum;
    }
    delete_2d_array(lagrange_basis_array, count_eval_points);
    return best_approximation_array;
}
/**
 * @brief Функция для построения наилучшего приближения на глобальном отрезке
 * @tparam T Тип данных (float, double, long double).
 * @param coefficients Двумерный массив локальных коэффициентов.
 * @param array_nodes Двумерный массив узлов.
 * @param eval_points Двумерный массив точек, в которых вычисляется наилучшее приближение.
 * @param count_eval_points Количество точек, на каждом подотрезке в которых вычисляется наилучшее приближение.
 * @param length_internal Количество узлов в конечном элементе (По умолчанию Task_const::N)
 * @param length_external Количество КЭ (По умолчанию Task_const::K)
 * @return T** Указатель на массив наилучшего приближения.
 * @note Внешний массив имеет размер length_external (По умолчанию Task_const::K). Внутренний — arr_count_eval_points[k](Тоесть длина внутреннего массива зависит от элемента)
 */
template <typename T>
T** best_approximation_global( 
                    T **coefficients, 
                    T **array_nodes, 
                    T **eval_points, 
                    const std::size_t count_eval_points,
                    const std::size_t length_internal,
                    const std::size_t length_external){
    T** array_best_approximation_global = new T*[length_external]{};
    for(std::size_t k = 0; k < length_external; k++)
        array_best_approximation_global[k] = best_approximation_local(coefficients[k], array_nodes[k], eval_points[k], count_eval_points, length_internal);

    return array_best_approximation_global;
}
/**
 * @brief Функция для записи двумерного массива в файл в формате json. 
 * @tparam T Тип данных (float, double, long double).
 * @param out Указатель на поток ввода.
 * @param array Записываемый двумерный массив.
 * @param name_array Имя массива.
 * @param length_internal Внутренняя размерность массива.
 * @param length_external Внешняя размерность массива (По умолчанию - Task_const::K)
 * @note {"name1" : [[value1, value2, ], [...], ...], "name2" : [[value1, value2, ], [...], ...], ...}
 */
template <typename T>
void write_to_file_arr_2d(std::ofstream &out, T **array, const std::string &name_array, const std::size_t length_internal, const std::size_t length_external){
    out << "\"" << name_array << "\"" << ": ["; // Начало массива в формате JSON
    for (std::size_t i = 0; i < length_external; ++i) {
        out << "[";  // Начало строки
        for (std::size_t j = 0; j < length_internal; ++j) {
            out << array[i][j];
            if (j != length_internal - 1) 
                out << ", ";  // Разделяем значения в строке
        }
        out << "]";  // Конец строки
        if (i != length_external - 1) 
            out << ",\n";  // Разделяем строки
    }
    out << "]";  // Конец массива
}
/**
 * @brief Функция для записи данных в файл, для последующего построения функций при помощи Python
 * @tparam T Тип данных (float, double, long double).
 * @param array_nodes Двумерный массив узлов аппроксимации.
 * @param array_f_nodes Двумерный массив значений функции в узлах.
 * @param array_x Двумерный массив точек, в которых будут строится графики сравнения функций.
 * @param arr_fx Двумерный массив значений функции.
 * @param arr_approximation Двумерный массив значений функции аппроксимации.
 * @param length_internal_values Количество точек, в которых будут строится графики сравнения функций.
 * @param length_internal_nodes Количество узлов аппроксимации внутри каждого элемента (По умолчанию Task_const::N).
 * @param length_external Внешняя размерность массива. Количество КЭ (По умолчанию Task_const::K)
 */
template <typename T>
void write_data_to_file(
            T **array_nodes,
            T **array_f_nodes, 
            T **array_x, 
            T **array_fx, 
            T **array_approximation, 
            const std::size_t length_internal_values, 
            const std::size_t length_internal_nodes,
            const std::size_t length_external
            ) {
    std::ofstream out;
    out.open("data.json");
    if (!out.is_open()) {
        throw std::runtime_error("Can't open file");
    } else {
        out << "{\n";
        write_to_file_arr_2d(out, array_nodes, "array_nodes", length_internal_nodes, length_external);
        out << ",\n";
        write_to_file_arr_2d(out, array_f_nodes, "array_f_nodes", length_internal_nodes, length_external);
        out << ",\n";
        write_to_file_arr_2d(out, array_x, "array_x", length_internal_values, length_external);
        out << ",\n";
        write_to_file_arr_2d(out, array_fx, "array_fx", length_internal_values, length_external);
        out << ",\n";
        write_to_file_arr_2d(out, array_approximation, "array_approximation", length_internal_values, length_external);
        out << "\n}";
        out.close();
    }
}

/**
 * @brief Функция для вычисления абсолютных погрешностей. Вычисляет абсолютные погрешности для норм: L1 = sum{abs(f(x_i) - l(x_i))}, L2 = sqrt(sum{[f(x_i) - l(x_i)]^2}), L_inf = max(f(x_i) - l(x_i))
 * @tparam T Тип данных (float, double, long double).
 * @param points_fx Двумерный массив значений заданной функции.
 * @param points_lx Двумерный массив значений приближающей функции.
 * @param length_internal Количество внутренних точек.
 * @param length_external Количество КЭ (По умолчанию Task_const::K)
 * @return Пара pair(map(absolute_norms),map(relative_norms)), где в каждой map содеражтся соответсвующие значения L_1, L_2, L_inf
 */
template <typename T>
std::pair<std::map<std::string, T>, std::map<std::string, T>> calculate_errors(
                                                                            T** points_fx,
                                                                            T** points_lx,
                                                                            const std::size_t length_internal,
                                                                            const std::size_t length_external
                                                                            ) {
    if (points_fx == nullptr || points_lx == nullptr) 
        throw std::invalid_argument("Input arrays cannot be null");
    if (length_internal == 0 || length_external == 0) 
        throw std::invalid_argument("Array sizes must be greater than 0");

    std::map<std::string, T> absolute_norms = {{"L_1", 0.0}, {"L_2", 0.0}, {"L_inf", 0.0}};
    std::map<std::string, T> relative_norms = {{"L_1", 0.0}, {"L_2", 0.0}, {"L_inf", 0.0}};

    T sum_abs = 0.0, sum_2_abs = 0.0, max_abs = 0.0;
    T sum_rel = 0.0, sum_2_rel = 0.0, max_rel = 0.0;

    for (std::size_t k = 0; k < length_external; k++) {
        std::size_t start_idx = (k == 0) ? 0 : 1; // Пропускаем первый узел на стыках
        for (std::size_t i = start_idx; i < length_internal; i++) {
            T abs_error = std::abs(points_fx[k][i] - points_lx[k][i]);
            sum_abs += abs_error;
            sum_2_abs += abs_error * abs_error;
            max_abs = std::max(max_abs, abs_error);

            if (std::abs(points_fx[k][i]) > std::numeric_limits<T>::epsilon() ) {
                T rel_error = abs_error / std::abs(points_fx[k][i]);
                sum_rel += rel_error;
                sum_2_rel += rel_error * rel_error;
                max_rel = std::max(max_rel, rel_error);
            }
        }
    }

    absolute_norms["L_1"] = sum_abs;
    absolute_norms["L_2"] = std::sqrt(sum_2_abs);
    absolute_norms["L_inf"] = max_abs;

    relative_norms["L_1"] = sum_rel;
    relative_norms["L_2"] = std::sqrt(sum_2_rel);
    relative_norms["L_inf"] = max_rel;

    return std::make_pair(absolute_norms, relative_norms);
}

/**
 * @brief Функция вывода значений абсолютной и относительной погрешностей в формате таблицы
 * @tparam T Тип данных (float, double, long double).
 * @param errors_random Пара двух map(absolute, relative), содержащих значения ошибок в случайных точках
 * @param errors_h_100 Пара двух map(absolute, relative), содержащих значения ошибок в точках с шагом h/100
 */
template <typename T>
void print_error_table(
                    const std::pair<std::map<std::string, T>, std::map<std::string, T>> errors_random, 
                    const std::pair<std::map<std::string, T>, std::map<std::string, T>> errors_h_100
                    ){
    // Функция для вывода одной строчки таблицы
    // Тип функции явно указан через std::function, можно заметить на auto
    std::function<void(const std::string&, const std::map<std::string, T>&)> print_row = 
        [](const std::string& label, const std::map<std::string, T>& data) { // Первый аргумент лямбда функции - названии выводимой строки, второй - данные
            std::cout << std::left // Выравнивание текста влево
                      << std::setw(18) << label // Устанавливает фиксированную ширину вывода
                      << std::scientific << std::setprecision(6) //  Устанавливает точность
                      << std::setw(15) << data.at("L_1")
                      << std::setw(15) << data.at("L_2")
                      << std::setw(15) << data.at("L_inf") << "\n" ;
        };
    // Заголовок таблицы
    std::cout << std::left
              << std::setw(18) << " "
              << std::setw(15) << "L_1"
              << std::setw(15) << "L_2"
              << std::setw(15) << "L_inf" << "\n";

    std::cout << std::string(61, '-') << "\n";

    // Заголовок для Absolute error
    std::cout << "Absolute error" << "\n";
    print_row("Random points", errors_random.first);
    print_row("h/100 points", errors_h_100.first);

    std::cout << std::string(61, '-') << "\n";

    // Заголовок для Relative error
    std::cout << "Relative error" << "\n";
    print_row("Random points", errors_random.second);
    print_row("h/100 points", errors_h_100.second);
}

