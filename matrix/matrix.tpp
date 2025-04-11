#pragma once
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <memory>

//----------------Sym Matrix ----------------------

template<typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
inline T& SymBandMatrix<T, NumUpDiag, SizeDiag>::operator()(std::size_t i, std::size_t j) noexcept {
    if (j < i)//тк матрица симметричная обращение к элементу под диагональю равно обращению к элементу над диагональю
        std::swap(i, j);
    
    std::size_t d = j - i; //Индекс диагонали соответсвующий переданным индексам, если значение j - i, - означает доступ к элементу под главной диагональю - модуль его отразит в симметричную верхнюю 
    std::size_t j_diag = i; //Индекс позиции элемента в диагонали соответсвует строке, откуда элемент
    return m_data[d * SizeDiag + j_diag];
}

template<typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
inline const T& SymBandMatrix<T, NumUpDiag, SizeDiag>::operator()(std::size_t i, std::size_t j) const noexcept {
    if (j < i)//тк матрица симметричная обращение к элементу под диагональю равно обращению к элементу над диагональю
        std::swap(i, j);
    
    std::size_t d = j - i; //Индекс диагонали соответсвующий переданным индексам, если значение j - i, - означает доступ к элементу под главной диагональю - модуль его отразит в симметричную верхнюю 
    std::size_t j_diag = i; //Индекс позиции элемента в диагонали соответсвует строке, откуда элемент
    return m_data[d * SizeDiag + j_diag];
}

template<typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
auto SymBandMatrix<T, NumUpDiag, SizeDiag>::ConvertToBandFromBlock(T** original) noexcept 
    -> std::unique_ptr<SymBandMatrix<T, NumUpDiag, SizeDiag>>
{
    auto result = std::make_unique<SymBandMatrix<T, NumUpDiag, SizeDiag>>();
    
    for(std::size_t i = 0; i < SizeDiag; i++)
        for(std::size_t j = i; j < i + std::min( NumUpDiag + 1, SizeDiag - i); j++) // идем по строке либо до конца блока, либо, если пришли к последнему блоку, до упора
            (*result)(i , j) = original[i][j];
        
    return result;
}

template<typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
void SymBandMatrix<T, NumUpDiag, SizeDiag>::PrintBandMatrix(std::size_t width) const noexcept{
    for (std::size_t i = 0; i < SizeDiag; ++i) {
        std::size_t start_idx = std::max(static_cast<int>(i - NumUpDiag), 0);
        std::size_t end_idx = i + std::min(NumUpDiag + 1, SizeDiag - i);
        for (std::size_t j = start_idx; j < end_idx; ++j)
            std::cout << std::setprecision(4) << std::setw(width) << (*this)(i,j);
        std::cout <<'\n';
    }
    std::cout << "-------------------------" << "\n";
}

template<typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
auto SymBandMatrix<T, NumUpDiag, SizeDiag>::clone() const noexcept
    -> std::unique_ptr<SymBandMatrix<T, NumUpDiag, SizeDiag>>
{   
    auto clone = std::make_unique<SymBandMatrix<T, NumUpDiag, SizeDiag>>();
    for (std::size_t i = 0; i < (NumUpDiag + 1) * SizeDiag; ++i)
        (*clone)[i] = (*this)[i];

    return clone;

}

template<typename T>
void PrintRawMatrix(T** data, std::size_t rows, std::size_t cols, int width = 8) {
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) 
            std::cout << std::setprecision(4) << std::setw(width) << data[i][j];
        std::cout <<'\n';
    }
    std::cout << "-------------------------" << "\n";
}

//--------------------------Band Matrix --------------------------------------------------------------

template<typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
inline T& BandMatrix<T, NumUpDiag, SizeDiag>::operator()(std::size_t i, std::size_t j) noexcept {
    std::size_t d = j + NumUpDiag - i;
    std::size_t j_diag = i < j ? i : j; //Индекс позиции элемента в диагонали соответсвует строке, откуда элемент, если над диагональю и столбцу, если под
    return m_data[d * SizeDiag + j_diag];
}

template<typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
inline const T& BandMatrix<T, NumUpDiag, SizeDiag>::operator()(std::size_t i, std::size_t j) const noexcept {
    std::size_t d = j + NumUpDiag - i;
    std::size_t j_diag = i < j ? i : j; //Индекс позиции элемента в диагонали соответсвует строке, откуда элемент, если над диагональю и столбцу, если под
    return m_data[d * SizeDiag + j_diag];
}

template<typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
auto BandMatrix<T, NumUpDiag, SizeDiag>::ConvertToBandFromSymBlock(T** original) noexcept 
    -> std::unique_ptr<BandMatrix<T, NumUpDiag, SizeDiag>>
{
    auto result = std::make_unique<BandMatrix<T, NumUpDiag, SizeDiag>>();
    
    for(std::size_t i = 0; i < SizeDiag; i++){
        std::size_t start_idx = std::max(static_cast<int>(i - NumUpDiag), 0);
        std::size_t end_idx = i + std::min(NumUpDiag + 1, SizeDiag - i);
        for(std::size_t j = start_idx; j < end_idx; j++){ // идем по строке либо до конца блока, либо, если пришли к последнему блоку, до упора
            (*result)(i , j) = original[i][j];
        }
    }
    return result;
}

template<typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
void BandMatrix<T, NumUpDiag, SizeDiag>::PrintBandMatrixByLines(std::size_t width) const noexcept{
    for (std::size_t i = 0; i < SizeDiag; ++i) {
        std::size_t start_idx = std::max(static_cast<int>(i - NumUpDiag), 0);
        std::size_t end_idx = i + std::min(NumUpDiag + 1, SizeDiag - i);
        for (std::size_t j = start_idx; j < end_idx; ++j)
            std::cout << std::setprecision(4) << std::setw(width) << (*this)(i,j);
        std::cout <<'\n';
    }
    std::cout << "-------------------------" << "\n";
}

template<typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
void BandMatrix<T, NumUpDiag, SizeDiag>::PrintBandMatrix(std::size_t width) const noexcept{

    for (std::size_t i = 0; i < (2 * NumUpDiag + 1) * SizeDiag; ++i) {
        if (i % SizeDiag == 0)
            std::cout <<'\n';
        std::cout << std::setprecision(4) << std::setw(width) << (*this)[i];
    }
    std::cout << "\n" << "-------------------------" << "\n";
}

template<typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
auto BandMatrix<T, NumUpDiag, SizeDiag>::clone() const noexcept
    -> std::unique_ptr<BandMatrix<T, NumUpDiag, SizeDiag>>
{   
    auto clone = std::make_unique<BandMatrix<T, NumUpDiag, SizeDiag>>();
    for (std::size_t i = 0; i < (2 * NumUpDiag + 1) * SizeDiag; ++i)
        (*clone)[i] = (*this)[i];

    return clone;

}