#pragma once
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <memory>

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

// template<typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
// BandMatrix<T, NumUpDiag, SizeDiag> BandMatrix<T, NumUpDiag, SizeDiag>::clone() const noexcept {   
//     auto clone = std::make_unique<BandMatrix<T, NumUpDiag, SizeDiag>>();
//     for (std::size_t i = 0; i < (2 * NumUpDiag + 1) * SizeDiag; ++i)
//         (*clone)[i] = (*this)[i];

//     return clone;

// }