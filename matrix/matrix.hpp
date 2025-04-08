#pragma once
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <memory>

/**
NumUpDiag - количество диагоналей над главной (симметричная матрица = количеству диагоналей под главной)
В памяти хранится только верхняя часть матрицы, нижняя часть получается отражением
*/
template<typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
struct SymBandMatrix {
    T m_data[(NumUpDiag + 1) * SizeDiag]{};
    
    ///Обращение как к обычной матрице
    inline T& operator()(std::size_t i, std::size_t j) noexcept; 
    inline const T& operator()(std::size_t i, std::size_t j) const noexcept;

    inline T& operator[](std::size_t i){return m_data[i];}
    inline const T& operator[](std::size_t i) const {return m_data[i];}

    std::unique_ptr<SymBandMatrix<T, NumUpDiag, SizeDiag>> clone() const noexcept;
    void PrintBandMatrix(std::size_t width = 8) const noexcept;
    static std::unique_ptr<SymBandMatrix<T, NumUpDiag, SizeDiag>> ConvertToBandFromBlock(T** original) noexcept;
};

template<typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
struct BandMatrix {
    T m_data[(2*NumUpDiag + 1) * SizeDiag]{};
    
    ///Обращение как к обычной матрице
    inline T& operator()(std::size_t i, std::size_t j) noexcept; 
    inline const T& operator()(std::size_t i, std::size_t j) const noexcept;

    inline T& operator[](std::size_t i){return m_data[i];}
    inline const T& operator[](std::size_t i) const {return m_data[i];}

    std::unique_ptr<BandMatrix<T, NumUpDiag, SizeDiag>> clone() const noexcept;
    void PrintBandMatrix(std::size_t width = 8) const noexcept;
    void PrintBandMatrixByLines(std::size_t width = 8) const noexcept;
    static std::unique_ptr<BandMatrix<T, NumUpDiag, SizeDiag>> ConvertToBandFromSymBlock(T** original) noexcept;
};

#include "matrix.tpp"