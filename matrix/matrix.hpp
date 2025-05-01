#pragma once
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <memory>

template<class Mat> struct MatrixTools; //Предекларация forward declaration

template<typename T, std::size_t NumUpDiag, std::size_t SizeDiag>
struct BandMatrix {
    T m_data[(2*NumUpDiag + 1) * SizeDiag]{};
    
    ///Обращение как к обычной матрице
    inline T& operator()(std::size_t i, std::size_t j) noexcept; 
    inline const T& operator()(std::size_t i, std::size_t j) const noexcept;

    inline T& operator[](std::size_t i){return m_data[i];}
    inline const T& operator[](std::size_t i) const {return m_data[i];}

    //BandMatrix<T, NumUpDiag, SizeDiag> clone() const noexcept;
    void PrintBandMatrix(std::size_t width = 8) const noexcept;
    void PrintBandMatrixByLines(std::size_t width = 8) const noexcept;
    static std::unique_ptr<BandMatrix<T, NumUpDiag, SizeDiag>> ConvertToBandFromSymBlock(T** original) noexcept;
    MatrixTools<BandMatrix<T, NumUpDiag, SizeDiag>> tools_{ *this };
    auto operator->() const { return &tools_; }  //Внешняя структура агрегирующая все вызовы методов (external proxy toolbox)
};

#include "solverlib.hpp"
#include "matrix.tpp"
