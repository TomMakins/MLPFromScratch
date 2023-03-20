#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <iostream>


//  g++ -std=c++11 -o exe MLP.cpp

template <typename Type> 
class Matrix{

  public:
    int cols;
    int rows;
    std::vector<Type> data;
    int size;

    Matrix(int row, int col){
      rows = row;
      cols = col;
      size = row * col;
      data.resize(row * col, Type());
    }

    void randInit() {
      for(int r = 0; r < rows; r++){
          for(int c = 0; c < cols; c++){
          Type temp = rand() % 200000;
          data[cols * r + c] = temp /100000 - 1;
        }
      }
    }

    void printShape() {
      std::cout << "Matrix Size([" << rows << ", " << cols << "])" << std::endl;
    }

    void print(){
      for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
          std::cout << data[cols * r + c] << " ";
        }
        std::cout << "" << std::endl;
      }
      std::cout << "" << std::endl;
    }
    // Overloading the () operator for indexing perpurses.
    Type& operator()(int row, int col) {
        return data[row * cols + col];
    }
    
    Matrix<Type> matmul(Matrix<Type> & target){
      assert(this->cols == target.rows);
      Matrix<Type> output(this->rows, target.cols);
      for(int r = 0; r < rows; r++){
        for(int c = 0; c < target.cols; c++){
          for(int k = 0; k < target.rows; k++){
            output(r,c) = output(r,c) + (*this)(r,k) * target(k,c);
          }
        }
      }
      return output;
    }

    Matrix multiply_elementwise(Matrix &target){
    assert(this->rows == target.rows);
    assert(this->cols == target.cols);
    Matrix output((*this));
    for (int r = 0; r < output.rows; ++r) {
      for (int c = 0; c < output.cols; ++c) {
        output(r, c) = target(r,c) * (*this)(r, c);
      }
    }
    return output;
  }

  Matrix multiply_scalar(Type scalar) {
    Matrix output((*this));
    for (int r = 0; r < output.rows; ++r) {
      for (int c = 0; c < output.cols; ++c) {
        output(r, c) = scalar * (*this)(r, c);
      }
    }
    return output;
  }

  Matrix add(Matrix &target) {
    assert(this->rows == target.rows);
    assert(this->cols == target.cols);
    Matrix output(rows, cols);

    for (int r = 0; r < output.rows; ++r) {
      for (int c = 0; c < output.cols; ++c) {
        output(r, c) = (*this)(r, c) + target(r, c);
      }
    }
    return output;
  }

  Matrix operator+(Matrix &target){
    return add(target);
  }

  Matrix sub(Matrix &target) {
    assert(this->rows == target.rows);
    assert(this->cols == target.cols);
    Matrix output(rows, cols);

    for (int r = 0; r < output.rows; ++r) {
      for (int c = 0; c < output.cols; ++c) {
        output(r, c) = (*this)(r, c) - target(r, c);
      }
    }
    return output;
  }

  Matrix operator-(Matrix &target){
    return sub(target);
  }

  Matrix transpose() {
    int new_rows = cols;
    int new_cols = rows;
    Matrix transposed(new_rows, new_cols);

    for (int r = 0; r < new_rows; ++r) {
      for (int c = 0; c < new_cols; ++c) {
        transposed(r, c) = (*this)(c, r);  // swap row and col
      }
    }
    return transposed;
  }

  Matrix T(){
    return transpose();
  }

  Matrix apply_function(const std::function<Type(const Type &)> &function) {
    Matrix output((*this));
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        output(r, c) = function((*this)(r, c));
      }
    }
    return output;
  }

};


