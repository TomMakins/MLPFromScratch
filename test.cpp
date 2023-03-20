#include "matrix.h"
#include "nn.h"
#include <iostream>
#include <fstream>

//  g++ -std=c++11 -o exe MLP.cpp
using namespace std;


 const float PI {3.14159};
void run(){

  std::vector<int> units = {1,3,3,3,1};
  MLP<float> model(units, 0.01f);
  cout << "Initilised" << endl;

  int max_iter{1000};
  float mse;

  ofstream MyFile("output.txt");

 
  for(int i = 1; i<=max_iter; ++i) {

    // generate (x, y) training data: y = sin^2(x)
    Matrix<float> x(1,1);
    x.randInit();
    x.multiply_scalar(PI);
    Matrix<float> y(1,1);
    y = x.apply_function([](float v) -> float { return sin(v) * sin(v); });

    // forward and backward
    Matrix<float> y_hat(1,1);



    y_hat = model.forward(x); 
    model.backProp(y); // loss and grads computed in here

    // function that logs (loss, x, y, y_hat)
    MyFile << x.data[0] << " " << y.data[0] << " " << y_hat.data[0] << endl;
  }

  cout << "Finished" << endl;

  MyFile.close();

}

void test(){
   std::vector<int> units = {1,3,3,3,1};
  MLP<float> model(units, 0.001f);

  Matrix<float> x(1,1);
  x.randInit();
  x.multiply_scalar(PI);
  Matrix<float> y(1,1);
  y = x.apply_function([](float v) -> float { return sin(v) * sin(v); });

  Matrix<float> y_hat(1,1);

  for(int j = 0; j < model.weight_matrices.size(); j++){
    Matrix<float> temp = model.weight_matrices[j].T();
    temp.printShape();
  }

  y_hat = model.forward(x); 
  model.backProp(y);
        
}


int main(){
 run();

 
}