
#pragma once
#include "matrix.h"
#include <vector>
// https://www.lyndonduong.com/mlp-build-cpp/

inline float sigmoid(float x) {
        return 1.0f / (1 + exp(-x));
    }

inline float d_sigmoid(float x){
    return (x * (1 - x));
    }
    


template<typename T>
class MLP{

    std::vector <int> unitsPerLayer;
    std::vector<Matrix<T> > bias_vectors;
    
    std::vector<Matrix<T> > activations;
    float lr;

    public:
    std::vector<Matrix<T> > weight_matrices;
    MLP(std::vector <int> units, float learningRate = 0.001f){
        unitsPerLayer = units;

        for(int i = 0; i < units.size() - 1; i++){
            int in_channels{units[i]};
            int out_channels{units[i+1]};

            Matrix<float> weights(out_channels, in_channels);
            weights.randInit();
            weight_matrices.push_back(weights);

            Matrix<float> bias(out_channels,1);
            bias.randInit();
            bias_vectors.push_back(bias);   
        }
    }

    

    Matrix<T> forward(Matrix<T> x) {
        
        assert(x.rows == unitsPerLayer[0]);
        activations.push_back(x);
        Matrix <T> prev(x);
        for (int i = 0; i < unitsPerLayer.size() - 1; ++i) {
            Matrix <T> y = weight_matrices[i].matmul(prev);
            y = y + bias_vectors[i];
            y = y.apply_function(sigmoid);
            activations.push_back(y);
            prev = y;
        }
        return prev;
    }


    void backProp(Matrix<T> target) {

        assert(target.rows == unitsPerLayer[unitsPerLayer.size() -1]);

        Matrix<T> y = target;
        Matrix<T> y_hat = activations.back();
        Matrix<T> PrevError = (target - y_hat);

        for (int i = weight_matrices.size() - 1; i >= 0; --i) {
            Matrix<T> Tweights = weight_matrices[i].T();
            Matrix<T> prev_errors = Tweights.matmul(PrevError);

            Matrix<T> d_outputs = activations[i+1].apply_function(d_sigmoid);
            Matrix<T> gradients = PrevError.multiply_elementwise(d_outputs);
            gradients = gradients.multiply_scalar(lr);

            Matrix<T> a_trans = activations[i].T();
            Matrix<T> weight_gradients = gradients.matmul(a_trans);

            weight_matrices[i] = weight_matrices[i].add(weight_gradients);
            bias_vectors[i] = bias_vectors[i].add(gradients);
            PrevError = activations[i] - prev_errors;
        }
    }
};


