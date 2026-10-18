#pragma once

#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <cmath>

enum class Activation {Sigmoid, Tanh, ReLU, Linear};

class Neuron {
public:
    Neuron(int input_number, Activation activation, std::mt19937& rng);

    void printNeuron();
    double activate(const std::vector<double>& inputs);
    void update_weights(double delta, double learning_rate);
    double sigmoid(double x){return 1.0 / (1.0 + std::exp(-x));}
    double derivative() const;
    
    std::vector<double> weights;
    double bias;
private:
    Activation activation;
    std::vector<double> inputs;
    double z = 0.0;
    double output;
};

class Layer {
public:
    Layer(int num_neurons, int input_number_per_neuron, Activation activation, std::mt19937& rng);
    std::vector<double> forward(const std::vector<double>& inputs);
    std::vector<double> backward(const std::vector<double>& errors, double learning_rate); 
    void printLayer();
private:
    std::vector<Neuron> neurons;
};

using Matrix = std::vector<std::vector<double>>;

class NeuralNetwork{
public:
    NeuralNetwork(const std::vector<int>& sizes,
                    Activation hidden = Activation::Tanh,
                    Activation output = Activation::Linear,
                    double learning_rate=0.1, 
                    unsigned seed = 42); 
    
    void train(const Matrix& X, const Matrix& Y, int epochs,
                const Matrix* X_val = nullptr, const Matrix* Y_val = nullptr,
                int log_every = 100);

    std::vector<double> predict(const std::vector<double>& input);
    double mse(const Matrix& X, const Matrix& Y);
private:
    double learning_rate;
    std::mt19937 rng;
    std::vector<Layer> layers;
};
