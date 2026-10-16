#pragma once

#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <cmath>

class Neuron {
public:
    Neuron(int input_number);
    void printNeuron();
    double activate(std::vector<double> inputs);
    void update_weights(double delta, double learning_rate);
    double sigmoid(double x){return 1.0 / (1.0 + std::exp(-x));}
    double sigmoid_derivative(){return this->output * (1 - this->output);}
    
    std::vector<double> inputs;
    std::vector<double> weights;
    double output;
    double bias;
};

class Layer {
public:
    Layer(int num_neurons, int input_number_per_neuron);
    std::vector<double> forward(std::vector<double> inputs);
    std::vector<double> backward(std::vector<double> errors, double learning_rate); 
    void printLayer();
private:
    std::vector<Neuron> neurons;
};

class NeuralNetwork{
public:
    NeuralNetwork(std::vector<int> layers, double learning_rate=0.1, int epochs=10000) 
    {
        this->learning_rate=learning_rate;
        this->epochs=epochs;

        for (int i = 0; i < layers.size() - 1; i++){
            this->layers.emplace_back(Layer(layers[i+1], layers[i]));
        }
    }
    void train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs);
    std::vector<double> predict(std::vector<double> input);
private:
    double learning_rate;
    int epochs;
    std::vector<Layer> layers;
};
