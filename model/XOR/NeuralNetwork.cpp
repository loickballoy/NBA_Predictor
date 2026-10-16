#include "NeuralNetwork.hpp"

Neuron::Neuron(int input_number){
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};
    std::uniform_real_distribution<> dist(0.0, 1.0);

    auto gen = [&]() {
        return dist(mersenne_engine);
    };

    this->weights.resize(input_number);
    std::generate(this->weights.begin(), this->weights.end(), gen);
    this->bias = gen();
    this->output = 0.0;
}

void Neuron::printNeuron(){
    std::cout << "weights : ";
    for (double w : this->weights){
        std::cout << w << " ";
    }
    std::cout << std::endl;
    std::cout << "bias : " << this->bias << std::endl;
}

double Neuron::activate(std::vector<double> inputs){
    this->inputs = inputs;
    if (inputs.size() != this->weights.size()) {throw "Incorrect Input size";}

    double dot = 0;
    for (int i = 0; i < inputs.size(); i++){
        dot += this->weights[i] * this->inputs[i];
    }
    this->output = this->sigmoid(dot + this->bias);
    return this->output;
}

void Neuron::update_weights(double delta, double learning_rate){
    this->bias += learning_rate * delta;
    for (int i = 0; i < inputs.size(); i++){
        this->weights[i] += learning_rate * delta * inputs[i];
    }
}

Layer::Layer(int num_neurons, int num_inputs_per_neuron){
    for (int i = 0; i < num_neurons; i++){
        this->neurons.emplace_back(Neuron(num_inputs_per_neuron));
    }
}

std::vector<double> Layer::forward(std::vector<double> inputs){
    std::vector<double> outputs;
    for (auto& neuron : this->neurons){
        outputs.emplace_back(neuron.activate(inputs));
    }
    return outputs;
}

std::vector<double> Layer::backward(std::vector<double> errors, double learning_rate){
    std::vector<double> res(this->neurons[0].weights.size(), 0.0);
    for (int i = 0; i < this->neurons.size(); i++){
        double delta = errors[i] * this->neurons[i].sigmoid_derivative();
        for (int j = 0; j < this->neurons[i].weights.size(); j++){
            res[j] += delta * this->neurons[i].weights[j];
        } 
        this->neurons[i].update_weights(delta, learning_rate);
    }
    return res;

}

void Layer::printLayer(){
    for (auto neuron : this->neurons){
        neuron.printNeuron();
    }
}

void NeuralNetwork::train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs){
    for (auto epoch = 0; epoch < this->epochs; epoch++){
        double total_error = 0.0;
        for (int i = 0; i < inputs.size(); i++){
            std::vector<double> x = inputs[i];
            std::vector<double> y = outputs[i];

            std::vector<std::vector<double>> activations;
            activations.reserve(layers.size() + 1);
            activations.push_back(x);

            for (Layer& layer : layers) {
                activations.push_back(layer.forward(activations.back()));
            }

            // Calculate error
            const std::vector<double>& prediction = activations.back();

            if (prediction.size() != y.size()) {
                throw std::invalid_argument("Output size mismatch");
            }

            std::vector<double> output_errors(y.size());

            for (size_t i = 0; i < y.size(); ++i) {
                output_errors[i] = y[i] - prediction[i];
                total_error += output_errors[i] * output_errors[i];
            }

            // Backward pass
            std::vector<double> errors = output_errors;

            for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
                errors = layers[i].backward(errors, learning_rate);
            }
        }

        if (epoch % 1000 == 0) {
            double mse = total_error / static_cast<double>(inputs.size());
            std::cout << "Epoch " << epoch
                      << ", MSE: " << mse << std::endl;
        }
    }
}

std::vector<double> NeuralNetwork::predict(std::vector<double> input) {
    std::vector<double> activations = input;

    for (Layer& layer : layers) {
        activations = layer.forward(activations);
    }

    return activations;
}