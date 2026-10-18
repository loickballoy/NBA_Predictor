#include "NeuralNetwork.hpp"

Neuron::Neuron(int input_number, Activation activation, std::mt19937& rng)
    : weights(input_number), activation(activation)
{
    double limit = 1.0 / std::sqrt(static_cast<double>(input_number));
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (double& w: weights) w = dist(rng);
    bias = (activation == Activation::ReLU) ? 0.01 : 0.0;
}

void Neuron::printNeuron(){
    std::cout << "weights : ";
    for (double w : this->weights){
        std::cout << w << " ";
    }
    std::cout << std::endl;
    std::cout << "bias : " << this->bias << std::endl;
}

double Neuron::activate(const std::vector<double>& inputs){
    if (inputs.size() != this->weights.size()) throw std::invalid_argument("Incorrect Input size");
    this->inputs = inputs;

    z = this->bias;
    for (size_t i = 0; i < inputs.size(); i++) z += this->weights[i] * inputs[i];

    switch (this->activation) {
        case Activation::Sigmoid:   this->output = 1.0 / (1.0 + std::exp(-z)); break;
        case Activation::Tanh:      this->output = std::tanh(z); break;
        case Activation::ReLU:      this->output = z > 0.0 ? z : 0.0; break;
        case Activation::Linear:    this->output = z; break;
    }

    return this->output;
}

double Neuron::derivative() const{
    switch (this->activation) {
        case Activation::Sigmoid:       return this->output * (1.0 - output);
        case Activation::Tanh:          return 1.0 - this->output * this->output;
        case Activation::ReLU:          return z > 0.0 ? 1.0 : 0.0;
        case Activation::Linear:         return 1.0; 
    }
    return 1.0;
}

void Neuron::update_weights(double delta, double learning_rate){
    this->bias += learning_rate * delta;
    for (int i = 0; i < inputs.size(); i++){
        this->weights[i] += learning_rate * delta * inputs[i];
    }
}

// LAYER

Layer::Layer(int num_neurons, int num_inputs_per_neuron, Activation activation, std::mt19937& rng){
    neurons.reserve(num_neurons);
    for (int i = 0; i < num_neurons; i++){
        this->neurons.emplace_back(num_inputs_per_neuron, activation, rng);
    }
}

std::vector<double> Layer::forward(const std::vector<double>& inputs){
    std::vector<double> outputs;
    outputs.reserve(this->neurons.size());
    for (auto& neuron : this->neurons){
        outputs.emplace_back(neuron.activate(inputs));
    }
    return outputs;
}

std::vector<double> Layer::backward(const std::vector<double>& errors, double learning_rate){
    std::vector<double> prev(this->neurons[0].weights.size(), 0.0);
    for (size_t i = 0; i < this->neurons.size(); i++){
        double delta = errors[i] * this->neurons[i].derivative();
        for (size_t j = 0; j < prev.size(); j++){
            prev[j] += delta * this->neurons[i].weights[j];
        } 
        this->neurons[i].update_weights(delta, learning_rate);
    }
    return prev;
}

void Layer::printLayer(){
    for (auto neuron : this->neurons){
        neuron.printNeuron();
    }
}

// NEURAL NETWORK

NeuralNetwork::NeuralNetwork(const std::vector<int>& sizes, Activation hidden, Activation output, double learning_rate, unsigned seed)
    : learning_rate(learning_rate), rng(seed)
{
    if (sizes.size() < 2) throw std::invalid_argument("Il faut au moins une couche entrée et une de sortie");
    for (size_t i = 1; i < sizes.size(); i++) {
        Activation act = (i == sizes.size() - 1) ? output : hidden;
        layers.emplace_back(sizes[i], sizes[i-1], act, rng);
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    std::vector<double> activations = input;

    for (Layer& layer : layers) {
        activations = layer.forward(activations);
    }

    return activations;
}

double NeuralNetwork::mse(const Matrix& X,const Matrix& Y){
    double total = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < X.size(); i++) {
        std::vector<double> p = predict(X[i]);
        for (size_t k = 0; k < p.size(); k++){
            double e = Y[i][k] - p[k];
            total += e*e;
            count++;
        }
    }

    return count ? total / static_cast<double>(count): 0.0;
}

void NeuralNetwork::train(const Matrix& X, const Matrix& Y, int epochs,
                            const Matrix* X_val, const Matrix* Y_val, int log_every){
    if (X.size() != Y.size() || X.empty()) throw std::invalid_argument("X and Y should have the same num of lines, non-null");
    
    const bool has_val = X_val && Y_val && !X_val->empty();
    double best_val = std::numeric_limits<double>::infinity();
    int best_epoch = -1;
    std::vector<Layer> best_layers = layers;

    std::vector<size_t> order(X.size());
    std::iota(order.begin(), order.end(), 0);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        std::shuffle(order.begin(), order.end(), rng);
        double total = 0.0;
        for(size_t idx : order){
            std::vector<double> pred = predict(X[idx]);
            if (pred.size() != Y[idx].size()) throw std::invalid_argument("Incorrect output size");
            
            std::vector<double> errors(pred.size());
            for (size_t k = 0; k < pred.size(); k++) {
                errors[k] = Y[idx][k] - pred[k];
                total += errors[k] * errors[k];
            }
            for (int l = static_cast<int>(layers.size()) - 1; l >= 0; --l)
                errors = layers[l].backward(errors, learning_rate);
        }
        double train_mse = total / static_cast<double>(X.size() * Y[0].size());
        double val_mse = 0.0;
        if (has_val) {
            val_mse = mse(*X_val, *Y_val);
            if (val_mse < best_val) {
                best_val = val_mse;
                best_epoch = epoch;
                best_layers = layers;
            }
        }

        if (log_every > 0 && (epoch % log_every == 0 || epoch == 1)){
            std::cout << "Epoch " << epoch << " train MSE " << train_mse;
            if (has_val) std::cout << "  val MSE " << val_mse;
            std::cout << '\n';
        }
    }

    if (has_val){
        layers = best_layers;
        std::cout << "Meilleur val MSE " << best_val << " (epoch " << best_epoch << ")\n";
    }
}

