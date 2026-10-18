#include "NeuralNetwork.hpp"

#include <cstring>
#include <iostream>
#include <limits>
#include <string>

int main(int argc, char* argv[]) {

    bool live = argc == 2 && std::strcmp(argv[1], "--live") == 0;
    
    if (argc > 2 || !live){
        std::cerr << "Usage" << argv[0] << " <--live>\n";
        return 1; 
    }
    
    Matrix X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    Matrix Y = {{0}, {1}, {1}, {0}};

    NeuralNetwork nn({2, 4, 1}, Activation::Sigmoid, Activation::Sigmoid, 0.5);
    nn.train(X, Y, 5000, nullptr, nullptr, 1000);

    for (const auto& x : X)
        std::cout << "XOR(" << x[0] << ", " << x[1] << ") -> " << nn.predict(x)[0] << '\n';

    while (live) {
        std::cout << "Input (e.g 1 0)\n";
            int a, b;

            if (!(std::cin >> a)) {
                std::cin.clear();
                std::string token;
                std::cin >> token;

                if (token == "q") break;

                std::cout << "Invalid input. Enter two integers 0 or 1.\n";
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                continue;
            }

            if (!(std::cin >> b)) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "You must enter two values.\n";
                continue;
            }

            if ((a != 0 && a != 1) || (b != 0 && b != 1)) {
                std::cout << "Inputs must be 0 or 1.\n";
                continue;
            }

            std::vector<double> prob = nn.predict({a, b});
            int prediction = prob[0] > 0.5 ? 1: 0;

            std::cout << "NN output: " << prob[0]
                      << " -> XOR(" << a << ", " << b << ") = "
                      << prediction << "\n\n";
    }

    return 0;  
}