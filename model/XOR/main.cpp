#include "NeuralNetwork.hpp"
#include <cstring>

int main(int argc, char* argv[]) {

    bool live = argc == 2 && std::strcmp(argv[1], "--live") == 0;
    std::cout << argc << " " << argv[1] << " " << live << std::endl;
    
    if (argc > 3 || !live){
        std::cerr << "Usage" << argv[0] << " <--live>\n";
        return 1; 
    }
    
    std::vector<std::vector<double>> trainingInputs = {{0, 0}, {0, 1}, {1, 0}, {1,1}};
    std::vector<std::vector<double>> outputs = {{0}, {1}, {1}, {0}};
    std::vector<int> layers = {2, 2, 1};
    NeuralNetwork nn(layers);

    nn.train(trainingInputs, outputs);

    if (live){
        while(1){
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
    }


    Neuron n(5);
    n.printNeuron();

    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};
    std::uniform_real_distribution<> dist(0.0, 1.0);

    auto gen = [&]() {
        return dist(mersenne_engine);
    };

    std::vector<double> inputs(5);
    std::generate(inputs.begin(), inputs.end(), gen);

    double output = n.activate(inputs);
    std::cout << "output: " << output << std::endl;

    Layer l(2, 5);
    l.printLayer();

    

    std::vector<std::vector<double>> out;
    for (auto in : trainingInputs){
        out.emplace_back(nn.predict(in));
    }

    for (auto o : out){
        for (auto act : o){
            std::cout << act << std::endl;
        }
    }

    return 0;    
}