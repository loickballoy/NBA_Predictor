#pragma once

#include "NeuralNetwork.hpp"

#include <string>
#include <vector>

struct Dataset {
    std::vector<std::string> feature_names;
    std::vector<std::string> target_names;
    std::vector<std::string> id_names;
    Matrix X;
    Matrix Y;
    Matrix ids;

    size_t size() const { return X.size();}
};

Dataset load_csv(const std::string& path);

class Standardizer {
public:
    void fit(const Matrix& data);
    std::vector<double> transform(const std::vector<double>& row) const;
    std::vector<double> inverse(const std::vector<double>& row) const;
    void transform_inplace(Matrix& data) const;
private:
    std::vector<double> mean;
    std::vector<double> stddev;
};