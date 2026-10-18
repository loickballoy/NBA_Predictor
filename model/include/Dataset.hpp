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

// require_targets = false : accepte un fichier sans colonnes y_* (données à prédire).
Dataset load_csv(const std::string& path, bool require_targets = true);

class Standardizer {
public:
    void fit(const Matrix& data);
    std::vector<double> transform(const std::vector<double>& row) const;
    std::vector<double> inverse(const std::vector<double>& row) const;
    void transform_inplace(Matrix& data) const;

    size_t size() const { return mean.size(); }

    std::string to_json() const;                   // {"mean": [...], "stddev": [...]}
    void from_json(const std::string& content);

    void save_json(const std::string& path) const;
    void load_json(const std::string& path);
private:
    std::vector<double> mean;
    std::vector<double> stddev;
};