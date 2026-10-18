#include "Dataset.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

static std::vector<std::string> split_csv_line(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        if (!field.empty() && field.back() == '\r') field.pop_back();  // fichiers Windows
        fields.push_back(field);
    }
    if (!line.empty() && line.back() == ',') fields.emplace_back();
    return fields;
}

Dataset load_csv(const std::string& path)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Impossible d'ouvrir " + path);

    std::string line;
    if (!std::getline(file, line)) throw std::runtime_error(path + " est vide");

    Dataset ds;
    std::vector<std::string> header = split_csv_line(line);
    std::vector<size_t> x_idx, y_idx, id_idx;
    for (size_t i = 0; i < header.size(); ++i) {
        const std::string& name = header[i];
        if (name.rfind("x_", 0) == 0) { x_idx.push_back(i); ds.feature_names.push_back(name); }
        else if (name.rfind("y_", 0) == 0) { y_idx.push_back(i); ds.target_names.push_back(name); }
        else if (name.rfind("id_", 0) == 0) { id_idx.push_back(i); ds.id_names.push_back(name); }
    }
    if (x_idx.empty() || y_idx.empty())
        throw std::runtime_error(path + " : aucune colonne x_* ou y_* dans l'en-tête");

    size_t line_no = 1;
    while (std::getline(file, line)) {
        ++line_no;
        if (line.empty() || line == "\r") continue;
        std::vector<std::string> fields = split_csv_line(line);
        if (fields.size() != header.size())
            throw std::runtime_error(path + ":" + std::to_string(line_no) +
                                     " : nombre de colonnes incorrect");

        auto parse = [&](size_t col) {
            try {
                size_t used = 0;
                double v = std::stod(fields[col], &used);
                if (used != fields[col].size() || !std::isfinite(v)) throw std::invalid_argument("");
                return v;
            } catch (const std::exception&) {
                throw std::runtime_error(path + ":" + std::to_string(line_no) + " : valeur '" +
                                         fields[col] + "' invalide dans " + header[col]);
            }
        };

        std::vector<double> x, y, id;
        x.reserve(x_idx.size());
        y.reserve(y_idx.size());
        for (size_t c : x_idx) x.push_back(parse(c));
        for (size_t c : y_idx) y.push_back(parse(c));
        for (size_t c : id_idx) id.push_back(parse(c));
        ds.X.push_back(std::move(x));
        ds.Y.push_back(std::move(y));
        ds.ids.push_back(std::move(id));
    }
    return ds;
}

void Standardizer::fit(const Matrix& data)
{
    if (data.empty()) throw std::invalid_argument("Standardizer::fit sur des données vides");
    const size_t n = data.size(), d = data[0].size();
    mean.assign(d, 0.0);
    stddev.assign(d, 0.0);

    for (const auto& row : data)
        for (size_t j = 0; j < d; ++j) mean[j] += row[j];
    for (double& m : mean) m /= static_cast<double>(n);

    for (const auto& row : data)
        for (size_t j = 0; j < d; ++j) stddev[j] += (row[j] - mean[j]) * (row[j] - mean[j]);
    for (double& s : stddev) {
        s = std::sqrt(s / static_cast<double>(n));
        if (s < 1e-12) s = 1.0;  // colonne constante : on évite la division par 0
    }
}

std::vector<double> Standardizer::transform(const std::vector<double>& row) const
{
    std::vector<double> out(row.size());
    for (size_t j = 0; j < row.size(); ++j) out[j] = (row[j] - mean[j]) / stddev[j];
    return out;
}

std::vector<double> Standardizer::inverse(const std::vector<double>& row) const
{
    std::vector<double> out(row.size());
    for (size_t j = 0; j < row.size(); ++j) out[j] = row[j] * stddev[j] + mean[j];
    return out;
}

void Standardizer::transform_inplace(Matrix& data) const
{
    for (auto& row : data) row = transform(row);
}