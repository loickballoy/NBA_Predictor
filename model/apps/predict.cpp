// Charge un modèle sauvegardé et prédit pour un fichier d'entrée.
//
//   bin/predict --model artifacts/<version> --input fichier.csv [--output predictions.csv]
//
// Le fichier d'entrée doit avoir les mêmes colonnes x_* que l'entraînement. Les colonnes y_*
// sont facultatives : si elles sont là, la sortie contient aussi les valeurs réelles et la MAE.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "Artifacts.hpp"
#include "Cli.hpp"
#include "Dataset.hpp"
#include "NeuralNetwork.hpp"

namespace fs = std::filesystem;

int main(int argc, char* argv[])
{
    try {
        Args args(argc, argv, {"model", "input", "output"});
        const fs::path model_dir = args.require("model");
        const std::string input = args.require("input");
        const std::string output = args.get("output", (model_dir / "predictions.csv").string());

        NeuralNetwork nn({1, 1});
        nn.load_binary((model_dir / "model.bin").string());
        Scalers sc;
        sc.load((model_dir / "scalers.json").string());

        if (nn.input_size() != sc.feature_names.size() || nn.output_size() != sc.target_names.size())
            throw std::runtime_error("model.bin et scalers.json ne viennent pas du même entraînement");

        Dataset ds = load_csv(input, /*require_targets=*/false);
        if (ds.feature_names != sc.feature_names)
            throw std::runtime_error(input + " : colonnes x_* différentes de celles du modèle");
        const bool has_truth = !ds.target_names.empty();
        if (has_truth && ds.target_names != sc.target_names)
            throw std::runtime_error(input + " : colonnes y_* différentes de celles du modèle");

        const std::vector<int> base = baseline_columns(sc.feature_names, sc.target_names);
        const size_t n_out = sc.target_names.size();

        std::ofstream out(output);
        if (!out) throw std::runtime_error("Impossible d'écrire " + output);
        out.precision(10);

        std::vector<std::string> header = ds.id_names;
        for (size_t k = 0; k < n_out; ++k) {
            std::string stat = sc.target_names[k].substr(2);
            if (has_truth) header.push_back(stat + "_true");
            header.push_back(stat + "_pred");
            if (base[k] >= 0) header.push_back(stat + "_last");
        }
        for (size_t c = 0; c < header.size(); ++c) out << (c ? "," : "") << header[c];
        out << '\n';

        std::vector<double> mae(n_out, 0.0);
        for (size_t i = 0; i < ds.size(); ++i) {
            std::vector<double> pred = sc.y.inverse(nn.predict(sc.x.transform(ds.X[i])));
            bool first = true;
            auto cell = [&](double v) { out << (first ? "" : ",") << v; first = false; };

            for (double id : ds.ids[i]) { out << (first ? "" : ",") << static_cast<long long>(id); first = false; }
            for (size_t k = 0; k < n_out; ++k) {
                if (has_truth) { cell(ds.Y[i][k]); mae[k] += std::fabs(pred[k] - ds.Y[i][k]); }
                cell(pred[k]);
                if (base[k] >= 0) cell(ds.X[i][base[k]]);
            }
            out << '\n';
        }

        std::cout << ds.size() << " prédictions écrites dans " << output << '\n';
        if (has_truth) {
            std::printf("%-16s %10s\n", "stat", "MAE");
            for (size_t k = 0; k < n_out; ++k)
                std::printf("%-16s %10.3f\n", sc.target_names[k].substr(2).c_str(), mae[k] / ds.size());
        }
    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << '\n';
        return 1;
    }
    return 0;
}
