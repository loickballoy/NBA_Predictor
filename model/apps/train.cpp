// Entraîne, évalue, sauvegarde.
//
//   bin/train --out artifacts/<version> [--data dir] [--epochs N] [--lr X] [--seed S] [--hidden 64,32,32]
//
// Écrit dans --out : model.bin, scalers.json, metrics.json.
// La validation sert à l'early stopping (et plus tard à la porte de qualité) ;
// le test n'est utilisé que pour le rapport final.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "Artifacts.hpp"
#include "Cli.hpp"
#include "Dataset.hpp"
#include "NeuralNetwork.hpp"

namespace fs = std::filesystem;

struct Evaluation {
    std::vector<double> mae, baseline_mae;
};

// MAE en unités réelles du modèle et de la baseline "dernière saison", par cible.
static Evaluation evaluate(NeuralNetwork& nn, const Scalers& sc, const Dataset& ds)
{
    const size_t n_out = sc.target_names.size();
    const std::vector<int> base = baseline_columns(sc.feature_names, sc.target_names);
    Evaluation ev{std::vector<double>(n_out, 0.0), std::vector<double>(n_out, 0.0)};

    for (size_t i = 0; i < ds.size(); ++i) {
        std::vector<double> pred = sc.y.inverse(nn.predict(sc.x.transform(ds.X[i])));
        for (size_t k = 0; k < n_out; ++k) {
            ev.mae[k] += std::fabs(pred[k] - ds.Y[i][k]);
            if (base[k] >= 0) ev.baseline_mae[k] += std::fabs(ds.X[i][base[k]] - ds.Y[i][k]);
        }
    }
    for (size_t k = 0; k < n_out; ++k) {
        ev.mae[k] /= ds.size();
        ev.baseline_mae[k] = base[k] >= 0 ? ev.baseline_mae[k] / ds.size() : NAN;
    }
    return ev;
}

static std::vector<int> parse_sizes(const std::string& text)
{
    std::vector<int> sizes;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        int v = std::stoi(token);
        if (v <= 0) throw std::invalid_argument("--hidden : tailles strictement positives attendues");
        sizes.push_back(v);
    }
    return sizes;
}

static std::string json_number(double v)
{
    if (!std::isfinite(v)) return "null";
    std::ostringstream out;
    out.precision(10);
    out << v;
    return out.str();
}

// Bloc JSON d'une évaluation, plus le score unique utilisé pour comparer deux modèles :
// la moyenne sur les cibles de MAE réseau / MAE baseline (< 1 = meilleur que la baseline).
static std::string evaluation_json(const Evaluation& ev, const std::vector<std::string>& targets,
                                   size_t n, const std::string& indent)
{
    auto block = [&](const std::string& name, auto value_of) {
        std::string s = indent + "  \"" + name + "\": {";
        for (size_t k = 0; k < targets.size(); ++k)
            s += (k ? ", \"" : "\"") + targets[k].substr(2) + "\": " + json_number(value_of(k));
        return s + "}";
    };
    double ratio_sum = 0.0;
    size_t ratio_n = 0;
    for (size_t k = 0; k < targets.size(); ++k)
        if (std::isfinite(ev.baseline_mae[k]) && ev.baseline_mae[k] > 0) {
            ratio_sum += ev.mae[k] / ev.baseline_mae[k];
            ++ratio_n;
        }

    std::string s = "{\n";
    s += indent + "  \"n_examples\": " + std::to_string(n) + ",\n";
    s += indent + "  \"mae_ratio_mean\": " + json_number(ratio_n ? ratio_sum / ratio_n : NAN) + ",\n";
    s += block("mae", [&](size_t k) { return ev.mae[k]; }) + ",\n";
    s += block("baseline_mae", [&](size_t k) { return ev.baseline_mae[k]; }) + ",\n";
    s += block("gain_pct", [&](size_t k) {
        return 100.0 * (ev.baseline_mae[k] - ev.mae[k]) / ev.baseline_mae[k];
    }) + "\n";
    return s + indent + "}";
}

static void print_table(const std::string& title, const Evaluation& ev, const std::vector<std::string>& targets)
{
    std::printf("\n%s\n%-16s %10s %12s %8s\n", title.c_str(), "stat", "réseau", "dern. saison", "gain");
    for (size_t k = 0; k < targets.size(); ++k) {
        std::printf("%-16s %10.3f", targets[k].substr(2).c_str(), ev.mae[k]);
        if (std::isfinite(ev.baseline_mae[k]) && ev.baseline_mae[k] > 0)
            std::printf(" %12.3f %+7.1f%%", ev.baseline_mae[k],
                        100.0 * (ev.baseline_mae[k] - ev.mae[k]) / ev.baseline_mae[k]);
        std::printf("\n");
    }
}

int main(int argc, char* argv[])
{
    try {
        Args args(argc, argv, {"data", "out", "epochs", "lr", "seed", "hidden"});
        const std::string data_dir = args.get("data", "../collector/data/processed");
        const fs::path out_dir = args.require("out");
        const int epochs = std::stoi(args.get("epochs", "1000"));
        const double lr = std::stod(args.get("lr", "0.001"));
        const unsigned seed = static_cast<unsigned>(std::stoul(args.get("seed", "42")));
        const std::vector<int> hidden = parse_sizes(args.get("hidden", "64,32,32"));

        const std::vector<std::string> files = {data_dir + "/train.csv", data_dir + "/val.csv",
                                                data_dir + "/test.csv"};
        Dataset train = load_csv(files[0]), val = load_csv(files[1]), test = load_csv(files[2]);
        for (const Dataset* d : {&val, &test})
            if (d->feature_names != train.feature_names || d->target_names != train.target_names)
                throw std::runtime_error("Les colonnes de val/test ne correspondent pas au train");

        Scalers sc;
        sc.feature_names = train.feature_names;
        sc.target_names = train.target_names;
        sc.x.fit(train.X);                       // appris sur le train uniquement
        sc.y.fit(train.Y);

        Matrix Xtr = train.X, Ytr = train.Y, Xva = val.X, Yva = val.Y;
        sc.x.transform_inplace(Xtr);
        sc.x.transform_inplace(Xva);
        sc.y.transform_inplace(Ytr);
        sc.y.transform_inplace(Yva);

        std::vector<int> sizes{static_cast<int>(sc.feature_names.size())};
        sizes.insert(sizes.end(), hidden.begin(), hidden.end());
        sizes.push_back(static_cast<int>(sc.target_names.size()));

        std::cout << train.size() << " train, " << val.size() << " val, " << test.size() << " test\n";
        NeuralNetwork nn(sizes, Activation::Tanh, Activation::Linear, lr, seed);
        nn.train(Xtr, Ytr, epochs, &Xva, &Yva, 50);

        Evaluation ev_val = evaluate(nn, sc, val);
        Evaluation ev_test = evaluate(nn, sc, test);
        print_table("MAE validation", ev_val, sc.target_names);
        print_table("MAE test (rapport final uniquement)", ev_test, sc.target_names);

        fs::create_directories(out_dir);
        nn.save_binary((out_dir / "model.bin").string());
        sc.save((out_dir / "scalers.json").string());

        std::string arch = "[";
        for (size_t i = 0; i < sizes.size(); ++i) arch += (i ? ", " : "") + std::to_string(sizes[i]);
        arch += "]";

        std::ofstream m(out_dir / "metrics.json");
        if (!m) throw std::runtime_error("Impossible d'écrire metrics.json");
        m << "{\n"
          << "  \"model\": {\"architecture\": " << arch
          << ", \"hidden_activation\": \"tanh\", \"output_activation\": \"linear\"},\n"
          << "  \"hyperparameters\": {\"epochs\": " << epochs << ", \"learning_rate\": "
          << json_number(lr) << ", \"seed\": " << seed << "},\n"
          << "  \"dataset\": {\"hash\": \"fnv1a64:" << hash_files(files) << "\", \"n_train\": "
          << train.size() << ", \"n_val\": " << val.size() << ", \"n_test\": " << test.size()
          << ", \"n_features\": " << sc.feature_names.size() << ", \"n_targets\": "
          << sc.target_names.size() << "},\n"
          << "  \"validation\": " << evaluation_json(ev_val, sc.target_names, val.size(), "  ") << ",\n"
          << "  \"test\": " << evaluation_json(ev_test, sc.target_names, test.size(), "  ") << "\n"
          << "}\n";

        std::cout << "\nArtefacts écrits dans " << out_dir << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << '\n';
        return 1;
    }
    return 0;
}
