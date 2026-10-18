#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>

#include "Dataset.hpp"
#include "NeuralNetwork.hpp"

// "2024" -> "2024-25"
static std::string season_label(double start_year)
{
    int y = static_cast<int>(start_year);
    char buf[16];
    std::snprintf(buf, sizeof buf, "%d-%02d", y, (y + 1) % 100);
    return buf;
}

// Valeur d'un identifiant (id_PLAYER_ID, id_TARGET_YEAR...) ou -1 s'il est absent.
static double id_value(const Dataset& ds, size_t row, const std::string& name)
{
    for (size_t j = 0; j < ds.id_names.size(); ++j)
        if (ds.id_names[j] == name) return ds.ids[row][j];
    return -1.0;
}

// Usage : bin/nba [dossier_données] [epochs] [learning_rate] [nb_exemples]
int main(int argc, char* argv[])
{
    const std::string dir = argc > 1 ? argv[1] : "../collector/data/processed";
    const int epochs = argc > 2 ? std::stoi(argv[2]) : 1000;
    const double lr = argc > 3 ? std::stod(argv[3]) : 0.001;
    const int n_examples = argc > 4 ? std::stoi(argv[4]) : 5;

    try {
        Dataset train = load_csv(dir + "/train.csv");
        Dataset val = load_csv(dir + "/val.csv");
        Dataset test = load_csv(dir + "/test.csv");

        for (const Dataset* d : {&val, &test})
            if (d->feature_names != train.feature_names || d->target_names != train.target_names)
                throw std::runtime_error("Les colonnes de val/test ne correspondent pas au train");

        std::cout << train.size() << " train, " << val.size() << " val, " << test.size()
                  << " test | " << train.feature_names.size() << " features -> "
                  << train.target_names.size() << " cibles\n";

        // Normalisation apprise sur le train seulement (sinon fuite d'information).
        Standardizer sx, sy;
        sx.fit(train.X);
        sy.fit(train.Y);
        Matrix Xtr = train.X, Ytr = train.Y, Xva = val.X, Yva = val.Y, Xte = test.X;
        sx.transform_inplace(Xtr);
        sx.transform_inplace(Xva);
        sx.transform_inplace(Xte);
        sy.transform_inplace(Ytr);
        sy.transform_inplace(Yva);

        const int n_in = static_cast<int>(train.feature_names.size());
        const int n_out = static_cast<int>(train.target_names.size());
        NeuralNetwork nn({n_in, 64, 32, 32, n_out}, Activation::Tanh, Activation::Linear, lr);
        nn.train(Xtr, Ytr, epochs, &Xva, &Yva, 10);

        // Nom court des cibles : "y_PTS_PER_GAME" -> "PTS_PER_GAME"
        std::vector<std::string> stat(n_out);
        for (int k = 0; k < n_out; ++k) stat[k] = train.target_names[k].substr(2);

        // Baseline : "le joueur refait exactement sa dernière saison" (x_s3_<stat>).
        std::vector<int> baseline_col(n_out, -1);
        for (int k = 0; k < n_out; ++k)
            for (int j = 0; j < n_in; ++j)
                if (train.feature_names[j] == "x_s3_" + stat[k]) baseline_col[k] = j;
        auto baseline = [&](size_t i, int k) {
            return baseline_col[k] >= 0 ? test.X[i][baseline_col[k]] : NAN;
        };

        // Prédictions sur tout le test, en unités réelles.
        Matrix preds(test.size());
        for (size_t i = 0; i < test.size(); ++i) preds[i] = sy.inverse(nn.predict(Xte[i]));

        // ---------- 1. Erreur moyenne par stat ----------
        std::cout << "\nMAE sur le test (unités réelles)\n";
        std::printf("%-16s %10s %12s %8s\n", "stat", "réseau", "dern. saison", "gain");
        for (int k = 0; k < n_out; ++k) {
            double m = 0.0, b = 0.0;
            for (size_t i = 0; i < test.size(); ++i) {
                m += std::fabs(preds[i][k] - test.Y[i][k]);
                if (baseline_col[k] >= 0) b += std::fabs(baseline(i, k) - test.Y[i][k]);
            }
            m /= test.size();
            b /= test.size();
            std::printf("%-16s %10.3f", stat[k].c_str(), m);
            if (baseline_col[k] >= 0) std::printf(" %12.3f %+7.1f%%", b, 100.0 * (b - m) / b);
            std::printf("\n");
        }

        // ---------- 2. Attendu vs prédit, sur quelques joueurs tirés au hasard ----------
        std::vector<size_t> order(test.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 rng(7);
        std::shuffle(order.begin(), order.end(), rng);

        const size_t shown = std::min<size_t>(std::max(n_examples, 0), test.size());
        for (size_t e = 0; e < shown; ++e) {
            size_t i = order[e];
            std::printf("\n=== Joueur %.0f, saison %s ===\n", id_value(test, i, "id_PLAYER_ID"),
                        season_label(id_value(test, i, "id_TARGET_YEAR")).c_str());
            std::printf("%-16s %9s %9s %9s %12s\n", "stat", "attendu", "réseau", "écart", "dern. saison");
            for (int k = 0; k < n_out; ++k) {
                double real = test.Y[i][k], pred = preds[i][k];
                std::printf("%-16s %9.2f %9.2f %+9.2f", stat[k].c_str(), real, pred, pred - real);
                if (baseline_col[k] >= 0) std::printf(" %12.2f", baseline(i, k));
                std::printf("\n");
            }
        }

        // ---------- 3. Export complet pour tracer les graphiques dans le notebook ----------
        const std::string out_path = dir + "/predictions_test.csv";
        std::ofstream out(out_path);
        if (!out) throw std::runtime_error("Impossible d'écrire " + out_path);
        for (const auto& name : test.id_names) out << name << ',';
        for (int k = 0; k < n_out; ++k)
            out << stat[k] << "_true," << stat[k] << "_pred," << stat[k] << "_last"
                << (k + 1 < n_out ? "," : "\n");
        for (size_t i = 0; i < test.size(); ++i) {
            for (double v : test.ids[i]) out << static_cast<long long>(v) << ',';
            for (int k = 0; k < n_out; ++k) {
                out << test.Y[i][k] << ',' << preds[i][k] << ',';
                if (baseline_col[k] >= 0) out << baseline(i, k);  // vide si pas de baseline
                out << (k + 1 < n_out ? "," : "\n");
            }
        }
        std::cout << "\nPrédictions du test écrites dans " << out_path << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << '\n';
        return 1;
    }
    return 0;
}