// Tests de la sauvegarde : `make test`. Code de sortie 0 si tout passe.
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>

#include "Artifacts.hpp"
#include "NeuralNetwork.hpp"

static int failures = 0;

static void check(bool ok, const std::string& name)
{
    std::cout << (ok ? "  ok    " : "  ÉCHEC ") << name << '\n';
    if (!ok) ++failures;
}

static bool throws(const std::function<void()>& f)
{
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

int main()
{
    const std::string dir = "/tmp/";
    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, 1.0);

    Matrix X(40, std::vector<double>(6)), Y(40, std::vector<double>(3));
    for (auto& r : X) for (double& v : r) v = noise(rng);
    for (size_t i = 0; i < X.size(); ++i)
        for (size_t k = 0; k < 3; ++k) Y[i][k] = X[i][k] * X[i][k + 1] + 0.1 * noise(rng);

    std::cout << "Réseau\n";
    NeuralNetwork original({6, 8, 5, 3}, Activation::Tanh, Activation::Linear, 0.01, 7);
    original.train(X, Y, 30, nullptr, nullptr, 0);   // poids entraînés, pas ceux de l'init
    original.save_binary(dir + "nn_test_model.bin");

    // Architecture, activations et graine différentes : tout doit venir du fichier.
    NeuralNetwork loaded({2, 2}, Activation::Sigmoid, Activation::Sigmoid, 0.5, 999);
    loaded.load_binary(dir + "nn_test_model.bin");

    check(loaded.architecture() == original.architecture(), "architecture restaurée");
    bool identical = true;
    for (int i = 0; i < 200; ++i) {
        std::vector<double> x(6);
        for (double& v : x) v = 3.0 * noise(rng);
        if (original.predict(x) != loaded.predict(x)) identical = false;   // égalité exacte
    }
    check(identical, "prédictions identiques au bit près sur 200 entrées");

    loaded.save_binary(dir + "nn_test_model_2.bin");
    auto read_all = [](const std::string& p) {
        std::ifstream in(p, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };
    check(read_all(dir + "nn_test_model.bin") == read_all(dir + "nn_test_model_2.bin"),
          "resauvegarde octet pour octet identique");

    std::cout << "Fichiers invalides\n";
    std::string bytes = read_all(dir + "nn_test_model.bin");
    auto write = [&](const std::string& p, const std::string& content) {
        std::ofstream(p, std::ios::binary) << content;
    };
    write(dir + "nn_bad_magic.bin", "XXXX" + bytes.substr(4));
    write(dir + "nn_truncated.bin", bytes.substr(0, bytes.size() / 2));
    write(dir + "nn_trailing.bin", bytes + "junk");
    NeuralNetwork victim({2, 2});
    check(throws([&] { victim.load_binary(dir + "nn_bad_magic.bin"); }), "mauvais en-tête refusé");
    check(throws([&] { victim.load_binary(dir + "nn_truncated.bin"); }), "fichier tronqué refusé");
    check(throws([&] { victim.load_binary(dir + "nn_trailing.bin"); }), "données en trop refusées");
    check(throws([&] { victim.load_binary(dir + "n_existe_pas.bin"); }), "fichier absent refusé");

    std::cout << "Scalers\n";
    Scalers sc;
    sc.feature_names = {"x_s1_GP", "x_s3_PTS_PER_GAME"};
    sc.target_names = {"y_PTS_PER_GAME"};
    sc.x.fit({{70, 12.3456789012345}, {55, 0.1}, {82, 31.4}});
    sc.y.fit({{1.0 / 3.0}, {2.0 / 7.0}, {25.5}});
    sc.save(dir + "nn_test_scalers.json");

    Scalers sc2;
    sc2.load(dir + "nn_test_scalers.json");
    check(sc2.feature_names == sc.feature_names && sc2.target_names == sc.target_names, "noms restaurés");
    std::vector<double> row = {61.0, 17.25}, yrow = {0.123456789};
    check(sc2.x.transform(row) == sc.x.transform(row), "Standardizer x identique au bit près");
    check(sc2.y.inverse(yrow) == sc.y.inverse(yrow), "Standardizer y identique au bit près");

    std::cout << (failures ? "\nDES TESTS ÉCHOUENT\n" : "\nTous les tests passent\n");
    return failures ? 1 : 0;
}
