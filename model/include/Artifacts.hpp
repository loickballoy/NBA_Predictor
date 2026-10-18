#pragma once

#include <map>
#include <string>
#include <vector>

#include "Dataset.hpp"

// Tout ce qu'il faut, en plus de model.bin, pour utiliser un modèle sans connaître le dataset
// d'origine : les deux Standardizer et les noms des colonnes attendues en entrée et en sortie.
struct Scalers {
    Standardizer x;
    Standardizer y;
    std::vector<std::string> feature_names;
    std::vector<std::string> target_names;

    void save(const std::string& path) const;   // scalers.json
    void load(const std::string& path);
};

// Empreinte FNV-1a 64 bits du contenu des fichiers, en hexadécimal :
// deux entraînements sur les mêmes données ont la même empreinte.
std::string hash_files(const std::vector<std::string>& paths);

std::string json_escape(const std::string& s);

// Index de la colonne baseline x_s3_<stat> pour chaque cible y_<stat>, -1 si absente.
std::vector<int> baseline_columns(const std::vector<std::string>& features,
                                  const std::vector<std::string>& targets);
