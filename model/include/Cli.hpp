#pragma once

#include <map>
#include <set>
#include <stdexcept>
#include <string>

// Arguments de la forme --cle valeur. Toute clé non déclarée est refusée,
// pour qu'une faute de frappe (--epoch au lieu de --epochs) ne passe pas en silence.
class Args {
public:
    Args(int argc, char* argv[], const std::set<std::string>& allowed)
    {
        for (int i = 1; i < argc; i += 2) {
            std::string key = argv[i];
            if (key.rfind("--", 0) != 0 || !allowed.count(key.substr(2)))
                throw std::invalid_argument("option inconnue : " + key);
            if (i + 1 >= argc) throw std::invalid_argument("valeur manquante pour " + key);
            values[key.substr(2)] = argv[i + 1];
        }
    }

    std::string get(const std::string& key, const std::string& fallback) const
    {
        auto it = values.find(key);
        return it == values.end() ? fallback : it->second;
    }

    std::string require(const std::string& key) const
    {
        auto it = values.find(key);
        if (it == values.end()) throw std::invalid_argument("option obligatoire manquante : --" + key);
        return it->second;
    }

private:
    std::map<std::string, std::string> values;
};
