#include "Artifacts.hpp"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <stdexcept>

std::string json_escape(const std::string& s)
{
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

static std::string json_string_array(const std::vector<std::string>& values)
{
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i)
        out += (i ? ", \"" : "\"") + json_escape(values[i]) + "\"";
    return out + "]";
}

// Lit ["a", "b"] après la clé donnée. Suffisant pour nos noms de colonnes
// (pas de virgule, de crochet ni de guillemet échappé dedans).
static std::vector<std::string> parse_string_array(const std::string& content, const std::string& key)
{
    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) throw std::runtime_error("scalers.json : clé manquante " + key);
    size_t open = content.find('[', key_pos);
    size_t close = content.find(']', open);
    if (open == std::string::npos || close == std::string::npos)
        throw std::runtime_error("scalers.json : tableau invalide pour " + key);

    std::vector<std::string> values;
    std::stringstream ss(content.substr(open + 1, close - open - 1));
    std::string token;
    while (std::getline(ss, token, ',')) {
        size_t a = token.find('"'), b = token.rfind('"');
        if (a == std::string::npos || b <= a) throw std::runtime_error("scalers.json : nom invalide dans " + key);
        values.push_back(token.substr(a + 1, b - a - 1));
    }
    return values;
}

// Renvoie le texte de l'objet {...} associé à la clé (sans objet imbriqué à l'intérieur).
static std::string extract_object(const std::string& content, const std::string& key)
{
    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) throw std::runtime_error("scalers.json : clé manquante " + key);
    size_t open = content.find('{', key_pos);
    size_t close = content.find('}', open);
    if (open == std::string::npos || close == std::string::npos)
        throw std::runtime_error("scalers.json : objet invalide pour " + key);
    return content.substr(open, close - open + 1);
}

void Scalers::save(const std::string& path) const
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Impossible d'écrire " + path);
    out << "{\n"
        << "  \"features\": " << json_string_array(feature_names) << ",\n"
        << "  \"targets\": " << json_string_array(target_names) << ",\n"
        << "  \"x\": " << x.to_json() << ",\n"
        << "  \"y\": " << y.to_json() << "\n"
        << "}\n";
    if (!out) throw std::runtime_error("Erreur d'écriture dans " + path);
}

void Scalers::load(const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Impossible de lire " + path);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    feature_names = parse_string_array(content, "features");
    target_names = parse_string_array(content, "targets");
    x.from_json(extract_object(content, "x"));
    y.from_json(extract_object(content, "y"));

    if (x.size() != feature_names.size() || y.size() != target_names.size())
        throw std::runtime_error(path + " : tailles incohérentes entre noms et statistiques");
}

std::string hash_files(const std::vector<std::string>& paths)
{
    uint64_t h = 1469598103934665603ULL;          // FNV offset basis
    for (const std::string& path : paths) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Impossible de lire " + path);
        char c;
        while (in.get(c)) {
            h ^= static_cast<unsigned char>(c);
            h *= 1099511628211ULL;                // FNV prime
        }
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << h;
    return out.str();
}

std::vector<int> baseline_columns(const std::vector<std::string>& features,
                                  const std::vector<std::string>& targets)
{
    std::vector<int> cols(targets.size(), -1);
    for (size_t k = 0; k < targets.size(); ++k) {
        std::string wanted = "x_s3_" + targets[k].substr(2);   // y_PTS -> x_s3_PTS
        for (size_t j = 0; j < features.size(); ++j)
            if (features[j] == wanted) cols[k] = static_cast<int>(j);
    }
    return cols;
}
