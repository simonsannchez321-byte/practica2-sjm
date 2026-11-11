#ifndef CSVREADER_HPP
#define CSVREADER_HPP

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Dataset numerico cargado desde CSV: la ULTIMA columna es la variable objetivo y.
struct Dataset {
    std::vector<std::string> featureNames;
    std::string targetName;
    std::vector<std::vector<double> > X;
    std::vector<double> y;
};

class CsvReader {
public:
    static Dataset load(const std::string &path) {
        std::ifstream in(path.c_str());
        if (!in) throw std::runtime_error("No se pudo abrir: " + path);

        Dataset d;
        std::string line;
        std::getline(in, line);
        // Elimina BOM UTF-8 si existe.
        if (line.size() >= 3 && (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB &&
            (unsigned char)line[2] == 0xBF)
            line = line.substr(3);
        std::vector<std::string> header = split(line);
        if (header.size() < 2) throw std::runtime_error("CSV con menos de 2 columnas");
        d.targetName = header.back();
        header.pop_back();
        d.featureNames = header;

        while (std::getline(in, line)) {
            if (line.empty() || line == "\r") continue;
            std::vector<std::string> f = split(line);
            if (f.size() != header.size() + 1) throw std::runtime_error("Fila con numero de columnas invalido");
            std::vector<double> row;
            for (std::size_t j = 0; j + 1 < f.size(); ++j) row.push_back(std::stod(f[j]));
            d.X.push_back(row);
            d.y.push_back(std::stod(f.back()));
        }
        return d;
    }

private:
    static std::vector<std::string> split(const std::string &line) {
        std::vector<std::string> out;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty() && cell[cell.size() - 1] == '\r') cell.erase(cell.size() - 1);
            out.push_back(cell);
        }
        return out;
    }
};

#endif
