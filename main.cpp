// Practica 2 - Regresion lineal simple y multiple en C++ (POO, sin librerias externas).
// Uso: main.exe [ruta_ice_cream.csv] [ruta_student_scores.csv]
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "CsvReader.hpp"
#include "LinearRegression.hpp"

typedef std::vector<std::vector<double> > Mat;

// Particion determinista: cada 5ta fila va a test (80% train / 20% test).
static void split(const Dataset &d, Mat &Xtr, std::vector<double> &ytr, Mat &Xte, std::vector<double> &yte) {
    for (std::size_t i = 0; i < d.X.size(); ++i) {
        if (i % 5 == 4) { Xte.push_back(d.X[i]); yte.push_back(d.y[i]); }
        else            { Xtr.push_back(d.X[i]); ytr.push_back(d.y[i]); }
    }
}

static void run(const std::string &title, const std::string &path, bool addSquare = false) {
    std::cout << "==================== " << title << " ====================\n";
    Dataset d = CsvReader::load(path);
    if (addSquare) { // ingenieria de caracteristicas: agrega x^2 (la relacion es parabolica)
        for (std::size_t i = 0; i < d.X.size(); ++i) d.X[i].push_back(d.X[i][0] * d.X[i][0]);
        d.featureNames.push_back(d.featureNames[0] + "^2");
    }
    Mat Xtr, Xte;
    std::vector<double> ytr, yte;
    split(d, Xtr, ytr, Xte, yte);
    std::cout << "Filas: " << d.X.size() << " (train=" << Xtr.size() << ", test=" << Xte.size() << ")\n";

    LinearRegression mlr(true);
    mlr.fit(Xtr, ytr);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "weights (escalados): ";
    for (std::size_t j = 0; j < mlr.getWeights().size(); ++j) std::cout << mlr.getWeights()[j] << " ";
    std::cout << "\nbias (escalado): " << mlr.getBias() << "\n";

    std::vector<double> w = mlr.getWeightsOriginalScale();
    std::cout << "Ecuacion (unidades originales): " << d.targetName << " = " << mlr.getBiasOriginalScale();
    for (std::size_t j = 0; j < w.size(); ++j)
        std::cout << (w[j] < 0 ? " - " : " + ") << std::fabs(w[j]) << "*[" << d.featureNames[j] << "]";
    std::cout << "\n\n";

    std::vector<double> yhat = mlr.predict(Xte);
    std::cout << "Predicciones sobre test:\n";
    for (std::size_t i = 0; i < yhat.size(); ++i)
        std::cout << "  y_hat: " << std::setw(10) << yhat[i] << "   (real: " << std::setw(10) << yte[i] << ")\n";

    std::cout << "\nscore R2 (train): " << mlr.score(Xtr, ytr) << "\n";
    std::cout << "score R2 (test) : " << mlr.score(Xte, yte) << "\n";
    std::cout << "MSE (test)      : " << mlr.mse(Xte, yte) << "\n";
    std::cout << "MAE (test)      : " << mlr.mae(Xte, yte) << "\n\n";
}

int main(int argc, char **argv) {
    std::string ice = argc > 1 ? argv[1] : "Ice_cream_selling_data.csv";
    std::string stu = argc > 2 ? argv[2] : "student_exam_scores.csv";
    try {
        run("REGRESION SIMPLE (helados)", ice);
        run("REGRESION MULTIPLE (examenes)", stu);
        run("EXTRA: helados con termino cuadratico x^2", ice, true);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
