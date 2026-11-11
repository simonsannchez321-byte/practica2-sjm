#ifndef LINEARREGRESSION_HPP
#define LINEARREGRESSION_HPP

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "Matrix.hpp"
#include "Stats.hpp"

// Regresion lineal multiple:  y_hat = bias + sum_j weights[j] * x_j
// (con una sola columna en X es regresion lineal simple).
//
// Entrenamiento por ecuaciones normales: (A^T A) beta = A^T y, donde A = [1 | X].
// Si useScaling = true, las columnas de X se estandarizan (z-score) antes de
// entrenar, y los mismos parametros (media/desviacion) se reutilizan al predecir.
class LinearRegression {
public:
    explicit LinearRegression(bool useScaling = true)
        : bias(0.0), useScaling(useScaling), fitted(false) {}

    // fit(): entrena el modelo con X (n x p) y y (n).
    void fit(const std::vector<std::vector<double> > &X, const std::vector<double> &y) {
        if (X.empty() || X.size() != y.size()) throw std::invalid_argument("fit: X e y incompatibles");
        const std::size_t n = X.size(), p = X[0].size();

        std::vector<std::vector<double> > Z = X;
        if (useScaling) Z = data_scaling(X, true);

        Matrix A(n, p + 1);
        for (std::size_t i = 0; i < n; ++i) {
            if (X[i].size() != p) throw std::invalid_argument("fit: filas de distinto largo");
            A(i, 0) = 1.0;
            for (std::size_t j = 0; j < p; ++j) A(i, j + 1) = Z[i][j];
        }
        Matrix At = A.transpose();
        std::vector<double> beta = Matrix::solve(At.multiply(A), At.multiply(y));

        bias = beta[0];
        weights.assign(beta.begin() + 1, beta.end());
        fitted = true;
    }

    // predict(): devuelve y_hat para cada fila de X.
    std::vector<double> predict(const std::vector<std::vector<double> > &X) const {
        if (!fitted) throw std::logic_error("predict: el modelo no ha sido entrenado");
        std::vector<std::vector<double> > Z = useScaling ? data_scaling(X) : X;
        std::vector<double> yhat(Z.size(), bias);
        for (std::size_t i = 0; i < Z.size(); ++i) {
            if (Z[i].size() != weights.size()) throw std::invalid_argument("predict: numero de columnas invalido");
            for (std::size_t j = 0; j < weights.size(); ++j) yhat[i] += weights[j] * Z[i][j];
        }
        return yhat;
    }

    // score(): coeficiente de determinacion R^2 sobre (X, y). MSE y MAE por separado.
    double score(const std::vector<std::vector<double> > &X, const std::vector<double> &y) const {
        return Stats::r2(y, predict(X));
    }
    double mse(const std::vector<std::vector<double> > &X, const std::vector<double> &y) const {
        return Stats::mse(y, predict(X));
    }
    double mae(const std::vector<std::vector<double> > &X, const std::vector<double> &y) const {
        return Stats::mae(y, predict(X));
    }

    // data_scaling(): estandarizacion z = (x - media) / desviacion.
    // Con fitParams = true calcula y guarda media/desviacion de X; si no, usa las guardadas.
    std::vector<std::vector<double> > data_scaling(const std::vector<std::vector<double> > &X,
                                                   bool fitParams = false) {
        if (fitParams) computeScaler(X);
        return static_cast<const LinearRegression *>(this)->data_scaling(X);
    }
    std::vector<std::vector<double> > data_scaling(const std::vector<std::vector<double> > &X) const {
        if (means.empty()) throw std::logic_error("data_scaling: escalador no ajustado");
        std::vector<std::vector<double> > Z = X;
        for (std::size_t i = 0; i < Z.size(); ++i) {
            if (Z[i].size() != means.size()) throw std::invalid_argument("data_scaling: columnas invalidas");
            for (std::size_t j = 0; j < means.size(); ++j) Z[i][j] = (Z[i][j] - means[j]) / stds[j];
        }
        return Z;
    }

    // Getters. Si hay escalado, los pesos estan en el espacio estandarizado.
    const std::vector<double> &getWeights() const { return weights; }
    double getBias() const { return bias; }

    // Pesos/sesgo expresados en las unidades originales de X (interpretables).
    std::vector<double> getWeightsOriginalScale() const {
        if (!useScaling) return weights;
        std::vector<double> w(weights.size());
        for (std::size_t j = 0; j < w.size(); ++j) w[j] = weights[j] / stds[j];
        return w;
    }
    double getBiasOriginalScale() const {
        if (!useScaling) return bias;
        double b = bias;
        for (std::size_t j = 0; j < weights.size(); ++j) b -= weights[j] * means[j] / stds[j];
        return b;
    }

private:
    void computeScaler(const std::vector<std::vector<double> > &X) {
        const std::size_t p = X[0].size();
        means.assign(p, 0.0);
        stds.assign(p, 1.0);
        for (std::size_t j = 0; j < p; ++j) {
            std::vector<double> col(X.size());
            for (std::size_t i = 0; i < X.size(); ++i) col[i] = X[i][j];
            means[j] = Stats::mean(col);
            double s = Stats::stddev(col);
            stds[j] = (s == 0.0) ? 1.0 : s; // columna constante: evita division por 0
        }
    }

    std::vector<double> weights; // pesos del modelo
    double bias;                 // termino independiente
    std::vector<double> means, stds; // parametros del escalador
    bool useScaling;
    bool fitted;
};

#endif
