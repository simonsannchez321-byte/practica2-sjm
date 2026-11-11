#ifndef STATS_HPP
#define STATS_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

// Funciones estadisticas auxiliares (implementadas a mano).
namespace Stats {

inline double mean(const std::vector<double> &v) {
    if (v.empty()) throw std::invalid_argument("Stats::mean: vector vacio");
    double s = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) s += v[i];
    return s / v.size();
}

// Desviacion estandar poblacional.
inline double stddev(const std::vector<double> &v) {
    double m = mean(v), s = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) s += (v[i] - m) * (v[i] - m);
    return std::sqrt(s / v.size());
}

inline double mse(const std::vector<double> &y, const std::vector<double> &yhat) {
    if (y.size() != yhat.size() || y.empty()) throw std::invalid_argument("Stats::mse: tamanos distintos");
    double s = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) s += (y[i] - yhat[i]) * (y[i] - yhat[i]);
    return s / y.size();
}

inline double mae(const std::vector<double> &y, const std::vector<double> &yhat) {
    if (y.size() != yhat.size() || y.empty()) throw std::invalid_argument("Stats::mae: tamanos distintos");
    double s = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) s += std::fabs(y[i] - yhat[i]);
    return s / y.size();
}

// Coeficiente de determinacion R^2 = 1 - SS_res / SS_tot.
inline double r2(const std::vector<double> &y, const std::vector<double> &yhat) {
    if (y.size() != yhat.size() || y.empty()) throw std::invalid_argument("Stats::r2: tamanos distintos");
    double m = mean(y), ssRes = 0.0, ssTot = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        ssRes += (y[i] - yhat[i]) * (y[i] - yhat[i]);
        ssTot += (y[i] - m) * (y[i] - m);
    }
    return ssTot == 0.0 ? 0.0 : 1.0 - ssRes / ssTot;
}

} // namespace Stats

#endif
