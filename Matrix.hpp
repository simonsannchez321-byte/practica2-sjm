#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

// Matriz densa minima (sin librerias externas): transpuesta, producto y
// solucion de sistemas lineales A x = b por Gauss-Jordan con pivoteo parcial.
class Matrix {
public:
    typedef std::vector<std::vector<double> > Data;

    Matrix(std::size_t rows, std::size_t cols, double init = 0.0)
        : rows_(rows), cols_(cols), data_(rows, std::vector<double>(cols, init)) {}

    explicit Matrix(const Data &d)
        : rows_(d.size()), cols_(d.empty() ? 0 : d[0].size()), data_(d) {}

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    double &operator()(std::size_t i, std::size_t j) { return data_[i][j]; }
    double operator()(std::size_t i, std::size_t j) const { return data_[i][j]; }

    Matrix transpose() const {
        Matrix t(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j) t(j, i) = data_[i][j];
        return t;
    }

    Matrix multiply(const Matrix &o) const {
        if (cols_ != o.rows_) throw std::invalid_argument("Matrix::multiply: dimensiones incompatibles");
        Matrix r(rows_, o.cols_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t k = 0; k < cols_; ++k) {
                double a = data_[i][k];
                for (std::size_t j = 0; j < o.cols_; ++j) r(i, j) += a * o(k, j);
            }
        return r;
    }

    std::vector<double> multiply(const std::vector<double> &v) const {
        if (cols_ != v.size()) throw std::invalid_argument("Matrix::multiply(vector): dimensiones incompatibles");
        std::vector<double> r(rows_, 0.0);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j) r[i] += data_[i][j] * v[j];
        return r;
    }

    // Resuelve A x = b (A cuadrada). Lanza excepcion si A es singular.
    static std::vector<double> solve(Matrix A, std::vector<double> b) {
        const std::size_t n = A.rows_;
        if (A.cols_ != n || b.size() != n) throw std::invalid_argument("Matrix::solve: sistema no cuadrado");
        for (std::size_t c = 0; c < n; ++c) {
            std::size_t piv = c;
            for (std::size_t r = c + 1; r < n; ++r)
                if (std::fabs(A(r, c)) > std::fabs(A(piv, c))) piv = r;
            if (std::fabs(A(piv, c)) < 1e-12) throw std::runtime_error("Matrix::solve: matriz singular");
            if (piv != c) {
                A.data_[piv].swap(A.data_[c]);
                std::swap(b[piv], b[c]);
            }
            for (std::size_t r = 0; r < n; ++r) {
                if (r == c) continue;
                double f = A(r, c) / A(c, c);
                if (f == 0.0) continue;
                for (std::size_t k = c; k < n; ++k) A(r, k) -= f * A(c, k);
                b[r] -= f * b[c];
            }
        }
        std::vector<double> x(n);
        for (std::size_t i = 0; i < n; ++i) x[i] = b[i] / A(i, i);
        return x;
    }

private:
    std::size_t rows_, cols_;
    Data data_;
};

#endif
