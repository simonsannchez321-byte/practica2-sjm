// Practica 2, seccion 3.2 - Analisis de memoria de LinearRegression.
#include <cstdio>
#include <iostream>
#include <vector>

#include "LinearRegression.hpp"

LinearRegression globalModel; // memoria global (segmento de datos/BSS)

static void show(const char *name, const LinearRegression *p) {
    std::printf("%-28s direccion: %p   sizeof: %zu bytes\n", name, (const void *)p, sizeof(*p));
}

int main() {
    LinearRegression localModel;                      // pila (stack)
    LinearRegression *heapModel = new LinearRegression(); // monticulo (heap)

    std::cout << "--- Objetos individuales ---\n";
    show("global (data/BSS)", &globalModel);
    show("local (stack)", &localModel);
    show("dinamico (heap)", heapModel);
    std::printf("Direccion de la variable puntero heapModel (stack): %p\n", (void *)&heapModel);

    std::cout << "\n--- Desglose de sizeof(LinearRegression) ---\n";
    std::printf("sizeof(std::vector<double>) = %zu  (x3: weights, means, stds)\n", sizeof(std::vector<double>));
    std::printf("sizeof(double) = %zu (bias)\n", sizeof(double));
    std::printf("sizeof(bool)   = %zu (x2: useScaling, fitted) + padding de alineacion\n", sizeof(bool));
    std::printf("sizeof(LinearRegression) = %zu\n", sizeof(LinearRegression));

    std::cout << "\n--- Efecto de entrenar: sizeof no cambia, el heap interno si ---\n";
    std::vector<std::vector<double> > X;
    std::vector<double> y;
    for (int i = 0; i < 20; ++i) { { double r[3] = {1.0 * i, 1.0 * (i % 4), 0.5 * i * i}; X.push_back(std::vector<double>(r, r + 3)); } y.push_back(2.0 * i + 1); }
    heapModel->fit(X, y);
    std::printf("Tras fit(): sizeof = %zu ; weights.data() apunta a %p (heap) con %zu elementos\n",
                sizeof(*heapModel), (const void *)heapModel->getWeights().data(), heapModel->getWeights().size());

    std::cout << "\n--- Vector de LinearRegression en memoria dinamica ---\n";
    const std::size_t N = 5;
    std::vector<LinearRegression> *vec = new std::vector<LinearRegression>(N);
    std::printf("Direccion del objeto std::vector (heap): %p ; sizeof(vector) = %zu\n", (void *)vec, sizeof(*vec));
    std::printf("Direccion del buffer de elementos      : %p\n", (const void *)vec->data());
    std::printf("size()=%zu, capacity()=%zu\n", vec->size(), vec->capacity());
    std::printf("Bytes reservados en el buffer = N * sizeof(LinearRegression) = %zu * %zu = %zu\n", N,
                sizeof(LinearRegression), N * sizeof(LinearRegression));
    for (std::size_t i = 0; i < N; ++i) std::printf("  elemento[%zu] en %p\n", i, (const void *)&(*vec)[i]);

    // Arreglo dinamico estilo C++ clasico
    LinearRegression *arr = new LinearRegression[N];
    std::printf("\nnew LinearRegression[%zu] en %p -> %zu bytes (sin contar heaps internos de cada vector)\n", N,
                (void *)arr, N * sizeof(LinearRegression));

    delete[] arr;
    delete vec;
    delete heapModel;
    return 0;
}
