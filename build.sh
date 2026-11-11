#!/bin/sh
# Compila ambos programas. -static evita conflictos de DLL de libstdc++ en MSYS2/MinGW.
g++ -std=c++17 -Wall -Wextra -O2 -static src/main.cpp -o main.exe &&
g++ -std=c++17 -Wall -Wextra -O2 -static src/memory_analysis.cpp -o memory_analysis.exe
