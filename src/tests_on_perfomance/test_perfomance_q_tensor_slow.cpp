#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <fstream>
#include <unistd.h>

#include "mkl.h"
#include "lindblad_utils.h"
#include "generate_matrices.h"
#include "matrix_decomposition.h"
#include "mkl_complex16.h"
#include "timer.h"

// Вспомогательная функция для печати вектора
void print_vector(const char* name, double t, const double* v, int M) {
    std::cout << name << "(t=" << t << "): [ ";
    for (int i = 0; i < M; ++i) {
        std::cout << v[i] << ", ";
    }
    std::cout << "]" << std::endl;
}



int main(int argc, char* argv[]) {
    // Параметры
    int N = 20;
    size_t M = N * N - 1;
    // на случай если передаем размер N в параметрах командной строки
    if (argc == 2) {
        N = std::atoi(argv[1]);
    }

    // Cоздаем гамильтониан
    VSLStreamStatePtr stream;
    int seed = 0;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    MKL_Complex16* hamiltonian = GenerateTracelessHamiltonian(N, stream);
    vslDeleteStream(&stream);

    // Вычисляем коэффициенты h
    std::vector<double> h_coeff = GetHCoef(hamiltonian, N);
    auto f_tensor = GenerateTensorF<0>(N);

    // Общее количество элементов
    size_t total_elements = M * M * M;
    // Выделение памяти для тензора
    double* f_tensor_sparse = (double*)mkl_malloc(total_elements * sizeof(double), 64);
    if (f_tensor_sparse == NULL) {
        std::cout << "not enough memory for allocation tensor f";
        return 0;
    }
    memset(f_tensor_sparse, 0, M * M * M * sizeof(double));

    for (const auto& [ind, value] : f_tensor) {
        const auto& [m, n, s] = ind;
        size_t index = m * M * M + n * M + s;
        f_tensor_sparse[index] = value;
    }

    RAMMeter meter;
    Timer timer;

    double* q_tensor_sparse = (double*)mkl_malloc(M * M * sizeof(double), 64);
    memset(q_tensor_sparse, 0, M * M * sizeof(double));

    if (q_tensor_sparse == NULL) {
        std::cout << "not enough memory for allocation tensor q";
        return 0;
    }

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < M; ++n) {
            for (size_t s = 0; s < M; ++s) {
                size_t q_index = s * M + n;
                size_t f_index = m * M * M + n * M + s;
                q_tensor_sparse[q_index] += h_coeff[m] * f_tensor_sparse[f_index];
            }
        }
    }

    timer.stop();
    meter.tick();


    std::cout << q_tensor_sparse[0]<< "\n";

    // Освобождение памяти
    mkl_free(hamiltonian);
    mkl_free(f_tensor_sparse);
    mkl_free(q_tensor_sparse);
}