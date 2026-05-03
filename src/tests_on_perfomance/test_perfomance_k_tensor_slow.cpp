#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>

#include "mkl.h"
#include "generate_matrices.h"
#include "lindblad_utils.h"
#include "mkl_complex16.h"
#include "matrix_decomposition.h"
#include "timer.h"



int main() {
    // Параметры
    int N = 9;
    int M = N * N - 1;

    // Cоздаем линдбладиан
    VSLStreamStatePtr stream;
    int seed = 0;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    MKL_Complex16* lindbladian = GenerateLp(N, stream);
    vslDeleteStream(&stream);


    // Вычисляем коэффициенты l
    std::vector<MKL_Complex16> l_coeff = GetLCoef(lindbladian, N);

    std::vector<MKL_Complex16> l_coeff_conjugate(l_coeff);
    for (auto& elem : l_coeff_conjugate) {
        elem = Conjugate(elem);
    }

    auto f_tensor = GenerateTensorF(N);

    size_t total_elements = M * M * M;
    double* f_tensor_nonsparse = (double*)mkl_malloc(total_elements * sizeof(double), 64);
    memset(f_tensor_nonsparse, 0, total_elements * sizeof(double));

    for (const auto& [ind, value] : f_tensor) {
        auto& [m, n, s] = ind;
        int f_ind = m * M * M + n * M + s;
        f_tensor_nonsparse[f_ind] = value;
    }

    RAMMeter meter;
    Timer timer;

    double* k_tensor = (double*)mkl_malloc(M * sizeof(double), 64);
    memset(k_tensor, 0, M * sizeof(double));

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < M; ++n) {
            // коэффициент матрицы Коссаковски
            MKL_Complex16 a_mn = l_coeff[m] * l_coeff_conjugate[n];
            for (int s = 0; s < M; ++s) {
                int f_index = m * M * M + n * M + s;
                k_tensor[s] += (a_mn * f_tensor_nonsparse[f_index]).imag;
            }
        }
    }

    for (int s = 0; s < M; ++s) {
        k_tensor[s] *= -1. / N;
    }

    timer.stop();
    meter.tick();

    // Освобождение памяти
    mkl_free(k_tensor);
    mkl_free(f_tensor_nonsparse);
    mkl_free(lindbladian);
    return 0;
}