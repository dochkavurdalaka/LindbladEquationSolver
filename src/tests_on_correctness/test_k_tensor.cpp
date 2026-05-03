#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>

#include "mkl.h"
#include "generate_matrices.h"
#include "mkl_complex16.h"
#include "lindblad_utils.h"
#include "matrix_decomposition.h"


double Check(double* first, double* second, int N) {
    double eps = 0.;
    for (int i = 0; i < N; ++i) {
        if (abs(first[i] - second[i]) > 1e-12) {
            std::cout << i << "\n";
        }
        eps = std::max(abs(first[i] - second[i]), eps);
    }
    return eps;
}


int main() {
    // Параметры
    int N = 12;
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

    auto f_tensor = GenerateTensorF(N, false);

    double* k_tensor = (double*)mkl_malloc(M * sizeof(double), 64);
    memset(k_tensor, 0, M * sizeof(double));

    size_t total_elements = M * M * M;
    double* f_tensor_nonsparse = (double*)mkl_malloc(total_elements * sizeof(double), 64);
    memset(f_tensor_nonsparse, 0, total_elements * sizeof(double));

    for (const auto& [ind, value] : f_tensor) {
        auto& [m, n, s] = ind;
        int f_ind = m * M * M + n * M + s;
        f_tensor_nonsparse[f_ind] = value;
    }

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < M; ++n) {
            // Коэффициент матрицы Коссаковски с индексами m, n
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

    // Функтор, вычисляющий матрицу Коссаковски
    auto kossakovski_func = [&l_coeff, &l_coeff_conjugate](size_t i, size_t j) {
        return l_coeff[i] * l_coeff_conjugate[j];
    };

    double* k_tensor_tst = GenerateVectorKWithFunctor(kossakovski_func, f_tensor, N);

    std::cout << Check(k_tensor, k_tensor_tst, M) << std::endl;

    // Освобождение памяти
    mkl_free(k_tensor_tst);
    mkl_free(k_tensor);
    mkl_free(f_tensor_nonsparse);
    mkl_free(lindbladian);
    return 0;
}