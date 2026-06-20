#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>

#include "mkl.h"
#include "generate_matrices.h"
#include "matrix_decomposition.h"
#include "mkl_complex16.h"
#include "lindblad_utils.h"


double Check(double* first, double* second, int N) {
    double eps = 0.;
    for (int i = 0; i < N * N; ++i) {
        eps = std::max(eps, abs(first[i] - second[i]));
        if (abs(first[i] - second[i]) > 1e-4) {
            std::cout << i << "\n";
        }
    }
    return eps;
}

int main() {
    // Параметры
    int N = 70;
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
    auto d_tensor = GenerateTensorD(N);
    auto z_tensor = GenerateTensorZ(f_tensor, d_tensor);


    double* r_matrix_fast1 = GenerateMatrixR(l_coeff, l_coeff_conjugate, &f_tensor, &z_tensor, N);

    // Функтор, вычисляющий матрицу Коссаковски
    auto kossakovski_func = [&l_coeff, &l_coeff_conjugate](size_t i, size_t j) {
        return l_coeff[i] * l_coeff_conjugate[j];
    };

    double* r_matrix_fast2 = GenerateMatrixR(kossakovski_func, &f_tensor, &z_tensor, N);

    std::cout << Check(r_matrix_fast1, r_matrix_fast2, M) << "\n";

    // Освобождение памяти
    mkl_free(r_matrix_fast1);
    mkl_free(r_matrix_fast2);

    mkl_free(lindbladian);
    return 0;
}