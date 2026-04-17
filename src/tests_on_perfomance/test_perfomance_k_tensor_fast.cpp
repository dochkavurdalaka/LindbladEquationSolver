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
#include "timer.h"

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
    int N = 9;

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

    auto f_tensor = GenerateTensorF<0>(N);

    // Функтор, вычисляющий матрицу Коссаковски
    auto kossakovski_func = [&l_coeff, &l_coeff_conjugate](size_t i, size_t j) {
        return l_coeff[i] * l_coeff_conjugate[j];
    };

    RAMMeter meter;
    Timer timer;
    double* k_tensor = GenerateVectorKWithFunctor(kossakovski_func, f_tensor, N);
    timer.stop();
    meter.tick();

    // Освобождение памяти
    mkl_free(k_tensor);
    mkl_free(lindbladian);
    return 0;
}