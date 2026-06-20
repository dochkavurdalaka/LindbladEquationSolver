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

    auto f_tensor = GenerateTensorF(N);

    // Функтор, вычисляющий матрицу Коссаковски
    auto kossakovski_func = [&l_coeff, &l_coeff_conjugate](size_t i, size_t j) {
        return l_coeff[i] * l_coeff_conjugate[j];
    };

    RAMMeter meter;
    Timer timer;
    double* k_vector = GenerateVectorKWithFunctor(kossakovski_func, f_tensor, N);
    timer.stop();
    meter.tick();

    // Освобождение памяти
    mkl_free(k_vector);
    mkl_free(lindbladian);
    return 0;
}