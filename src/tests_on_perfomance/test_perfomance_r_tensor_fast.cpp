#include <iostream>
#include <vector>

#include "mkl.h"
#include "generate_matrices.h"
#include "mkl_complex16.h"
#include "lindblad_utils.h"
#include "matrix_decomposition.h"
#include "timer.h"

int main(int argc, char* argv[]) {
    // Параметры
    int N = 50;
    // на случай если передаем размер N в параметрах командной строки
    if (argc == 2) {
        N = std::atoi(argv[1]);
    }

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

    RAMMeter meter;
    Timer timer;

    double* r_tensor = GenerateMatrixR(l_coeff, l_coeff_conjugate, &f_tensor, &z_tensor, N);

    timer.stop();
    meter.tick();
    // филлерный код, чтобы компилятор не выкинул вышенаписанный код
    std::cout << r_tensor[0];
    
    mkl_free(lindbladian);
    mkl_free(r_tensor);
    return 0;
}