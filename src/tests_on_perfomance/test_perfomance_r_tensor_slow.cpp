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

int main(int argc, char* argv[]) {
    // Параметры
    int N = 7;
    // на случай если передаем размер N в параметрах командной строки
    if (argc == 2) {
        N = std::atoi(argv[1]);
    }

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

    // Общее количество элементов
    size_t total_elements = M * M * M;
    // Выделение памяти для тензора
    double* f_tensor_nonsparse = (double*)mkl_malloc(total_elements * sizeof(double), 64);
    memset(f_tensor_nonsparse, 0, total_elements * sizeof(double));

    MKL_Complex16* z_tensor_nonsparse =
        (MKL_Complex16*)mkl_malloc(total_elements * sizeof(MKL_Complex16), 64);
    memset(z_tensor_nonsparse, 0, total_elements * sizeof(MKL_Complex16));

    for (const auto& [ind, value] : f_tensor) {
        const auto& [m, n, s] = ind;
        int index = m * M * M + n * M + s;
        f_tensor_nonsparse[index] = value;
    }

    for (const auto& [ind, value] : z_tensor) {
        const auto& [m, n, s] = ind;
        int index = m * M * M + n * M + s;
        z_tensor_nonsparse[index] = value;
    }


    RAMMeter meter;
    Timer timer;

    double* r_tensor = (double*)mkl_malloc(M * M * sizeof(double), 64);

    // Циклы для вычисления каждого элемента r_sn
    for (int s = 0; s < M; ++s) {
        for (int n = 0; n < M; ++n) {

            double sum_jkl = 0.;  // Сумма по j, k, l

            for (int j = 0; j < M; ++j) {
                for (int k = 0; k < M; ++k) {
                    for (int l = 0; l < M; ++l) {
                        // Вычисляем z_jln * f_kls
                        int z_ind = j * M * M + l * M + n;
                        int f_ind = k * M * M + l * M + s;
                        MKL_Complex16 term = z_tensor_nonsparse[z_ind] * f_tensor_nonsparse[f_ind];

                        // Вычисляем z_bar_kln * f_jls
                        z_ind = k * M * M + l * M + n;
                        f_ind = j * M * M + l * M + s;
                        term += Conjugate(z_tensor_nonsparse[z_ind]) * f_tensor_nonsparse[f_ind];

                        sum_jkl += (l_coeff[j] * l_coeff_conjugate[k] * term).real;
                    }
                }
            }
            r_tensor[s * M + n] = -0.25 * sum_jkl;
        }
    }

    timer.stop();
    meter.tick();
    // филлерный код, чтобы компилятор не выкинул выщеприведенный бенчмарк
    std::cout << r_tensor[0] << "\n";

    // Освобождение памяти
    mkl_free(r_tensor);
    mkl_free(f_tensor_nonsparse);
    mkl_free(z_tensor_nonsparse);
    mkl_free(lindbladian);
    return 0;
}