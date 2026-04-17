#include <iostream>
#include <vector>

#include "mkl.h"
#include "generate_matrices.h"
#include "lindblad_utils.h"
#include "mkl_complex16.h"
#include "matrix_decomposition.h"
#include "timer.h"

int main(int argc, char* argv[]) {
    // Параметры
    int N = 50;
    // int M = N * N - 1;
    // на случай если передаем размер N в параметрах командной строки
    if (argc == 2) {
        N = std::atoi(argv[1]);
    }

    VSLStreamStatePtr stream;
    int seed = 0;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    MKL_Complex16* hamiltonian = GenerateTracelessHamiltonian(N, stream);
    vslDeleteStream(&stream);

    // Вычисляем коэффициенты h
    std::vector<double> h_coeff = GetHCoef(hamiltonian, N);
    auto f_tensor = GenerateTensorF<0>(N);

    RAMMeter meter;
    Timer timer;

    std::vector<std::pair<std::tuple<int, int>, double>> q_matrix =
        GenerateCOOMatrixQ<2>(&f_tensor, h_coeff, N);

    timer.stop();
    meter.tick();
    std::cout << q_matrix[0].second << "\n";

    // Освобождение памяти
    mkl_free(hamiltonian);
}