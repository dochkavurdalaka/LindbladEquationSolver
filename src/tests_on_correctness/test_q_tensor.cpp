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


int main() {
    // Параметры
    int N = 11;
    int M = N * N - 1;

    // Cоздаем гамильтониан
    VSLStreamStatePtr stream;
    int seed = 0;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    MKL_Complex16* hamiltonian = GenerateTracelessHamiltonian(N, stream);
    vslDeleteStream(&stream);

    std::vector<double> h_coeff = GetHCoef(hamiltonian, N);

    auto f_tensor = GenerateTensorF(N, false);

    std::vector<std::pair<std::tuple<int, int>, double>> q_tensor =
        GenerateCOOMatrixQ(&f_tensor, h_coeff);

    // Общее количество элементов
    size_t total_elements = M * M * M;
    // Выделение памяти для тензора
    double* f_tensor_nonsparse = (double*)mkl_malloc(total_elements * sizeof(double), 64);
    memset(f_tensor_nonsparse, 0, total_elements * sizeof(double));
    for (const auto& [ind, value] : f_tensor) {
        const auto& [m, n, s] = ind;
        int index = m * M * M + n * M + s;
        f_tensor_nonsparse[index] = value;
    }

    double* q_tensor_nonsparse = (double*)mkl_malloc(M * M * sizeof(double), 64);
    memset(q_tensor_nonsparse, 0, M * M * sizeof(double));

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < M; ++n) {
            for (int s = 0; s < M; ++s) {
                int q_index = s * M + n;
                int f_index = m * M * M + n * M + s;
                q_tensor_nonsparse[q_index] += h_coeff[m] * f_tensor_nonsparse[f_index];
            }
        }
    }

    std::vector<std::pair<std::tuple<int, int>, double>> new_q_tensor;
    for (int n = 0; n < M; ++n) {
        for (int s = 0; s < M; ++s) {
            int q_index = s * M + n;
            if (q_tensor_nonsparse[q_index] != 0) {
                new_q_tensor.emplace_back(std::tuple(s, n), q_tensor_nonsparse[q_index]);
            }
        }
    }
    std::sort(new_q_tensor.begin(), new_q_tensor.end());

    if (q_tensor.size() != new_q_tensor.size()) {
        std::cout << "false\n";
    }

    double eps = 0;
    for (size_t i = 0; i < q_tensor.size(); ++i) {
        eps = std::max(abs(q_tensor[i].second - new_q_tensor[i].second), eps);
        if ((q_tensor[i].first != new_q_tensor[i].first) or
            abs(q_tensor[i].second - new_q_tensor[i].second) > 1e-12) {
            std::cout << "false\n";
        }
    }
    std::cout << eps << "\n";

    // Освобождение памяти
    mkl_free(f_tensor_nonsparse);
    mkl_free(q_tensor_nonsparse);
    mkl_free(hamiltonian);
}