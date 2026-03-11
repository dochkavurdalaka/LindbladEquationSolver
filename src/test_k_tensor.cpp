#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <chrono>

#include "mkl.h"
#include "generate_matrices.h"
#include "mkl_complex16.h"
#include "lindblad_utils.h"

using namespace std::chrono;

MKL_Complex16 Trace(const MKL_Complex16* matrix, int n) {
    MKL_Complex16 tr = {0.0, 0.0};
    for (int i = 0; i < n; ++i) {
        int diag_idx = i * n + i;
        tr.real += matrix[diag_idx].real;
        tr.imag += matrix[diag_idx].imag;
    }
    return tr;
}

// здесь и далее попробуем решить систему для константного H
// тогда матрица Q тоже будет константной

// Вспомогательная функция для печати вектора
void print_vector(const char* name, double t, const double* v, int M) {
    std::cout << name << "(t=" << t << "): [ ";
    for (int i = 0; i < M; ++i) {
        std::cout << v[i] << ", ";
    }
    std::cout << "]" << std::endl;
}

double Check(double* first, double* second, int N) {
    double eps = 0.;
    for (int i = 0; i < N; ++i) {
        if (abs(first[i] - second[i]) > 1e-4) {
            std::cout << i << "\n";
        }
        eps = std::max(abs(first[i] - second[i]), eps);
    }
    return eps;
}

std::vector<double> GetHCoef(MKL_Complex16* hamiltonian, int N) {
    std::vector<double> h_coeff;

    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            int index = j * N + k;
            h_coeff.push_back(sqrt(2.) * hamiltonian[index].real);
        }
    }

    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            int index = k * N + j;
            h_coeff.push_back(sqrt(2.) * hamiltonian[index].imag);
        }
    }

    for (int l = 0; l < N - 1; ++l) {
        double coeff = 0.;

        for (int k = 0; k < l + 1; ++k) {
            int index = k * N + k;
            coeff += hamiltonian[index].real / sqrt((l + 1) * (l + 2));
        }

        int index = (l + 1) * N + (l + 1);
        coeff += -sqrt(l + 1) * hamiltonian[index].real / sqrt(l + 2);

        h_coeff.push_back(coeff);
    }

    return h_coeff;
}

std::vector<MKL_Complex16> GetLCoef(MKL_Complex16* lindbladian, int N) {
    std::vector<MKL_Complex16> l_coeff;
    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            MKL_Complex16 coeff = (lindbladian[j * N + k] + lindbladian[k * N + j]) / sqrt(2);
            l_coeff.push_back(coeff);
        }
    }

    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {

            int ind_one = j * N + k;
            int ind_two = k * N + j;
            MKL_Complex16 coeff;
            coeff.real = lindbladian[ind_two].imag - lindbladian[ind_one].imag;
            coeff.imag = lindbladian[ind_one].real - lindbladian[ind_two].real;
            l_coeff.push_back(coeff / sqrt(2));
        }
    }

    for (int l = 0; l < N - 1; ++l) {
        MKL_Complex16 coeff = {0., 0.};

        for (int k = 0; k < l + 1; ++k) {
            int index = k * N + k;
            coeff += lindbladian[index] / sqrt((l + 1) * (l + 2));
        }

        int index = (l + 1) * N + (l + 1);
        coeff += -sqrt(l + 1) * lindbladian[index] / sqrt(l + 2);

        l_coeff.push_back(coeff);
    }

    return l_coeff;
}


int main() {
    // Параметры
    int N = 9;
    int M = N * N - 1;

    // Cоздаем гамильтониан
    MKL_Complex16* hamiltonian;
    GenerateTracelessHamiltonian(N, 2, hamiltonian);

    // Cоздаем линдбладиан
    MKL_Complex16* lindbladian;
    GenerateLp(N, 2, lindbladian);

    // Cоздаем матрицу плотности
    MKL_Complex16* rho;
    GenerateDensity(N, 2, rho);

    // Вычисляем коэффициенты h
    std::vector<double> h_coeff = GetHCoef(hamiltonian, N);

    // Вычисляем коэффициенты l
    std::vector<MKL_Complex16> l_coeff = GetLCoef(lindbladian, N);

    std::vector<MKL_Complex16> l_coeff_conjugate(l_coeff);
    for (auto& elem : l_coeff_conjugate) {
        elem = Conjugate(elem);
    }

    // Вычисляем матрицу Коссаковски
    std::vector<MKL_Complex16> a(M * M);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            a[i * M + j] = l_coeff[i] * l_coeff_conjugate[j];
        }
    }

    auto f_tensor = GenerateTensorF(N);

    double* k_tensor = (double*)mkl_malloc(M * sizeof(double), 64);
    memset(k_tensor, 0, M * sizeof(double));

    size_t total_elements = M * M * M;
    double* f_tensor_nonsparse = (double*)mkl_malloc(total_elements * sizeof(double), 64);
    memset(f_tensor_nonsparse, 0, total_elements * sizeof(double));


    for (const auto& el : f_tensor) {
        int f_ind =
            std::get<0>(el.first) * M * M + std::get<1>(el.first) * M + std::get<2>(el.first);
        f_tensor_nonsparse[f_ind] = el.second;
    }

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < M; ++n) {
            for (int s = 0; s < M; ++s) {
                int f_index = m * M * M + n * M + s;
                k_tensor[s] += (a[m * M + n] * f_tensor_nonsparse[f_index]).imag;
            }
        }
    }

    for (int s = 0; s < M; ++s) {
        k_tensor[s] *= -1. / N;
    }

    double* k_tensor_tst = (double*)mkl_malloc(M * sizeof(double), 64);
    memset(k_tensor_tst, 0, M * sizeof(double));
    for (const auto& el : f_tensor) {
        k_tensor_tst[std::get<2>(el.first)] += -1. * (a[std::get<0>(el.first) * M + std::get<1>(el.first)] * el.second).imag / N;
    }

    std::cout << Check(k_tensor, k_tensor_tst, M) << std::endl;


    // print_double_matrix_rowmajor(r_tensor, M, "AAA");

    return 0;
}