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
    for (int i = 0; i < N * N; ++i) {
        eps = std::max(eps, abs(first[i] - second[i]));
        if (abs(first[i] - second[i]) > 1e-4) {
            std::cout << i << "\n";
        }
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


size_t CountZeros(const double* v, int M) {
    size_t count = 0;
    for (int i = 0; i < M; ++i) {
        if (abs(v[i]) < 1e-4) {
            count += 1;
        }
    }
    return count;
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
    auto d_tensor = GenerateTensorD(N);
    auto z_tensor = GenerateTensorZ(f_tensor, d_tensor);

    double* r_tensor = (double*)mkl_malloc(M * M * sizeof(double), 64);

    // Общее количество элементов
    size_t total_elements = M * M * M;
    // Выделение памяти для тензора
    double* f_tensor_nonsparse = (double*)mkl_malloc(total_elements * sizeof(double), 64);
    memset(f_tensor_nonsparse, 0, total_elements * sizeof(double));

    MKL_Complex16* z_tensor_nonsparse =
        (MKL_Complex16*)mkl_malloc(total_elements * sizeof(MKL_Complex16), 64);
    memset(z_tensor_nonsparse, 0, total_elements * sizeof(MKL_Complex16));

    for (const auto& el : f_tensor) {
        int f_ind =
            std::get<0>(el.first) * M * M + std::get<1>(el.first) * M + std::get<2>(el.first);
        f_tensor_nonsparse[f_ind] = el.second;
    }

    for (const auto& el : z_tensor) {
        int z_ind =
            std::get<0>(el.first) * M * M + std::get<1>(el.first) * M + std::get<2>(el.first);
        z_tensor_nonsparse[z_ind] = el.second;
    }

    // сортировка f_tensor по второму и третьему индексу

    auto cmp = []<typename T>(const std::pair<std::tuple<int, int, int>, T>& left,
                              const std::pair<std::tuple<int, int, int>, T>& right) {
        if (std::get<1>(left.first) == std::get<1>(right.first)) {
            return std::get<2>(left.first) < std::get<2>(right.first);
        }
        return std::get<1>(left.first) < std::get<1>(right.first);
    };
    std::sort(f_tensor.begin(), f_tensor.end(), cmp);

    auto f_tensor_cmplx = DoubleToComplexTensor(f_tensor);

    std::pair<std::tuple<int, int>, std::array<MKL_Complex16, 2>> elem = {
        std::tuple(std::get<1>(f_tensor_cmplx[0].first), std::get<2>(f_tensor_cmplx[0].first)),
        {l_coeff_conjugate[std::get<0>(f_tensor_cmplx[0].first)] * f_tensor_cmplx[0].second,
         l_coeff[std::get<0>(f_tensor_cmplx[0].first)] * f_tensor_cmplx[0].second}};

    std::vector<std::pair<std::tuple<int, int>, std::array<MKL_Complex16, 2>>> f_tensor_ss;

    for (size_t i = 1; i < f_tensor_cmplx.size(); ++i) {
        if (std::get<1>(f_tensor_cmplx[i - 1].first) == std::get<1>(f_tensor_cmplx[i].first) and
            std::get<2>(f_tensor_cmplx[i - 1].first) == std::get<2>(f_tensor_cmplx[i].first)) {
            elem.second[0] +=
                l_coeff_conjugate[std::get<0>(f_tensor_cmplx[i].first)] * f_tensor_cmplx[i].second;

            elem.second[1] +=
                l_coeff[std::get<0>(f_tensor_cmplx[i].first)] * f_tensor_cmplx[i].second;
        } else {
            f_tensor_ss.push_back(elem);

            elem = {
                std::tuple(std::get<1>(f_tensor_cmplx[i].first),
                           std::get<2>(f_tensor_cmplx[i].first)),
                {l_coeff_conjugate[std::get<0>(f_tensor_cmplx[i].first)] * f_tensor_cmplx[i].second,
                 l_coeff[std::get<0>(f_tensor_cmplx[i].first)] * f_tensor_cmplx[i].second}};
        }
    }
    f_tensor_ss.push_back(elem);

    std::sort(z_tensor.begin(), z_tensor.end(), cmp);

    elem = {std::tuple(std::get<1>(z_tensor[0].first), std::get<2>(z_tensor[0].first)),
            {l_coeff[std::get<0>(z_tensor[0].first)] * z_tensor[0].second,
             l_coeff_conjugate[std::get<0>(z_tensor[0].first)] * Conjugate(z_tensor[0].second)}};

    std::vector<std::pair<std::tuple<int, int>, std::array<MKL_Complex16, 2>>> z_tensor_ss;
    
    for (size_t i = 1; i < z_tensor.size(); ++i) {
        if (std::get<1>(z_tensor[i - 1].first) == std::get<1>(z_tensor[i].first) and
            std::get<2>(z_tensor[i - 1].first) == std::get<2>(z_tensor[i].first)) {
            elem.second[0] += l_coeff[std::get<0>(z_tensor[i].first)] * z_tensor[i].second;
            elem.second[1] +=
                l_coeff_conjugate[std::get<0>(z_tensor[i].first)] * Conjugate(z_tensor[i].second);
        } else {
            z_tensor_ss.push_back(elem);

            elem = {std::tuple(std::get<1>(z_tensor[i].first), std::get<2>(z_tensor[i].first)),
                    {l_coeff[std::get<0>(z_tensor[i].first)] * z_tensor[i].second,
                     l_coeff_conjugate[std::get<0>(z_tensor[i].first)] *
                         Conjugate(z_tensor[i].second)}};
        }
    }
    z_tensor_ss.push_back(elem);

    double* r_tensor_tst = (double*)mkl_malloc(M * M * sizeof(double), 64);
    memset(r_tensor_tst, 0, M * M * sizeof(double));

    // Добавляем страж-элемент для упрощения проверки границ в циклах while
    // Чтобы ниже в циулах не проверять end_z < z_tensor_ss.size() и end_f < f_tensor_ss.size()
    z_tensor_ss.emplace_back(std::tuple(M, 0),
                             std::array{MKL_Complex16{0, 0}, MKL_Complex16{0, 0}});
    f_tensor_ss.emplace_back(std::tuple(M, 0),
                             std::array{MKL_Complex16{0, 0}, MKL_Complex16{0, 0}});

    size_t start_z = 0;
    size_t start_f = 0;
    for (int l = 0; l < M; ++l) {
        size_t end_z = start_z;
        size_t end_f = start_f;

        while (std::get<0>(z_tensor_ss[end_z].first) == l) {
            ++end_z;
        }

        while (std::get<0>(f_tensor_ss[end_f].first) == l) {
            ++end_f;
        }

        for (size_t i = start_z; i < end_z; ++i) {
            for (size_t j = start_f; j < end_f; ++j) {
                int s = std::get<1>(f_tensor_ss[j].first);
                int n = std::get<1>(z_tensor_ss[i].first);

                r_tensor_tst[s * M + n] +=
                    -0.25 * (f_tensor_ss[j].second[0] * z_tensor_ss[i].second[0] +
                             f_tensor_ss[j].second[1] * z_tensor_ss[i].second[1])
                                .real;
            }
        }

        start_z = end_z;
        start_f = end_f;
    }

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

    std::cout << Check(r_tensor, r_tensor_tst, M) << "\n";

    // print_double_matrix_rowmajor(r_tensor_tst, M, "AAA");
    // std::cout << CountZeros(r_tensor_tst, M * M) << "\n";

    return 0;
}