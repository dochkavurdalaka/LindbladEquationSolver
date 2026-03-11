#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <chrono>

#include "mkl.h"
#include "generate_matrices.h"
#include "mkl_complex16.h"
#include "lindblad_utils.h"

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

// Реализация нашей векторной функции f(t, v) = (Q(t) + R)v + K
// result = (Q(t) + R)v + K
void calculate_f(double t, const double* v, double* result,
                 const std::vector<std::pair<std::tuple<int, int>, double>>& q_matrix,
                 const double* r_matrix, const double* k_vector, double* workspace, int M) {
    double* A = workspace;

    // 1. Вычисляем матрицу A(t) = Q(t) + R
    cblas_dcopy(M * M, r_matrix, 1, workspace, 1);

    for (const auto& [tpl, coeff] : q_matrix) {
        size_t index = std::get<0>(tpl) * M + std::get<1>(tpl);
        A[index] += coeff;
    }

    // 2. Вычисляем A(t) * v
    // cblas_dgemv: result = alpha*A*v + beta*result
    // Мы хотим result = 1.0 * A * v + 0.0 * result
    cblas_dgemv(CblasRowMajor, CblasNoTrans, M, M,  // Размеры матрицы A
                1.0,                                // alpha
                A, M,                               // Матрица A и ее lda
                v, 1,                               // Вектор v и его инкремент
                0.0,                                // beta
                result, 1);                         // Результирующий вектор и его инкремент

    // 3. Добавляем вектор K
    // cblas_daxpy: result = 1.0 * K + result
    cblas_daxpy(M, 1.0, k_vector, 1, result, 1);
}

std::vector<double> GetHCoef(MKL_Complex16* hamiltonian, int N) {
    std::vector<double> h_coeff;
    h_coeff.reserve(N * N - 1);

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
    l_coeff.reserve(N * N - 1);

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

double* GetVCoef(MKL_Complex16* rho, int N) {
    double* v_coeff = (double*)mkl_malloc((N * N - 1) * sizeof(double), 64);
    size_t ind = 0;
    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            int index = j * N + k;
            v_coeff[ind] = sqrt(2.) * rho[index].real;
            ++ind;
        }
    }

    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            int index = k * N + j;
            v_coeff[ind] = sqrt(2.) * rho[index].imag;
            ++ind;
        }
    }

    for (int l = 0; l < N - 1; ++l) {
        double coeff = 0.;

        for (int k = 0; k < l + 1; ++k) {
            int index = k * N + k;
            coeff += rho[index].real / sqrt((l + 1) * (l + 2));
        }

        int index = (l + 1) * N + (l + 1);
        coeff += -sqrt(l + 1) * rho[index].real / sqrt(l + 2);

        v_coeff[ind] = coeff;
        ++ind;
    }

    return v_coeff;
}

// Проверка приближенной эрмитовости: M == M^† в пределах eps
double check_hermitian_approx(const MKL_Complex16* M, int d) {
    double result = 0.0;
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            const MKL_Complex16& a = M[i * d + j];
            const MKL_Complex16& b = M[j * d + i];  // should be conj(a)
            double re_diff = a.real - b.real;
            double im_diff = a.imag + b.imag;  // a.imag - (-b.imag) since b should be conj(a)
            result = std::max(re_diff * re_diff + im_diff * im_diff, result);
        }
    }
    return result;
}

double* GenerateVectorFunctorK(
    int N, auto kossakovski_func,
    const std::vector<std::pair<std::tuple<int, int, int>, double>>& f_tensor) {
    int M = N * N - 1;
    double* k_vector = (double*)mkl_malloc(M * sizeof(double), 64);
    memset(k_vector, 0, M * sizeof(double));

    for (const auto& el : f_tensor) {
        MKL_Complex16 a = kossakovski_func(std::get<0>(el.first), std::get<1>(el.first));
        k_vector[std::get<2>(el.first)] += -1. * (a * el.second).imag / N;
    }

    return k_vector;
}

int main() {
    // Параметры
    int N = 5;
    int M = N * N - 1;

    // создаем гамильтониан с нулевым следом
    MKL_Complex16* hamiltonian;
    GenerateTracelessHamiltonian(N, 2, hamiltonian);

    // создаем массив линдбладианов и заполняем его нулями
    constexpr size_t lindbladian_cnt = 2;
    std::array<MKL_Complex16*, lindbladian_cnt> lindbladians;
    for (size_t i = 0; i < lindbladian_cnt; ++i) {
        GenerateLp(N, i, lindbladians[i]);
    }

    MKL_Complex16* rho;
    GenerateDensity(N, 2, rho);

    // вычисляем коэффициенты h
    std::vector<double> h_coeff = GetHCoef(hamiltonian, N);

    // вычисляем коэффициенты l
    std::array<std::vector<MKL_Complex16>, lindbladian_cnt> l_coeffs;
    for (size_t i = 0; i < lindbladian_cnt; ++i) {
        l_coeffs[i] = GetLCoef(lindbladians[i], N);
    }

    std::array<std::vector<MKL_Complex16>, lindbladian_cnt> l_coeffs_conjugate(l_coeffs);

    for (auto& vec : l_coeffs_conjugate) {
        for (auto& elem : vec) {
            elem = Conjugate(elem);
        }
    }

    auto kossakovski_func = [&l_coeffs, &l_coeffs_conjugate, lindbladian_cnt](size_t i, size_t j) {
        MKL_Complex16 result = {0., 0.};
        for (size_t cnt = 0; cnt < lindbladian_cnt; ++cnt) {
            result += l_coeffs[cnt][i] * l_coeffs_conjugate[cnt][j];
        }
        return result;
    };

    auto f_tensor = GenerateTensorF(N);
    auto d_tensor = GenerateTensorD(N);
    auto z_tensor = GenerateTensorZ(f_tensor, d_tensor);

    auto q_matrix = GenerateCOOMatrixQ(&f_tensor, h_coeff);

    double* k_vector = GenerateVectorFunctorK(N, kossakovski_func, f_tensor);

    double* r_matrix = GenerateMatrixRVec(N, l_coeffs, l_coeffs_conjugate, &f_tensor, &z_tensor);

    // print_double_matrix_rowmajor(r_matrix, M, "r_matrix");

    // Начальные условия
    double t0 = 0.0;

    // Временные векторы для РК4
    double* k1 = (double*)mkl_malloc(M * sizeof(double), 64);
    double* k2 = (double*)mkl_malloc(M * sizeof(double), 64);
    double* k3 = (double*)mkl_malloc(M * sizeof(double), 64);
    double* k4 = (double*)mkl_malloc(M * sizeof(double), 64);
    double* v_temp = (double*)mkl_malloc(M * sizeof(double), 64);
    double* v_sum = (double*)mkl_malloc(M * sizeof(double), 64);

    // Параметры
    double t_end = 1.3;
    double h = 0.01;

    double t = t0;
    std::cout << std::fixed << std::setprecision(6);
    double* workspace = (double*)mkl_malloc(M * M * sizeof(double), 64);
    double* v = GetVCoef(rho, N);
    while (t < t_end + h / 2) {

        // --- Шаг метода Рунге-Кутты 4-го порядка ---

        // k1 = f(t, v)
        calculate_f(t, v, k1, q_matrix, r_matrix, k_vector, workspace, M);

        // k2 = f(t + h/2, v + h/2 * k1)
        cblas_dcopy(M, v, 1, v_temp, 1);            // v_temp = v
        cblas_daxpy(M, 0.5 * h, k1, 1, v_temp, 1);  // v_temp = v + 0.5*h*k1
        calculate_f(t + 0.5 * h, v_temp, k2, q_matrix, r_matrix, k_vector, workspace, M);

        // k3 = f(t + h/2, v + h/2 * k2)
        cblas_dcopy(M, v, 1, v_temp, 1);            // v_temp = v
        cblas_daxpy(M, 0.5 * h, k2, 1, v_temp, 1);  // v_temp = v + 0.5*h*k2
        calculate_f(t + 0.5 * h, v_temp, k3, q_matrix, r_matrix, k_vector, workspace, M);

        // k4 = f(t + h, v + h * k3)
        cblas_dcopy(M, v, 1, v_temp, 1);      // v_temp = v
        cblas_daxpy(M, h, k3, 1, v_temp, 1);  // v_temp = v + h*k3
        calculate_f(t + h, v_temp, k4, q_matrix, r_matrix, k_vector, workspace, M);

        // Обновляем v: v = v + (h/6) * (k1 + 2k2 + 2k3 + k4)
        cblas_dcopy(M, k1, 1, v_sum, 1);          // v_sum = k1
        cblas_daxpy(M, 2.0, k2, 1, v_sum, 1);     // v_sum = k1 + 2*k2
        cblas_daxpy(M, 2.0, k3, 1, v_sum, 1);     // v_sum = k1 + 2*k2 + 2*k3
        cblas_daxpy(M, 1.0, k4, 1, v_sum, 1);     // v_sum = k1 + 2*k2 + 2*k3 + k4
        cblas_daxpy(M, h / 6.0, v_sum, 1, v, 1);  // v = v + (h/6)*v_sum

        print_vector("v", t, v, M);
        t += h;
    }

    // MKL_Complex16* matrix_rho = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
    // memset(matrix_rho, 0, N * N * sizeof(MKL_Complex16));

    // for (int i = 0; i < N * N; ++i) {
    //     for (int j = 0; j < M; ++j) {
    //         matrix_rho[i] += v[j] * basis_array[j][i];
    //     }
    // }

    // // print_matrix_rowmajor(matrix_rho, N, "result");
    // MKL_Complex16 tr = Trace(matrix_rho, N);

    // std::cout << std::fixed << std::setprecision(17) << tr.real << "\n";

    // std::cout << std::fixed << std::setprecision(17) << check_hermitian_approx(matrix_rho, N)
    //           << "\n";

    mkl_free(hamiltonian);
    for (auto& el : lindbladians) {
        mkl_free(el);
    }
    mkl_free(rho);
    mkl_free(workspace);
    mkl_free(k_vector);
    mkl_free(r_matrix);

    // Освобождение памяти
    mkl_free(v);
    mkl_free(k1);
    mkl_free(k2);
    mkl_free(k3);
    mkl_free(k4);
    mkl_free(v_temp);
    mkl_free(v_sum);

    return 0;
}
