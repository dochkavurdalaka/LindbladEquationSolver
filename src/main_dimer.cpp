#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <chrono>

#include "mkl.h"
#include "generate_matrices_v2.h"
#include "mkl_complex16.h"
#include "lindblad_utils.h"

#include "lindblad_utils_dimer.h"

#include "mkl_spblas.h"

#include "testing_dimer.h"

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


void GetBaseHCoefDimer(std::vector<double>* h, const DimerModel& model, double t) {
    std::fill(h->begin(), h->end(), 0);

    int N = model.N;
    auto& h_coeff = *h;

    for (int n = 0; n + 1 < N; ++n) {
        int index = Mapping(n, n + 1, N);
        h_coeff[index] = -model.J * sqrt(2. * (n + 1) * (N - 1 - n));
    }

    std::vector<double> hamiltonian_diag(N, 0);
    for (int n = 0; n < N; ++n) {
        hamiltonian_diag[n] = (2.0 * model.U / (N - 1)) * (n * (n - 1) + (N - 1 - n) * (N - 2 - n));
        hamiltonian_diag[n] += (model.E + model.A * model.theta(t)) * (N - 1 - 2 * n);
    }

    for (int l = 0; l < N - 1; ++l) {
        double coeff = 0.;

        for (int k = 0; k < l + 1; ++k) {
            coeff += hamiltonian_diag[k] / sqrt((l + 1) * (l + 2));
        }

        coeff += -sqrt(l + 1) * hamiltonian_diag[l + 1] / sqrt(l + 2);

        h_coeff[N * (N - 1) + l] = coeff;
    }
}


void GetUpdateHCoefDimer(std::vector<double>* h, const DimerModel& model, double t) {
    int N = model.N;

    std::fill(h->begin() + N*(N-1), h->end(), 0);

    auto& h_coeff = *h;

    std::vector<double> hamiltonian_diag(N, 0);
    for (int n = 0; n < N; ++n) {
        hamiltonian_diag[n] = (2.0 * model.U / (N - 1)) * (n * (n - 1) + (N - 1 - n) * (N - 2 - n));
        hamiltonian_diag[n] += (model.E + model.A * model.theta(t)) * (N - 1 - 2 * n);
    }

    for (int l = 0; l < N - 1; ++l) {
        double coeff = 0.;

        for (int k = 0; k < l + 1; ++k) {
            coeff += hamiltonian_diag[k] / sqrt((l + 1) * (l + 2));
        }

        coeff += -sqrt(l + 1) * hamiltonian_diag[l + 1] / sqrt(l + 2);

        h_coeff[N * (N - 1) + l] = coeff;
    }
}


std::vector<MKL_Complex16> GetLCoefDimer(int N, const DimerModel& model) {
    double mult_coeff = sqrt(model.gamma / (N - 1));
    std::vector<MKL_Complex16> l_coeff(N * N - 1, {0, 0});
    for (int n = 0; n + 1 < N; ++n) {
        int index = Mapping(n, n + 1, N);
        l_coeff[index].real = mult_coeff * sqrt(2. * (n + 1) * (N - 1 - n));
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

// функция, которая переводит sparse_matrix_t в CSR формате в обычную матрицу (ленту)
double* sparse_to_dense_flat(const sparse_matrix_t& mat, MKL_INT& rows, MKL_INT& cols) {
    sparse_index_base_t indexing;
    MKL_INT *rows_start, *rows_end, *col_indices;
    double* values;

    // Экспортируем данные матрицы
    sparse_status_t status = mkl_sparse_d_export_csr(mat, &indexing, &rows, &cols, &rows_start,
                                                     &rows_end, &col_indices, &values);

    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to export sparse matrix");
    }

    // Выделяем память под плотный массив (инициализируем нулями)
    size_t total_elements = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    double* dense = new double[total_elements]();  // Круглые скобки инициализируют нулями

    // Заполняем массив значениями из CSR формата
    for (MKL_INT i = 0; i < rows; ++i) {
        MKL_INT row_start = rows_start[i] - indexing;
        MKL_INT row_end = rows_end[i] - indexing;

        // Вычисляем смещение для текущей строки в плоском массиве
        size_t row_offset = static_cast<size_t>(i) * static_cast<size_t>(cols);

        for (MKL_INT j = row_start; j < row_end; ++j) {
            MKL_INT col = col_indices[j] - indexing;
            dense[row_offset + col] = values[j];
        }
    }

    return dense;
}

// Реализация нашей векторной функции f(t, v) = (Q(t) + R)v + K
// result = (Q(t) + R)v + K
// сказать Линеву про mkl_sparse_d_mv
void calculate_f(double t, sparse_matrix_t q_matrix, sparse_matrix_t r_matrix, const double* v,
                 const double* k_vector, std::vector<double>* h_coef, SparseQBuilder* q_builder,
                 const DimerModel& model, double* result) {
    int N = model.N;
    int M = N * N - 1;

    auto& h_coeff = *h_coef;
    GetUpdateHCoefDimer(&h_coeff, model, t);
    q_builder->update_values(h_coeff);

    cblas_dcopy(M, k_vector, 1, result, 1);

    // Создаем дескриптор один раз с помощью static
    static struct matrix_descr descr = {
        SPARSE_MATRIX_TYPE_GENERAL,  // Обычная матрица
        SPARSE_FILL_MODE_UPPER,      // Не используется, но заполняем
        SPARSE_DIAG_NON_UNIT         // Не используется, но заполняем
    };

    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, q_matrix, descr, v, 1.0, result);

    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, r_matrix, descr, v, 1.0, result);
}

int main() {
    // Параметры
    const int N = 5;
    const int M = N * N - 1;
    DimerModel model;

    model.theta = [T = model.T](double t) {
        t = fmod(t, T);
        if (t < T / 2) {
            return 1;
        }
        return -1;
    };

    model.N = N;


    // --- Генератор рандомных чисел в нормальном распределении
    VSLStreamStatePtr stream;
    int seed = 0;
    if (vslNewStream(&stream, VSL_BRNG_MT19937, seed) != VSL_STATUS_OK) {
        std::cerr << "Failed to create VSL stream in GenerateTraceless function\n";
        return 0;
    }

    MKL_Complex16* bufer = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
    MKL_Complex16* rho = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
    // матрица bufer здесь используется чисто как буфер для вычислений
    GenerateDensity(N, stream, rho, bufer);

    vslDeleteStream(&stream);

    // вычисляем коэффициенты l
    std::vector<MKL_Complex16> l_coeffs;
    l_coeffs = GetLCoefDimer(N, model);

    std::vector<MKL_Complex16> l_coeffs_conjugate(l_coeffs);

    for (auto& elem : l_coeffs_conjugate) {
        elem = Conjugate(elem);
    }

    auto kossakovski_func = [&l_coeffs, &l_coeffs_conjugate](size_t i, size_t j) {
        MKL_Complex16 result = {0., 0.};
        result += l_coeffs[i] * l_coeffs_conjugate[j];
        return result;
    };

    auto f_tensor = GenerateTensorF(N);
    auto d_tensor = GenerateTensorD(N);
    auto z_tensor = GenerateTensorZ(f_tensor, d_tensor);

    double t0 = 0.0;

    std::vector<double> h_coeff(N * N - 1, 0);
    GetBaseHCoefDimer(&h_coeff, model, t0);

    SparseQBuilder q_builder(&f_tensor, h_coeff, N);
    sparse_matrix_t q_matrix = q_builder.get_matrix();

    double* k_vector = GenerateVectorKWithFunctor(N, kossakovski_func, f_tensor);

    SparseRBuilder r_builder(l_coeffs, l_coeffs_conjugate, &f_tensor, &z_tensor, N);
    sparse_matrix_t r_matrix = r_builder.get_matrix();

    // Временные векторы для РК4
    double* k1 = (double*)mkl_malloc(M * sizeof(double), 64);
    double* k2 = (double*)mkl_malloc(M * sizeof(double), 64);
    double* k3 = (double*)mkl_malloc(M * sizeof(double), 64);
    double* k4 = (double*)mkl_malloc(M * sizeof(double), 64);
    double* v_temp = (double*)mkl_malloc(M * sizeof(double), 64);

    // Параметры
    double t_end = 1.1;
    double h = 0.01;

    double t = t0;
    std::cout << std::fixed << std::setprecision(6);
    double* v = GetVCoef(rho, N);

    while (t < t_end + h / 2) {

        // --- Шаг метода Рунге-Кутты 4-го порядка ---
        // k1 = f(t, v)
        calculate_f(t, q_matrix, r_matrix, v, k_vector, &h_coeff, &q_builder, model, k1);

        // k2 = f(t + h/2, v + h/2 * k1)
        cblas_dcopy(M, v, 1, v_temp, 1);            // v_temp = v
        cblas_daxpy(M, 0.5 * h, k1, 1, v_temp, 1);  // v_temp = v + 0.5*h*k1
        calculate_f(t + 0.5 * h, q_matrix, r_matrix, v_temp, k_vector, &h_coeff, &q_builder, model,
                    k2);

        // k3 = f(t + h/2, v + h/2 * k2)
        cblas_dcopy(M, v, 1, v_temp, 1);            // v_temp = v
        cblas_daxpy(M, 0.5 * h, k2, 1, v_temp, 1);  // v_temp = v + 0.5*h*k2
        calculate_f(t + 0.5 * h, q_matrix, r_matrix, v_temp, k_vector, &h_coeff, &q_builder, model, k3);

        // k4 = f(t + h, v + h * k3)
        cblas_dcopy(M, v, 1, v_temp, 1);      // v_temp = v
        cblas_daxpy(M, h, k3, 1, v_temp, 1);  // v_temp = v + h*k3
        calculate_f(t + h, q_matrix, r_matrix, v_temp, k_vector, &h_coeff, &q_builder, model, k4);

        // Обновляем v: v = v + (h/6) * (k1 + 2k2 + 2k3 + k4)
        cblas_dcopy(M, k1, 1, v_temp, 1);          // v_sum = k1
        cblas_daxpy(M, 2.0, k2, 1, v_temp, 1);     // v_sum = k1 + 2*k2
        cblas_daxpy(M, 2.0, k3, 1, v_temp, 1);     // v_sum = k1 + 2*k2 + 2*k3
        cblas_daxpy(M, 1.0, k4, 1, v_temp, 1);     // v_sum = k1 + 2*k2 + 2*k3 + k4
        cblas_daxpy(M, h / 6.0, v_temp, 1, v, 1);  // v = v + (h/6)*v_sum

        print_vector("v", t, v, M);
        t += h;
    }


    mkl_free(rho);
    mkl_free(bufer);
    mkl_free(k_vector);

    // Освобождение памяти
    mkl_free(v);
    mkl_free(k1);
    mkl_free(k2);
    mkl_free(k3);
    mkl_free(k4);
    mkl_free(v_temp);

    return 0;
}
