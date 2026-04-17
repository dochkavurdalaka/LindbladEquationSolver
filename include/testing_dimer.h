// этот файл чисто для тестирования
#pragma once

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


namespace Testing {

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

void GetHCoefDimer(std::vector<double>* h, const DimerModel& model, double t) {
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

std::vector<MKL_Complex16> GetLCoefDimer(int N, const DimerModel& model) {
    double mult_coeff = sqrt(model.gamma / (N - 1));
    std::vector<MKL_Complex16> l_coeff(N * N - 1, {0, 0});
    for (int n = 0; n + 1 < N; ++n) {
        int index = Mapping(n, n + 1, N);
        l_coeff[index].real = mult_coeff * sqrt(2. * (n + 1) * (N - 1 - n));
    }
    return l_coeff;
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

        // большая часть a у нас будут 0
        if (a.real != 0 or a.imag != 0) {
            k_vector[std::get<2>(el.first)] += -1. * (a * el.second).imag / N;
        }
    }

    return k_vector;
}

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

void do_what_i_want(double t, const double* v, double* result,
                 const std::vector<std::pair<std::tuple<int, int>, double>>& q_matrix,
                 const double* r_matrix, const double* k_vector, double* workspace, int M) {
    double* A = workspace;

    // 1. Вычисляем матрицу A(t) = Q(t) + R
    // cblas_dcopy(M * M, r_matrix, 1, workspace, 1);


    // 2. Вычисляем A(t) * v
    // cblas_dgemv: result = alpha*A*v + beta*result
    // Мы хотим result = 1.0 * A * v + 0.0 * result
    cblas_dgemv(CblasRowMajor, CblasNoTrans, M, M,  // Размеры матрицы A
                1.0,                                // alpha
                r_matrix, M,                               // Матрица A и ее lda
                v, 1,                               // Вектор v и его инкремент
                0.0,                                // beta
                result, 1);                         // Результирующий вектор и его инкремент

}


// Реализация нашей векторной функции f(t, v) = (Q(t) + R)v + K
// result = (Q(t) + R)v + K
// сказать Линеву про mkl_sparse_d_mv
void do_what_i_want2(double t, const double* v, double* result,
                 const std::vector<std::pair<std::tuple<int, int>, double>>& q_matrix,
                 const double* r_matrix, const double* k_vector, double* workspace, int M) {
    double* A = workspace;

    // 1. Вычисляем матрицу A(t) = Q(t) + R
    memset(A, 0, M * M * sizeof(double));

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
    // cblas_daxpy(M, 1.0, k_vector, 1, result, 1);
}


void do_what_i_want3(double t, const double* v, double* result,
                 const std::vector<std::pair<std::tuple<int, int>, double>>& q_matrix,
                 const double* r_matrix, const double* k_vector, double* workspace, int M) {
    double* A = workspace;

    // 1. Вычисляем матрицу A(t) = Q(t) + R
    memset(A, 0, M * M * sizeof(double));

    for (const auto& [tpl, coeff] : q_matrix) {
        size_t index = std::get<0>(tpl) * M + std::get<1>(tpl);
        A[index] += coeff;
    }

    print_vector("result_hardcore", t, A, M * M);

    // 2. Вычисляем A(t) * v
    // cblas_dgemv: result = alpha*A*v + beta*result
    // Мы хотим result = 1.0 * A * v + 0.0 * result
    // cblas_dgemv(CblasRowMajor, CblasNoTrans, M, M,  // Размеры матрицы A
    //             1.0,                                // alpha
    //             A, M,                               // Матрица A и ее lda
    //             v, 1,                               // Вектор v и его инкремент
    //             0.0,                                // beta
    //             result, 1);                         // Результирующий вектор и его инкремент

    // 3. Добавляем вектор K
    // cblas_daxpy: result = 1.0 * K + result
    // cblas_daxpy(M, 1.0, k_vector, 1, result, 1);
}

void FillHamiltonian(MKL_Complex16* hamiltonian, const DimerModel& model, double t) {
    int N = model.N;
    memset(hamiltonian, 0, N * N * sizeof(MKL_Complex16));
    for (int n = 0; n < N; ++n) {
        int index = n * N + n;
        hamiltonian[index].real =
            (2.0 * model.U / (N - 1)) * (n * (n - 1) + (N - 1 - n) * (N - 2 - n));
        hamiltonian[index].real += (model.E + model.A * model.theta(t)) * (N - 1 - 2 * n);
    }

    for (int n = 0; n < N - 1; ++n) {
        int index1 = n * N + n + 1;
        int index2 = (n + 1) * N + n;
        hamiltonian[index1].real = hamiltonian[index2].real =
            -model.J * sqrt((n + 1) * (N - 1 - n));
    }
}

double AllFunc() {
    // Параметры
    int N = 5;
    int M = N * N - 1;
    DimerModel model;

    model.theta = [T = model.T](double t) {
        t = fmod(t, T);
        if (t < T / 2) {
            return 1;
        }
        return -1;
    };

    model.N = N;

    // --- RNG
    VSLStreamStatePtr stream;
    int seed = 0;
    if (vslNewStream(&stream, VSL_BRNG_MT19937, seed) != VSL_STATUS_OK) {
        std::cerr << "Failed to create VSL stream in GenerateTraceless function\n";
        return 0;
    }

    MKL_Complex16* hamiltonian = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);

    MKL_Complex16* rho = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);

    // матрица hamiltonian здесь используется чисто как буфер для вычислений
    GenerateDensity(N, stream, rho, hamiltonian);
    vslDeleteStream(&stream);

    FillHamiltonian(hamiltonian, model, 0.0);

    // auto h_2 = GetHCoef(hamiltonian, N);
    // (void)h_2;

    // GenerateTracelessHamiltonian(N, 2, stream, hamiltonian);

    // создаем массив линдбладианов и заполняем его нулями
    // constexpr size_t lindbladian_cnt = 1;
    // std::array<MKL_Complex16*, lindbladian_cnt> lindbladians;
    // for (size_t i = 0; i < lindbladian_cnt; ++i) {
    //     lindbladians[i] = GenerateLp(N, i);
    // }

    // // вычисляем коэффициенты h
    // std::vector<double> h_coeff = GetHCoef(hamiltonian, N);

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

    // auto cmp = [](const std::pair<std::tuple<int, int, int>, double>& left,
    //               const std::pair<std::tuple<int, int, int>, double>& right) {
    //     if (std::get<2>(left.first) == std::get<2>(right.first)) {
    //         return std::get<1>(left.first) < std::get<1>(right.first);
    //     }
    //     return std::get<2>(left.first) < std::get<2>(right.first);
    // };

    // std::sort(f_tensor.begin(), f_tensor.end(), cmp);

    // std::vector<double> h_coeff(N * N - 1, 0);
    //  auto h_coeff = GetHCoef(hamiltonian, N);
    //  GetHCoefDimer(&h_coeff, model, t0);

    // sparse_matrix_t q = GenerateSparseMatrixQ(f_tensor, h_coeff);

    // MKL_INT rows = N * N - 1;
    // MKL_INT cols = N * N - 1;
    // double* lenta1 = sparse_to_dense_flat(q1, rows, cols);

    // (void)lenta1;

    // SparseQBuilder q_builder(f_tensor, h_coeff, N);
    // sparse_matrix_t q_matrix = q_builder.get_matrix();

    // GetHCoefDimer(&h_1, model, N, 2.0);
    // builder.update_values(h_1);
    // // GetHCoefDimer(&h_1, model, N, 5.0);
    // // builder.update_values(h_1);

    // double* lenta2 = sparse_to_dense_flat(q2, rows, cols);

    // (void)lenta2;

    // auto q_matrix = GenerateCOOMatrixQ(&f_tensor, h_1);

    double* k_vector = GenerateVectorFunctorK(N, kossakovski_func, f_tensor);

    double* r_matrix = GenerateMatrixRVec(N, l_coeffs, l_coeffs_conjugate, &f_tensor, &z_tensor);

    // SparseRBuilder r_builder(l_coeffs, l_coeffs_conjugate, &f_tensor, &z_tensor, N);
    // sparse_matrix_t r_matrix = r_builder.get_matrix();

    // double* r_matrix2 = sparse_to_dense_flat(r2, rows, cols);

    // double eps = 0.;
    // for (int i = 0; i < rows * cols; ++i) {
    //     eps = std::max(eps, abs(r_matrix1[i] - r_matrix2[i]));
    // }

    // std::cout << eps << "\n";

    // // print_double_matrix_rowmajor(r_matrix, M, "r_matrix");

    // Начальные условия
    // double t0 = 0.0;

    // Временные векторы для РК4
    double* k1 = (double*)mkl_malloc(M * sizeof(double), 64);
    double* k2 = (double*)mkl_malloc(M * sizeof(double), 64);
    double* k3 = (double*)mkl_malloc(M * sizeof(double), 64);
    double* k4 = (double*)mkl_malloc(M * sizeof(double), 64);
    double* v_temp = (double*)mkl_malloc(M * sizeof(double), 64);
    double* v_sum = (double*)mkl_malloc(M * sizeof(double), 64);

    // Параметры
    double t_end = 0.1;
    double h = 0.01;

    double t = t0;
    std::cout << std::fixed << std::setprecision(6);
    double* workspace = (double*)mkl_malloc(M * M * sizeof(double), 64);
    double* v = GetVCoef(rho, N);
    // std::cout << "lala: ";
    // print_vector("v", t, v, M);

    FillHamiltonian(hamiltonian, model, t);
    auto h_coeff = GetHCoef(hamiltonian, N);

    auto q_matrix = GenerateCOOMatrixQ(&f_tensor, h_coeff);

    // --- Шаг метода Рунге-Кутты 4-го порядка ---

    // // k1 = f(t, v)
    calculate_f(t, v, k1, q_matrix, r_matrix, k_vector, workspace, M);
    print_vector("k1_hardcore", t, k1, M);

    // FillHamiltonian(hamiltonian, model, t);
    // h_coeff = GetHCoef(hamiltonian, N);
    // print_vector("h_coeff", t, h_coeff.data(), M);

    // print_vector("k1", t, k1, M);

    double* result = (double*)mkl_malloc(M * sizeof(double), 64);
    // do_what_i_want3(t, v, result, q_matrix, r_matrix, k_vector, workspace, M);

    // SparseQBuilder q_builder(f_tensor, h_coeff, N);
    // sparse_matrix_t q_1x = q_builder.get_matrix();

    // GetHCoefDimer(&h_1, model, N, 2.0);
    // builder.update_values(h_1);
    // // GetHCoefDimer(&h_1, model, N, 5.0);
    // // builder.update_values(h_1);

    MKL_INT rows = M;
    MKL_INT cols = M;

    // double* lenta2 = sparse_to_dense_flat(q_1x, rows, cols);

    // print_vector("result_hardcore", 0, lenta2, M * M);

    // while (t < t_end + h / 2) {
    //     FillHamiltonian(hamiltonian, model, t);
    //     auto h_coeff = GetHCoef(hamiltonian, N);

    //     auto q_matrix = GenerateCOOMatrixQ(&f_tensor, h_coeff);

    //     // --- Шаг метода Рунге-Кутты 4-го порядка ---

    //     // k1 = f(t, v)
    //     calculate_f(t, v, k1, q_matrix, r_matrix, k_vector, workspace, M);

    //     // k2 = f(t + h/2, v + h/2 * k1)
    //     cblas_dcopy(M, v, 1, v_temp, 1);            // v_temp = v
    //     cblas_daxpy(M, 0.5 * h, k1, 1, v_temp, 1);  // v_temp = v + 0.5*h*k1
    //     calculate_f(t + 0.5 * h, v_temp, k2, q_matrix, r_matrix, k_vector, workspace, M);

    //     // k3 = f(t + h/2, v + h/2 * k2)
    //     cblas_dcopy(M, v, 1, v_temp, 1);            // v_temp = v
    //     cblas_daxpy(M, 0.5 * h, k2, 1, v_temp, 1);  // v_temp = v + 0.5*h*k2
    //     calculate_f(t + 0.5 * h, v_temp, k3, q_matrix, r_matrix, k_vector, workspace, M);

    //     // k4 = f(t + h, v + h * k3)
    //     cblas_dcopy(M, v, 1, v_temp, 1);      // v_temp = v
    //     cblas_daxpy(M, h, k3, 1, v_temp, 1);  // v_temp = v + h*k3
    //     calculate_f(t + h, v_temp, k4, q_matrix, r_matrix, k_vector, workspace, M);

    //     // Обновляем v: v = v + (h/6) * (k1 + 2k2 + 2k3 + k4)
    //     cblas_dcopy(M, k1, 1, v_sum, 1);          // v_sum = k1
    //     cblas_daxpy(M, 2.0, k2, 1, v_sum, 1);     // v_sum = k1 + 2*k2
    //     cblas_daxpy(M, 2.0, k3, 1, v_sum, 1);     // v_sum = k1 + 2*k2 + 2*k3
    //     cblas_daxpy(M, 1.0, k4, 1, v_sum, 1);     // v_sum = k1 + 2*k2 + 2*k3 + k4
    //     cblas_daxpy(M, h / 6.0, v_sum, 1, v, 1);  // v = v + (h/6)*v_sum

    //     print_vector("v", t, v, M);

    //     t += h;
    // }

    // // MKL_Complex16* matrix_rho = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
    // // memset(matrix_rho, 0, N * N * sizeof(MKL_Complex16));

    // // for (int i = 0; i < N * N; ++i) {
    // //     for (int j = 0; j < M; ++j) {
    // //         matrix_rho[i] += v[j] * basis_array[j][i];
    // //     }
    // // }

    // // // print_matrix_rowmajor(matrix_rho, N, "result");
    // // MKL_Complex16 tr = Trace(matrix_rho, N);

    // // std::cout << std::fixed << std::setprecision(17) << tr.real << "\n";

    // // std::cout << std::fixed << std::setprecision(17) << check_hermitian_approx(matrix_rho, N)
    // //           << "\n";

    // mkl_free(hamiltonian);
    // for (auto& el : lindbladians) {
    //     mkl_free(el);
    // }
    // mkl_free(rho);
    // mkl_free(workspace);
    // mkl_free(k_vector);
    // mkl_free(r_matrix);

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
}  // namespace Testing