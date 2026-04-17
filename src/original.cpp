#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>

#include "mkl.h"
#include "generate_matrices.h"
#include "original.h"
#include "mkl_complex16.h"

// здесь и далее решаем систему для константного H
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
// В нашей модели H = const -> Q = const -> зараннее предрасчитываем  S = Q + R и передаем в функцию
// result = Sv + K
void CalculateFunc(const double* S, const double* v, const double* K, int M, double* result) {
    cblas_dcopy(M, K, 1, result, 1);
    // cblas_dgemv: result = alpha*A*v + beta*result
    // Мы хотим result = 1.0 * A * v + 0.0 * result
    cblas_dgemv(CblasRowMajor, CblasNoTrans, M, M,  // Размеры матрицы A
                1.0,                                // alpha
                S, M,                               // Матрица A и ее lda
                v, 1,                               // Вектор v и его инкремент
                1.0,                                // beta
                result, 1);                         // Результирующий вектор и его инкремент
}

struct Package {

    Package(int N) {
        int M = N * N - 1;
        k1 = (double*)mkl_malloc(M * sizeof(double), 64);
        k2 = (double*)mkl_malloc(M * sizeof(double), 64);
        k3 = (double*)mkl_malloc(M * sizeof(double), 64);
        k4 = (double*)mkl_malloc(M * sizeof(double), 64);
        v_temp = (double*)mkl_malloc(M * sizeof(double), 64);
    }

    double* k1;
    double* k2;
    double* k3;
    double* k4;
    double* v_temp;

    ~Package() {
        mkl_free(k1);
        mkl_free(k2);
        mkl_free(k3);
        mkl_free(k4);
        mkl_free(v_temp);
    }
};

std::vector<double> GetHCoef(const std::vector<MKL_Complex16*>& basis_array,
                             MKL_Complex16* hamiltonian, int N, MKL_Complex16* result_matrix) {
    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    std::vector<double> h_coeff;
    for (MKL_Complex16* mat : basis_array) {
        cblas_zgemm(CblasRowMajor,  // Указывает, что матрицы хранятся построчно
                                    // (стандарт для C/C++)
                    CblasNoTrans,   // Операция для A: не транспонировать
                    CblasNoTrans,   // Операция для B: не транспонировать
                    N,              // Количество строк в матрице A (и C)
                    N,              // Количество столбцов в матрице B (и C)
                    N,              // Количество столбцов в A и строк в B
                    &alpha,         // Указатель на скаляр alpha
                    hamiltonian,    // Матрица A
                    N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                    // количество столбцов.
                    mat,            // Матрица B
                    N,              // Ведущий размер для B.
                    &beta,          // Указатель на скаляр beta
                    result_matrix,  // Матрица C (результат)
                    N               // Ведущий размер для C.
        );
        h_coeff.push_back(Trace(result_matrix, N).real);
    }

    return h_coeff;
}

std::vector<MKL_Complex16> GetLCoef(const std::vector<MKL_Complex16*>& basis_array,
                                    MKL_Complex16* lindbladian, int N,
                                    MKL_Complex16* result_matrix) {
    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    std::vector<MKL_Complex16> l_coeff;

    for (MKL_Complex16* mat : basis_array) {
        cblas_zgemm(CblasRowMajor,  // Указывает, что матрицы хранятся построчно
                                    // (стандарт для C/C++)
                    CblasNoTrans,   // Операция для A: не транспонировать
                    CblasNoTrans,   // Операция для B: не транспонировать
                    N,              // Количество строк в матрице A (и C)
                    N,              // Количество столбцов в матрице B (и C)
                    N,              // Количество столбцов в A и строк в B
                    &alpha,         // Указатель на скаляр alpha
                    lindbladian,    // Матрица A
                    N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                    // количество столбцов.
                    mat,            // Матрица B
                    N,              // Ведущий размер для B.
                    &beta,          // Указатель на скаляр beta
                    result_matrix,  // Матрица C (результат)
                    N               // Ведущий размер для C.
        );
        l_coeff.push_back(Trace(result_matrix, N));
    }
    return l_coeff;
}

double* GetVCoef(const std::vector<MKL_Complex16*>& basis_array, MKL_Complex16* rho, int N,
                 MKL_Complex16* result_matrix) {
    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    double* v_coeff = (double*)mkl_malloc((N * N - 1) * sizeof(double), 64);
    size_t ind = 0;
    for (MKL_Complex16* mat : basis_array) {
        // print_matrix_rowmajor(mat, 2, "f");
        cblas_zgemm(CblasRowMajor,  // Указывает, что матрицы хранятся построчно
                                    // (стандарт для C/C++)
                    CblasNoTrans,   // Операция для A: не транспонировать
                    CblasNoTrans,   // Операция для B: не транспонировать
                    N,              // Количество строк в матрице A (и C)
                    N,              // Количество столбцов в матрице B (и C)
                    N,              // Количество столбцов в A и строк в B
                    &alpha,         // Указатель на скаляр alpha
                    rho,            // Матрица A
                    N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                    // количество столбцов.
                    mat,            // Матрица B
                    N,              // Ведущий размер для B.
                    &beta,          // Указатель на скаляр beta
                    result_matrix,  // Матрица C (результат)
                    N               // Ведущий размер для C.
        );
        v_coeff[ind] = Trace(result_matrix, N).real;
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

// ----------------------------------------------------------------------
// helper: out = a + b * scalar (out = a + coeff * b), length N*N
void AddScaled(size_t nn, const double* a, const double* b, double coeff,
               double* out) {
 
    for (size_t k = 0; k < nn; ++k) {
        out[k] = a[k] + coeff * b[k];
    }
 
    // cblas_dcopy(nn, a, 1, out, 1);
    // MKL_Complex16 alpha = {coeff, 0.0};
    // cblas_daxpy(nn, coeff, b, 1, out, 1);
}

// ----------------------------------------------------------------------
// RK4 integrator step: rho_{n+1} = rho_n + dt/6*(k1 + 2k2 + 2k3 + k4)
// where k1 = L(rho_n), k2 = L(rho_n + dt/2*k1), ...
void RK4Step(int N, double* s_matrix, double* k_vector, double dt, double* v_in,
             double* v_out, const Package& package) {

    size_t M = N * N - 1;

    const auto& [k1, k2, k3, k4, v_temp] = package;

    // --- Шаг метода Рунге-Кутты 4-го порядка ---

    // k1 = f(t, v)
    CalculateFunc(s_matrix, v_in, k_vector, M, k1);

    // k2 = f(t + h/2, v + h/2 * k1)
    AddScaled(M, v_in, k1, dt * 0.5, v_temp);
    CalculateFunc(s_matrix, v_temp, k_vector, M, k2);

    // k3 = f(t + h/2, v + h/2 * k2)
    AddScaled(M, v_in, k2, dt * 0.5, v_temp);
    CalculateFunc(s_matrix, v_temp, k_vector, M, k3);

    // k4 = f(t + h, v + h * k3)
    AddScaled(M, v_in, k3, dt, v_temp);
    CalculateFunc(s_matrix, v_temp, k_vector, M, k4);


    // v_out = v_in + dt/6 * (k1 + 2k2 + 2k3 + k4)
    for (size_t k = 0; k < M; ++k) {
        v_out[k] = v_in[k] + dt / 6 * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
    }

}

int main() {
    // Параметры
    int N = 7;
    size_t M = N * N - 1;

    VSLStreamStatePtr stream;
    int seed = 0;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    MKL_Complex16* hamiltonian = GenerateTracelessHamiltonian(N, stream);
    MKL_Complex16* lindbladian = GenerateLp(N, stream);
    MKL_Complex16* rho = GenerateDensity(N, stream);
    vslDeleteStream(&stream);

    MKL_Complex16* dummy_result = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);

    // 1. Создаем вектор, который будет хранить указатели на наши матрицы
    std::vector<MKL_Complex16*> basis_array = CreateBasisArray(N);
    double* v = GetVCoef(basis_array, rho, N, dummy_result);

    // вычисляем коэффициенты h
    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    std::vector<double> h_coeff = GetHCoef(basis_array, hamiltonian, N, dummy_result);

    std::cout << "h_coeff:\n";
    for (auto el : h_coeff) {
        std::cout << el << " ";
    }
    std::cout << "\n";
    // вычисляем коэффициенты l
    std::vector<MKL_Complex16> l_coeff = GetLCoef(basis_array, lindbladian, N, dummy_result);

    std::cout << "l_coeff:\n";
    for (auto el : l_coeff) {
        std::cout << el.real << " " << el.imag << " ";
    }

    // Общее количество элементов
    size_t total_elements = M * M * M;

    // Выделение памяти для тензора
    double* f_tensor = (double*)mkl_malloc(total_elements * sizeof(double), 64);

    double* d_tensor = (double*)mkl_malloc(total_elements * sizeof(double), 64);

    MKL_Complex16* z_tensor =
        (MKL_Complex16*)mkl_malloc(total_elements * sizeof(MKL_Complex16), 64);

    std::vector<std::vector<MKL_Complex16*>> commutator(M);
    std::vector<std::vector<MKL_Complex16*>> anticommutator(M);
    for (size_t m = 0; m < M; ++m) {
        commutator[m].resize(M);
        anticommutator[m].resize(M);
        for (size_t n = 0; n < M; ++n) {
            commutator[m][n] = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            Commutator(basis_array[m], basis_array[n], commutator[m][n], N);
            anticommutator[m][n] = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            AntiCommutator(basis_array[m], basis_array[n], anticommutator[m][n], N);
        }
    }

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < M; ++n) {
            for (size_t s = 0; s < M; ++s) {
                // Вычисляем линейный индекс
                size_t index = m * M * M + n * M + s;

                cblas_zgemm(CblasRowMajor,   // Указывает, что матрицы хранятся построчно
                                             // (стандарт для C/C++)
                            CblasNoTrans,    // Операция для A: не транспонировать
                            CblasNoTrans,    // Операция для B: не транспонировать
                            N,               // Количество строк в матрице A (и C)
                            N,               // Количество столбцов в матрице B (и C)
                            N,               // Количество столбцов в A и строк в B
                            &alpha,          // Указатель на скаляр alpha
                            basis_array[s],  // Матрица A
                            N,  // Ведущий размер (leading dimension) для A. Для RowMajor
                                // это количество столбцов.
                            commutator[m][n],  // Матрица B
                            N,                 // Ведущий размер для B.
                            &beta,             // Указатель на скаляр beta
                            dummy_result,      // Матрица C (результат)
                            N                  // Ведущий размер для C.
                );

                f_tensor[index] = Trace(dummy_result, N).imag;

                cblas_zgemm(CblasRowMajor,   // Указывает, что матрицы хранятся построчно
                                             // (стандарт для C/C++)
                            CblasNoTrans,    // Операция для A: не транспонировать
                            CblasNoTrans,    // Операция для B: не транспонировать
                            N,               // Количество строк в матрице A (и C)
                            N,               // Количество столбцов в матрице B (и C)
                            N,               // Количество столбцов в A и строк в B
                            &alpha,          // Указатель на скаляр alpha
                            basis_array[s],  // Матрица A
                            N,  // Ведущий размер (leading dimension) для A. Для RowMajor
                                // это количество столбцов.
                            anticommutator[m][n],  // Матрица B
                            N,                     // Ведущий размер для B.
                            &beta,                 // Указатель на скаляр beta
                            dummy_result,          // Матрица C (результат)
                            N                      // Ведущий размер для C.
                );

                d_tensor[index] = Trace(dummy_result, N).real;

                z_tensor[index].real = f_tensor[index];
                z_tensor[index].imag = d_tensor[index];
            }
        }
    }

    double* q_matrix = (double*)mkl_malloc(M * M * sizeof(double), 64);
    memset(q_matrix, 0, M * M * sizeof(double));

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < M; ++n) {
            for (size_t s = 0; s < M; ++s) {
                size_t q_index = s * M + n;
                size_t f_index = m * M * M + n * M + s;

                q_matrix[q_index] += h_coeff[m] * f_tensor[f_index];
            }
        }
    }

    double* k_vector = (double*)mkl_malloc(M * sizeof(double), 64);
    memset(k_vector, 0, M * sizeof(double));

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < M; ++n) {
            for (size_t s = 0; s < M; ++s) {
                size_t f_index = m * M * M + n * M + s;
                k_vector[s] += (l_coeff[m] * Conjugate(l_coeff[n]) * f_tensor[f_index]).imag;
            }
        }
    }

    for (size_t s = 0; s < M; ++s) {
        k_vector[s] *= -1. / N;
        std::cout << k_vector[s] << " ";
    }

    double* r_matrix = (double*)mkl_malloc(M * M * sizeof(double), 64);

    // Циклы для вычисления каждого элемента r_sn
    for (size_t s = 0; s < M; ++s) {
        for (size_t n = 0; n < M; ++n) {

            double sum_jkl = 0.;  // Сумма по j, k, l

            for (size_t j = 0; j < M; ++j) {
                for (size_t k = 0; k < M; ++k) {
                    for (size_t l = 0; l < M; ++l) {
                        // Вычисляем z_jln * f_kls
                        size_t z_ind = j * M * M + l * M + n;
                        size_t f_ind = k * M * M + l * M + s;
                        MKL_Complex16 term = z_tensor[z_ind] * f_tensor[f_ind];

                        // Вычисляем z_bar_kln * f_jls
                        z_ind = k * M * M + l * M + n;
                        f_ind = j * M * M + l * M + s;
                        term += Conjugate(z_tensor[z_ind]) * f_tensor[f_ind];

                        sum_jkl += (l_coeff[j] * Conjugate(l_coeff[k]) * term).real;
                    }
                }
            }
            r_matrix[s * M + n] = -0.25 * sum_jkl;
        }
    }

    // print_double_matrix_rowmajor(r_matrix, M, "SSS");

    // Начальные условия
    double t0 = 0.0;

    Package package(N);

    // Параметры
    double t_end = 1.3;
    double h = 0.01;

    double t = t0;
    std::cout << std::fixed << std::setprecision(6);
    double* v_next = (double*)mkl_malloc(M * sizeof(double), 64);

    // Мы подсчитываем функцию f(t, v) = (Q(t) + R)v + K
    // В нашей модели H = const -> Q = const -> зараннее предрасчитываем  S = Q + R и передаем в функцию RK4Step

    // S = Q + R
    // R = 1.0 * Q + R  (то есть R = Q + R)
    cblas_daxpy(M * M, 1.0, q_matrix, 1, r_matrix, 1);
    double* s_matrix = r_matrix;


    while (t < t_end + h / 2) {
        RK4Step(N, s_matrix, k_vector, h, v, v_next, package);
 
        // swap rho and rho_next
        std::swap(v, v_next);

        print_vector("v", t, v, M);

        t += h;
    }

    MKL_Complex16* rho_final = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
    memset(rho_final, 0, N * N * sizeof(MKL_Complex16));

    for (int i = 0; i < N * N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            rho_final[i] += v[j] * basis_array[j][i];
        }
    }

    // print_matrix_rowmajor(matrix_rho, N, "result");

    // --- Очистка памяти ---
    for (MKL_Complex16* mat_ptr : basis_array) {
        mkl_free(mat_ptr);
    }

    mkl_free(dummy_result);
    mkl_free(hamiltonian);
    mkl_free(lindbladian);
    mkl_free(rho);
    mkl_free(rho_final);
    mkl_free(f_tensor);
    mkl_free(d_tensor);
    mkl_free(z_tensor);
    mkl_free(q_matrix);
    mkl_free(k_vector);
    mkl_free(r_matrix);

    mkl_free(v);
    mkl_free(v_next);

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < M; ++n) {
            mkl_free(commutator[m][n]);
            mkl_free(anticommutator[m][n]);
        }
    }

    return 0;
}
