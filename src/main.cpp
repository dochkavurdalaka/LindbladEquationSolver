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

// ----------------------------------------------------------------------
// small utility: print matrix
void print_mat(int N, const MKL_Complex16* A, const char* name) {
    printf("%s =\n", name);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            MKL_Complex16 x = A[i * N + j];
            printf("(% .6f,% .6f)  ", x.real, x.imag);
        }
        printf("\n");
    }
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
void AddScaled(size_t nn, const double* a, const double* b, double coeff, double* out) {

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
void RK4Step(int N, double* s_matrix, double* k_vector, double dt, double* v_in, double* v_out,
             const Package& package) {

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
    int P = 2;
    int M = N * N - 1;

    VSLStreamStatePtr stream;
    int seed = 0;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    MKL_Complex16* hamiltonian = GenerateTracelessHamiltonian(N, stream);
    std::vector<MKL_Complex16*> lindbladians(P);
    for (int i = 0; i < P; ++i) {
        lindbladians[i] = GenerateLp(N, stream);
    }
    MKL_Complex16* rho = GenerateDensity(N, stream);
    vslDeleteStream(&stream);

    // вычисляем коэффициенты h
    std::vector<double> h_coeff = GetHCoef(hamiltonian, N);

    // вычисляем коэффициенты l
    std::vector<std::vector<MKL_Complex16>> l_coeffs(P);
    for (int i = 0; i < P; ++i) {
        l_coeffs[i] = GetLCoef(lindbladians[i], N);
    }

    std::vector<std::vector<MKL_Complex16>> l_coeffs_conjugate(l_coeffs);
    for (auto& vec : l_coeffs_conjugate) {
        for (auto& elem : vec) {
            elem = Conjugate(elem);
        }
    }

    // Функтор, вычисляющий матрицу Коссаковски
    auto kossakovski_func = [&l_coeffs, &l_coeffs_conjugate, P](size_t i, size_t j) {
        MKL_Complex16 result = {0., 0.};
        for (int cnt = 0; cnt < P; ++cnt) {
            result += l_coeffs[cnt][i] * l_coeffs_conjugate[cnt][j];
        }
        return result;
    };

    // вычисляем начальное значение вектора v
    double* v = GetVCoef(rho, N);

    auto f_tensor = GenerateTensorF(N);
    auto d_tensor = GenerateTensorD(N);
    auto z_tensor = GenerateTensorZ(f_tensor, d_tensor);

    auto q_matrix = GenerateCOOMatrixQ(&f_tensor, h_coeff, N);

    double* k_vector = GenerateVectorKWithFunctor(kossakovski_func, f_tensor, N);

    double* r_matrix = GenerateMatrixR(l_coeffs, l_coeffs_conjugate, &f_tensor, &z_tensor, N);

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
    // В нашей модели H = const -> Q = const -> зараннее предрасчитываем  S = Q + R и передаем в
    // функцию RK4Step

    // S = Q + R
    // R = 1.0 * Q + R  (то есть R = Q + R)
    for (const auto& [tpl, coeff] : q_matrix) {
        const auto& [s, n] = tpl;
        size_t index = s * M + n;
        r_matrix[index] += coeff;
    }
    double* s_matrix = r_matrix;

    while (t < t_end + h / 2) {
        RK4Step(N, s_matrix, k_vector, h, v, v_next, package);

        // swap rho and rho_next
        std::swap(v, v_next);

        print_vector("v", t, v, M);

        t += h;
    }


    MKL_Complex16* rho_final = GetDensityBySUDecomposition(v, N);
    print_mat(N, rho_final, "rho");

    mkl_free(rho_final);
    mkl_free(hamiltonian);
    for (int i = 0; i < P; ++i) {
        mkl_free(lindbladians[i]);
    }
    mkl_free(rho);
    mkl_free(k_vector);
    mkl_free(r_matrix);

    // Освобождение памяти
    mkl_free(v);
    mkl_free(v_next);

    return 0;
}
