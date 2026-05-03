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
#include "matrix_decomposition.h"

#include "lindblad_utils_dimer.h"

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

// Реализация нашей векторной функции f(t, v) = (Q(t) + R)v + K
// result = (Q(t) + R)v + K
void CalculateFunc(size_t M, const std::vector<std::pair<std::tuple<int, int>, double>>& q_matrix,
                   double* r_matrix, const double* v, const double* k_vector, double* result) {

    // result = K
    cblas_dcopy(M, k_vector, 1, result, 1);

    // 2. Вычисляем Rv + K
    // получаем result += Rv
    cblas_dgemv(CblasRowMajor, CblasNoTrans, M, M,  // Размеры матрицы A
                1.0,                                // alpha
                r_matrix, M,                        // Матрица A и ее lda
                v, 1,                               // Вектор v и его инкремент
                1.0,                                // beta
                result, 1);                         // Результирующий вектор и его инкремент

    // 2. Вычисляем Qv + Rv + K
    // получаем result += Qv
    for (const auto& [tpl, value] : q_matrix) {
        const auto& [s, n] = tpl;
        result[s] += value * v[n];
    }
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
// RK4 integrator step: v_{n+1} = v_n + dt/6*(k1 + 2k2 + 2k3 + k4)
// where k1 = L(v_n), k2 = L(v_n + dt/2*k1), ...
void RK4Step(double t, std::vector<std::pair<std::tuple<int, int, int>, double>>* f_tens,
             MKL_Complex16* hamiltonian, const auto& model, double* r_matrix, double* k_vector,
             double dt, double* v_in, double* v_out, const Package& package) {
    size_t N = model.N;
    size_t M = N * N - 1;

    const auto& [k1, k2, k3, k4, v_temp] = package;

    // --- Шаг метода Рунге-Кутты 4-го порядка ---

    // k1 = f(t, v)
    model.FillHamiltonian(hamiltonian, t);
    std::vector<double> h_coeff = GetHCoef(hamiltonian, N);
    auto q_matrix = GenerateCOOMatrixQ(f_tens, h_coeff, false);
    CalculateFunc(M, q_matrix, r_matrix, v_in, k_vector, k1);

    // k2 = f(t + h/2, v + h/2 * k1)
    model.FillHamiltonian(hamiltonian, t + 0.5 * dt);
    h_coeff = GetHCoef(hamiltonian, N);
    q_matrix = GenerateCOOMatrixQ(f_tens, h_coeff, false);
    AddScaled(M, v_in, k1, dt * 0.5, v_temp);
    CalculateFunc(M, q_matrix, r_matrix, v_temp, k_vector, k2);

    // k3 = f(t + h/2, v + h/2 * k2)
    // q_matrix никак не изменился с прошлого подсчета
    AddScaled(M, v_in, k2, dt * 0.5, v_temp);
    CalculateFunc(M, q_matrix, r_matrix, v_temp, k_vector, k3);

    // k4 = f(t + h, v + h * k3)
    model.FillHamiltonian(hamiltonian, t + dt);
    h_coeff = GetHCoef(hamiltonian, N);
    q_matrix = GenerateCOOMatrixQ(f_tens, h_coeff, false);
    AddScaled(M, v_in, k3, dt, v_temp);
    CalculateFunc(M, q_matrix, r_matrix, v_temp, k_vector, k4);

    // v_out = v_in + dt/6 * (k1 + 2k2 + 2k3 + k4)
    for (size_t k = 0; k < M; ++k) {
        v_out[k] = v_in[k] + dt / 6 * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
    }
}

int main() {
    // Параметры
    int N = 5;
    int M = N * N - 1;

    auto theta = [T = 2 * std::numbers::pi](double t) {
        t = fmod(t, T);
        if (t < T / 2) {
            return 1.;
        }
        return -1.;
    };
    DimerModel<decltype(theta)> model(theta, N);

    // --- RNG
    VSLStreamStatePtr stream;
    int seed = 0;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    MKL_Complex16* rho = GenerateDensity(N, stream);
    vslDeleteStream(&stream);

    MKL_Complex16* hamiltonian = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
    MKL_Complex16* lindbladian = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
    model.FillLindbladian(lindbladian);

    std::vector<MKL_Complex16> l_coeff = GetLCoef(lindbladian, N);
    std::vector<MKL_Complex16> l_coeff_conjugate(l_coeff);
    for (auto& elem : l_coeff_conjugate) {
        elem = Conjugate(elem);
    }

    auto kossakovski_func = [&l_coeff, &l_coeff_conjugate](size_t i, size_t j) {
        MKL_Complex16 result = {0., 0.};
        result += l_coeff[i] * l_coeff_conjugate[j];
        return result;
    };

    auto f_tensor = GenerateTensorF(N);
    auto d_tensor = GenerateTensorD(N);
    auto z_tensor = GenerateTensorZ(f_tensor, d_tensor);

    double t0 = 0.0;

    double* k_vector = GenerateVectorKWithFunctor(kossakovski_func, f_tensor, N);
    double* r_matrix = GenerateMatrixR(l_coeff, l_coeff_conjugate, &f_tensor, &z_tensor, N);

    Package package(N);

    std::cout << std::fixed << std::setprecision(6);

    // Параметры
    double t_end = 1.1;
    double h = 0.01;

    double t = t0;
    double* v = GetVCoef(rho, N);
    double* v_next = (double*)mkl_malloc(M * sizeof(double), 64);

    auto cmp = [](const std::pair<std::tuple<int, int, int>, double>& left,
                  const std::pair<std::tuple<int, int, int>, double>& right) {
        if (std::get<2>(left.first) == std::get<2>(right.first)) {
            return std::get<1>(left.first) < std::get<1>(right.first);
        }
        return std::get<2>(left.first) < std::get<2>(right.first);
    };
    std::sort(f_tensor.begin(), f_tensor.end(), cmp);

    while (t < t_end + h / 2) {
        RK4Step(t, &f_tensor, hamiltonian, model, r_matrix, k_vector, h, v, v_next, package);

        // swap rho and rho_next
        std::swap(v, v_next);

        print_vector("v", t, v, M);

        t += h;
    }

    mkl_free(hamiltonian);
    mkl_free(lindbladian);

    mkl_free(k_vector);
    mkl_free(r_matrix);
    mkl_free(rho);

    // Освобождение памяти
    mkl_free(v);
    mkl_free(v_next);

    return 0;
}
