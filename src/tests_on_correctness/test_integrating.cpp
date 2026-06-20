#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include "mkl_complex16.h"
#include "generate_matrices.h"

#include "mkl_complex16.h"
#include "lindblad_utils.h"
#include "matrix_decomposition.h"

namespace hardcore {
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

// "Быстрая" версия функции, которая раскладывают матрицу плотности по базису SU(N)
void SUBasisDecompose(MKL_Complex16* matrix, size_t N, std::vector<double>* res) {
    auto& result = *res;
    result.clear();
    for (size_t j = 0; j < N; ++j) {
        for (size_t k = j + 1; k < N; ++k) {
            size_t index = j * N + k;
            result.push_back(sqrt(2.) * matrix[index].real);
        }
    }

    for (size_t j = 0; j < N; ++j) {
        for (size_t k = j + 1; k < N; ++k) {
            size_t index = k * N + j;
            result.push_back(sqrt(2.) * matrix[index].imag);
        }
    }

    std::vector<double> matrix_diag_prefixsum(N);
    for (size_t k = 0; k < N; ++k) {
        matrix_diag_prefixsum[k] = matrix[k * N + k].real;
    }
    for (size_t k = 1; k < N; ++k) {
        matrix_diag_prefixsum[k] += matrix_diag_prefixsum[k - 1];
    }

    for (size_t l = 0; l + 1 < N; ++l) {
        double coeff = matrix_diag_prefixsum[l] / sqrt((l + 1) * (l + 2));
        size_t index = (l + 1) * N + (l + 1);
        coeff += -sqrt(l + 1) * matrix[index].real / sqrt(l + 2);

        result.push_back(coeff);
    }
}

// Helper: allocate and check
static void* xmalloc(size_t n) {
    void* p = mkl_malloc(n, 64);
    if (!p) {
        fprintf(stderr, "Allocation failed\n");
        exit(1);
    }
    return p;
}

struct Package {

    Package(int N) {
        auto& [M, T1, T2, T3] = compute_ld;
        M = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16) * N * N);   // Ldag * L
        T1 = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16) * N * N);  // L * rho
        T2 = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16) * N * N);  // M * rho
        T3 = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16) * N * N);  // rho * M

        liouvillian_of_rho.Ld = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16) * N * N);

        auto& [k1, k2, k3, k4, rho_temp] = rk4_step;
        k1 = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16) * N * N);
        k2 = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16) * N * N);
        k3 = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16) * N * N);
        k4 = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16) * N * N);
        rho_temp = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16) * N * N);
    }

    struct ComputeLd {
        MKL_Complex16* M;
        MKL_Complex16* T1;
        MKL_Complex16* T2;
        MKL_Complex16* T3;
    } compute_ld;

    struct LiouvillianOfRho {
        MKL_Complex16* Ld;
    } liouvillian_of_rho;

    struct RK4Step {
        MKL_Complex16* k1;
        MKL_Complex16* k2;
        MKL_Complex16* k3;
        MKL_Complex16* k4;
        MKL_Complex16* rho_temp;
    } rk4_step;

    ~Package() {

        auto& [M, T1, T2, T3] = compute_ld;
        mkl_free(M);
        mkl_free(T1);
        mkl_free(T2);
        mkl_free(T3);

        mkl_free(liouvillian_of_rho.Ld);

        auto& [k1, k2, k3, k4, rho_temp] = rk4_step;
        mkl_free(k1);
        mkl_free(k2);
        mkl_free(k3);
        mkl_free(k4);
        mkl_free(rho_temp);
    }
};

// ----------------------------------------------------------------------
// compute -i [H, rho]  (row-major)
// out must be preallocated N*N
void ComputeCommutator(int N, const MKL_Complex16* H, const MKL_Complex16* rho,
                       MKL_Complex16* out) {

    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    cblas_zgemm(CblasRowMajor,  // Указывает, что матрицы хранятся построчно
                                // (стандарт для C/C++)
                CblasNoTrans,   // Операция для A: не транспонировать
                CblasNoTrans,   // Операция для B: не транспонировать
                N,              // Количество строк в матрице A (и C)
                N,              // Количество столбцов в матрице B (и C)
                N,              // Количество столбцов в A и строк в B
                &alpha,         // Указатель на скаляр alpha
                H,              // Матрица A
                N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                // количество столбцов.
                rho,            // Матрица B
                N,              // Ведущий размер для B.
                &beta,          // Указатель на скаляр beta
                out,            // Матрица C (результат)
                N               // Ведущий размер для C.
    );
    alpha = {0.0, 1.};
    beta = {0.0, -1.};
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
                H,              // Матрица B
                N,              // Ведущий размер для B.
                &beta,          // Указатель на скаляр beta
                out,            // Матрица C (результат)
                N               // Ведущий размер для C.
    );
}

// ----------------------------------------------------------------------
// compute Ld = L rho L^dag - 1/2 (L^dag L rho + rho L^dag L)
// uses temporaries, all row-major
void ComputeLd(int N, const MKL_Complex16* L, const MKL_Complex16* rho, MKL_Complex16* Ld,
               const Package& package) {

    MKL_Complex16 alpha, beta;
    alpha = {1.0, 0.0};
    beta = {0.0, 0.0};
    const auto& [M, T1, T2, T3] = package.compute_ld;

    // T1 = L * rho
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, &alpha, L, N, rho, N, &beta, T1,
                N);
    // Ld = T1 * Ldag = L rho Ldag
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, N, N, N, &alpha, T1, N, L, N, &beta,
                Ld, N);
    // M = Ldag * L
    cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, N, N, N, &alpha, L, N, L, N, &beta, M,
                N);
    // T2 = M * rho
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, &alpha, M, N, rho, N, &beta, T2,
                N);
    // T3 = rho * M
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, &alpha, rho, N, M, N, &beta, T3,
                N);

    // Ld = L rho Ldag - 0.5*(T2 + T3)

    for (size_t i = 0; i < N * N; ++i) {
        Ld[i] -= 0.5 * (T2[i] + T3[i]);
    }

    // alpha = {-0.5, 0.0};
    // cblas_zaxpy(N * N, &alpha, T2, 1, Ld, 1);
    // cblas_zaxpy(N * N, &alpha, T3, 1, Ld, 1);
}

// ----------------------------------------------------------------------
// total Liouvillian L(rho) = -i[H,rho] + Ld
void LiouvillianOfRho(int N, const MKL_Complex16* H,
                      const std::vector<MKL_Complex16*>& lindbladians, const MKL_Complex16* rho,
                      MKL_Complex16* out, const Package& package) {

    const auto& Ld = package.liouvillian_of_rho.Ld;

    ComputeCommutator(N, H, rho, out);

    for (size_t i = 0; i < lindbladians.size(); ++i) {
        ComputeLd(N, lindbladians[i], rho, Ld, package);
        MKL_Complex16 alpha = {1.0, 0.0};
        cblas_zaxpy(N * N, &alpha, Ld, 1, out, 1);
    }
}

// ----------------------------------------------------------------------
// helper: out = a + b * scalar (out = a + coeff * b), length N*N
void AddScaled(int nn, const MKL_Complex16* a, const MKL_Complex16* b, double coeff,
               MKL_Complex16* out) {

    for (int k = 0; k < nn; ++k) {
        out[k] = a[k] + coeff * b[k];
    }

    // cblas_zcopy(nn, a, 1, out, 1);
    // MKL_Complex16 alpha = {coeff, 0.0};
    // cblas_zaxpy(nn, &alpha, b, 1, out, 1);
}

// ----------------------------------------------------------------------
// RK4 integrator step: rho_{n+1} = rho_n + dt/6*(k1 + 2k2 + 2k3 + k4)
// where k1 = L(rho_n), k2 = L(rho_n + dt/2*k1), ...
void RK4Step(int N, const MKL_Complex16* H, const std::vector<MKL_Complex16*>& lindbladians,
             double dt, const MKL_Complex16* rho_in, MKL_Complex16* rho_out,
             const Package& package) {
    int nn = N * N;

    const auto& [k1, k2, k3, k4, rho_temp] = package.rk4_step;

    // k1 = L(rho_in)
    LiouvillianOfRho(N, H, lindbladians, rho_in, k1, package);

    // rho_temp = rho_in + dt/2 * k1
    AddScaled(nn, rho_in, k1, dt * 0.5, rho_temp);
    // k2 = L(rho_in + dt/2 * k1)
    LiouvillianOfRho(N, H, lindbladians, rho_temp, k2, package);

    // rho_temp = rho_in + dt/2 * k2
    AddScaled(nn, rho_in, k2, dt * 0.5, rho_temp);
    // k3 = L(rho_in + dt/2 * k2)
    LiouvillianOfRho(N, H, lindbladians, rho_temp, k3, package);

    // rho_temp = rho_in + dt * k3
    AddScaled(nn, rho_in, k3, dt, rho_temp);
    // k4 = L(rho_in + dt * k3)
    LiouvillianOfRho(N, H, lindbladians, rho_temp, k4, package);

    // rho_out = rho_in + dt/6 * (k1 + 2k2 + 2k3 + k4)
    for (int k = 0; k < nn; ++k) {
        rho_out[k] = rho_in[k] + dt / 6 * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
    }
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

MKL_Complex16* main(int N, int P, double t0, double t_end, double h, double eps_gr) {
    Package package(N);

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

    // std::cout << "\n";

    std::vector<double> v;
    v.resize(N * N - 1);

    MKL_Complex16* rho_next = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16) * N * N);

    double t = t0;
    // std::cout << std::fixed << std::setprecision(6);
    while (t + h < t_end) {

        RK4Step(N, hamiltonian, lindbladians, h, rho, rho_next, package);

        // swap rho and rho_next
        std::swap(rho, rho_next);

        // SUBasisDecompose(rho, N, &v);
        // print_vector("v", t + h, v.data(), N * N - 1);

        t += h;
    }

    if (t_end - t > eps_gr) {
        RK4Step(N, hamiltonian, lindbladians, h, rho, rho_next, package);
        // swap rho and rho_next
        std::swap(rho, rho_next);
        // SUBasisDecompose(rho, N, &v);
        // print_vector("v", t_end, v.data(), N * N - 1);
    }

    // print_mat(N, rho, "rho");

    mkl_free(rho_next);

    mkl_free(hamiltonian);
    for (int i = 0; i < P; ++i) {
        mkl_free(lindbladians[i]);
    }
    return rho;
}
}  // namespace hardcore

namespace main_ns {
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
// RK4 integrator step: v_{n+1} = v_n + dt/6*(k1 + 2k2 + 2k3 + k4)
// where k1 = L(v_n), k2 = L(v_n + dt/2*k1), ...
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

MKL_Complex16* main(int N, int P, double t0, double t_end, double h, double eps_gr) {
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

    auto q_matrix = GenerateCOOMatrixQ(&f_tensor, h_coeff);

    double* k_vector = GenerateVectorKWithFunctor(kossakovski_func, f_tensor, N);

    double* r_matrix = GenerateMatrixR(l_coeffs, l_coeffs_conjugate, &f_tensor, &z_tensor, N);

    Package package(N);
    // Начальные условия

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

    while (t + h < t_end) {
        RK4Step(N, s_matrix, k_vector, h, v, v_next, package);

        // swap rho and rho_next
        std::swap(v, v_next);

        // print_vector("v", t + h, v, M);

        t += h;
    }

    if (t_end - t > eps_gr) {
        RK4Step(N, s_matrix, k_vector, t_end - t, v, v_next, package);
        // swap rho and rho_next
        std::swap(v, v_next);
        // print_vector("v", t_end, v, M);
    }

    MKL_Complex16* rho_final = GetDensityBySUDecomposition(v, N);
    // print_mat(N, rho_final, "rho");

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

    return rho_final;
}

}  // namespace main_ns

double Check(MKL_Complex16* one, MKL_Complex16* two, int N) {
    double result = 0.;
    for (int i = 0; i < N * N; ++i) {
        result = std::max(abs(one[i].real - two[i].real), result);
        result = std::max(abs(one[i].imag - two[i].imag), result);
    }
    return result;
}

int main() {
    int N = 10;

    double t0 = 0.0;
    // Параметры
    double t_end = 1.0;
    double h = 0.01;
    double eps_gr = 1e-7;

    MKL_Complex16* hrd = hardcore::main(N, 1, t0, t_end, h, eps_gr);
    MKL_Complex16* mn = main_ns::main(N, 1, t0, t_end, h, eps_gr);
    double check_result = Check(hrd, mn, N);
    std::cout << std::scientific << check_result << "\n";

    mkl_free(hrd);
    mkl_free(mn);
}