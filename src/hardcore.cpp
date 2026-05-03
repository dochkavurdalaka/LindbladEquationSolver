#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include "mkl_complex16.h"
#include "generate_matrices.h"

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

int main() {
    // Параметры
    int N = 7;
    int P = 1;

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

    std::cout << "\n";

    std::vector<double> v;
    v.resize(N * N - 1);

    MKL_Complex16* rho_next = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16) * N * N);

    // Параметры
    double t_end = 1.3;
    double h = 0.01;
    double eps_gr = 1e-7;

    double t0 = 0.0;

    double t = t0;
    std::cout << std::fixed << std::setprecision(6);
    while (t + h < t_end) {

        RK4Step(N, hamiltonian, lindbladians, h, rho, rho_next, package);

        // swap rho and rho_next
        std::swap(rho, rho_next);

        SUBasisDecompose(rho, N, &v);
        print_vector("v", t + h, v.data(), N * N - 1);

        t += h;
    }

    if(t_end - t > eps_gr) {
        RK4Step(N, hamiltonian, lindbladians, h, rho, rho_next, package);
        // swap rho and rho_next
        std::swap(rho, rho_next);
        SUBasisDecompose(rho, N, &v);
        print_vector("v", t_end, v.data(), N * N - 1);
    }

    print_mat(N, rho, "rho");

    mkl_free(rho_next);

    mkl_free(hamiltonian);
    for (int i = 0; i < P; ++i) {
        mkl_free(lindbladians[i]);
    }
    mkl_free(rho);

    return 0;
}
