#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>

#include "mkl.h"
#include "generate_matrices.h"
#include "mkl_complex16.h"
#include "matrix_decomposition.h"

#include "lindblad_utils_dimer.h"

#include "timer.h"

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
void CalculateFunc(size_t M, sparse_matrix_t q_matrix, sparse_matrix_t r_matrix, const double* v, const double* k_vector, double* result) {
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
void RK4Step(double t, sparse_matrix_t r_matrix, const double* k_vector, const auto& container, double dt, double* v_in, double* v_out,
             const Package& package) {

    const auto& [h_coef, model, q_builder] = container;
    size_t N = container.model->N;
    size_t M = N * N - 1;

    sparse_matrix_t q_matrix = q_builder->GetMatrix();

    const auto& [k1, k2, k3, k4, v_temp] = package;

    // --- Шаг метода Рунге-Кутты 4-го порядка ---

    // k1 = f(t, v)
    
    model->GetUpdateHCoefDimer(h_coef, t);
    q_builder->UpdateValues(*h_coef);
    CalculateFunc(M, q_matrix, r_matrix, v_in, k_vector, k1);

    // k2 = f(t + h/2, v + h/2 * k1)
    model->GetUpdateHCoefDimer(h_coef, t + 0.5*dt);
    q_builder->UpdateValues(*h_coef);
    AddScaled(M, v_in, k1, dt * 0.5, v_temp);
    CalculateFunc(M, q_matrix, r_matrix, v_temp, k_vector, k2);

    // k3 = f(t + h/2, v + h/2 * k2)
    AddScaled(M, v_in, k2, dt * 0.5, v_temp);
    CalculateFunc(M, q_matrix, r_matrix, v_temp, k_vector, k3);

    // k4 = f(t + h, v + h * k3)
    model->GetUpdateHCoefDimer(h_coef, t + dt);
    q_builder->UpdateValues(*h_coef);
    AddScaled(M, v_in, k3, dt, v_temp);
    CalculateFunc(M, q_matrix, r_matrix, v_temp, k_vector, k4);

    // v_out = v_in + dt/6 * (k1 + 2k2 + 2k3 + k4)
    for (size_t k = 0; k < M; ++k) {
        v_out[k] = v_in[k] + dt / 6 * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
    }
}

template <class Functor>
struct CalculateFunctionContainer {
    std::vector<double>* h_coeff;
    DimerModel<Functor>* model;
    SparseQBuilder* q_builder;
    CalculateFunctionContainer(DimerModel<Functor>* m, SparseQBuilder* q_build) : model(m), q_builder(q_build) {}
};

int main(int argc, char* argv[]) {
    size_t N = 20;
    if (argc == 2) {
        N = std::atoi(argv[1]);
    }

    size_t M = N * N - 1;
    Package package(N);


    auto theta = [T = 2 * std::numbers::pi](double t) {
        t = fmod(t, T);
        if (t < T / 2) {
            return 1.;
        }
        return -1.;
    };
    DimerModel<decltype(theta)> model(theta, N);

    VSLStreamStatePtr stream;
    int seed = 0;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);

    RAMMeter meter;
    Timer timer;

    MKL_Complex16* rho = GenerateDensity(N, stream);
    vslDeleteStream(&stream);

    std::vector<MKL_Complex16> l_coeff = model.GetLCoefDimer();
    std::vector<MKL_Complex16> l_coeff_conjugate(l_coeff);
    for (auto& elem : l_coeff_conjugate) {
        elem = Conjugate(elem);
    }

    auto kossakovski_func = [&l_coeff, &l_coeff_conjugate](size_t i, size_t j) {
        MKL_Complex16 result = {0., 0.};
        result += l_coeff[i] * l_coeff_conjugate[j];
        return result;
    };
    std::vector<double> h_coeff = model.GetBaseHCoefDimer(0.);
    auto f_tensor = GenerateTensorF(N);
    auto d_tensor = GenerateTensorD(N);
    auto z_tensor = GenerateTensorZ(f_tensor, d_tensor);

    SparseQBuilder q_builder(&f_tensor, h_coeff, N);
    SparseRBuilder r_builder(l_coeff, l_coeff_conjugate, &f_tensor, &z_tensor, N);

    CalculateFunctionContainer container(&model, &q_builder);
    container.h_coeff = &h_coeff;

    double* k_vector = GenerateVectorKWithFunctor(kossakovski_func, f_tensor, N);
    sparse_matrix_t r_matrix = r_builder.GetMatrix();

    // Параметры
    double t0 = 0.0;
    double t_end = 1.0;
    double h = 0.0001;

    double eps_gr = 1e-7;

    double t = t0;
    std::cout << std::fixed << std::setprecision(6);
    double* v = GetVCoef(rho, N);
    double* v_next = (double*)mkl_malloc(M * sizeof(double), 64);

    while (t + h < t_end) {
        RK4Step(t, r_matrix, k_vector, container, h, v, v_next, package);

        // swap rho and rho_next
        std::swap(v, v_next);

        //print_vector("v", t, v, M);

        t += h;
    }


    if(t_end - t > eps_gr) {
        RK4Step(t, r_matrix, k_vector, container, t_end - t, v, v_next, package);
        // swap rho and rho_next
        std::swap(v, v_next);
        // print_vector("v", t_end, v, M);
    }


    MKL_Complex16* rho_final = GetDensityBySUDecomposition(v, N);

    timer.stop();
    meter.tick();

    std::cout << rho_final[0].real << "\n";
    mkl_free(rho_final);
    mkl_free(rho);
    mkl_free(k_vector);

    // Освобождение памяти
    mkl_free(v);
    mkl_free(v_next);
    return 0;
}
