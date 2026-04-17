#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include "mkl.h"


MKL_Complex16 make_complex(double re, double im) {
    MKL_Complex16 z; 
    z.real = re; 
    z.imag = im; 
    return z;
}


// Генерация случайной комплексной матрицы A (row-major), длина = d*d
void generate_random_complex_matrix_vsl(int d, MKL_Complex16* A, VSLStreamStatePtr stream) {
    int n = d * d;

    // Генерируем нормальные распределения с mean=0, sigma=1
    // Используем метод BOX-MULLER
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 2*n, reinterpret_cast<double*>(A), 0., 1.);
}

void print_matrix_rowmajor(const MKL_Complex16* M, int d, const std::string &name="M") {
    std::cout << name << " (" << d << "x" << d << "):\n";
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            const MKL_Complex16 &c = M[i*d + j];
            std::cout << "(" << std::setw(9) << c.real << "," << std::setw(9) << c.imag << ") ";
        }
        std::cout << "\n";
    }
    std::cout << std::defaultfloat;
}

void print_double_matrix_rowmajor(const double* M, int d, const std::string &name="M") {
    std::cout << name << " (" << d << "x" << d << "):\n";
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            double c = M[i*d + j];
            std::cout << "(" << std::setw(9) << c << "), ";
        }
        std::cout << "\n";
    }
    std::cout << std::defaultfloat;
}


// Проверка приближенной эрмитовости: M == M^† в пределах eps
bool is_hermitian_approx(const MKL_Complex16* M, int d, double eps = 1e-10) {
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            const MKL_Complex16 &a = M[i*d + j];
            const MKL_Complex16 &b = M[j*d + i]; // should be conj(a)
            double re_diff = a.real -  b.real;
            double im_diff = a.imag +  b.imag; // a.imag - (-b.imag) since b should be conj(a)
            if (std::abs(re_diff) > eps || std::abs(im_diff) > eps) return false;
        }
    }
    return true;
}

MKL_Complex16* GenerateDensity(int d, VSLStreamStatePtr stream) {

    MKL_Complex16* A = (MKL_Complex16*)mkl_malloc(d * d * sizeof(MKL_Complex16), 64);
    MKL_Complex16* M = (MKL_Complex16*)mkl_malloc(d * d * sizeof(MKL_Complex16), 64);
    // 1) Генерация A
    generate_random_complex_matrix_vsl(d, A, stream);
    //print_matrix_rowmajor(A.data(), d, "A");

    // 2) Вычислить M = A * A^H
    // cblas_zgemm: C = alpha * A * B + beta * C
    // B = A^H -> используем CblasConjTrans
    MKL_Complex16 alpha = {1.0, 0.0};
    MKL_Complex16 beta  = {0.0, 0.0};
    // Используем CblasRowMajor, A (row-major), B = A^H
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans,
                d,       // M rows of op(A)
                d,       // N cols of op(B)
                d,       // K inner dimension
                &alpha,
                A, d,    // A: pointer, lda = d (row stride)
                A, d,    // B (we pass same pointer, but set ConjTrans)
                &beta,
                M, d);   // C, ldc = d

    // 3) Вычислить след (след матрицы M). Для матрицы AA^H диагональ реально-неотрицательная.
    double trace = 0.0;
    for (int i = 0; i < d; ++i) {
        MKL_Complex16 c = M[i*d + i];
        trace += c.real; // мнимая часть диагонали должна быть ~0
    }

    // if (trace <= 0.0) {
    //     std::cerr << "Trace non-positive or zero (trace = " << trace << "). Aborting.\n";
    //     vslDeleteStream(&stream);
    //     return 1;
    // }

    // std::cout << "Trace(A A^H) = " << trace << "\n";

    // 4) Нормировка: rho = M / trace
    alpha.real = 1./trace;
    alpha.imag = 0.0;
    cblas_zscal(d * d, &alpha, M, 1);


    // 5) Освобождение матрицы A за ненужностью
    mkl_free(A);

    // print_matrix_rowmajor(M, d, "rho (normalized)");

    // 5) Проверка эрмитовости (в пределах погрешности)
    // bool herm = is_hermitian_approx(M, d, 1e-10);
    // std::cout << "Hermitian check: " << (herm ? "OK" : "NOT hermitian (within tol)") << "\n";

    // Доп. проверка: след(rho) == 1
    // double trace_rho = 0.0;
    // for (int i = 0; i < d; ++i) trace_rho += M[i*d + i].real;
    // std::cout << "Trace(rho) = " << trace_rho << "\n";
    return M;
}

// MKL_Complex16* GenerateHamiltonian(int d, int seed) {
//     int n = d * d;

//     MKL_Complex16* H = (MKL_Complex16*)mkl_malloc(n * sizeof(MKL_Complex16), 64);
//     if (H == nullptr) {
//         std::cerr << "Failed to allocate memory in GenerateHamiltonian function\n";
//         return nullptr;
//     }

//     // --- RNG
//     VSLStreamStatePtr stream;
//     if (vslNewStream(&stream, VSL_BRNG_MT19937, seed) != VSL_STATUS_OK) {
//         std::cerr << "Failed to create VSL stream in GenerateHamiltonian function\n";
//         return nullptr;
//     }

//     // генерим случайные нормальные числа
//     vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 2*n, reinterpret_cast<double*>(H), 0., 1.);

//     for (int i = 0; i < d; i++) {
//         // диагональ должна быть вещественной
//         H[i*d + i].imag = 0.0;

//         for (int j = i+1; j < d; j++) {
//             MKL_Complex16 a = H[i*d + j];
//             MKL_Complex16 b = H[j*d + i];

//             // новое значение для верхнего элемента
//             MKL_Complex16 newval;
//             newval.real = 0.5 * (a.real + b.real);
//             newval.imag = 0.5 * (a.imag - b.imag);

//             // записываем симметрично
//             H[i*d + j] = newval;
//             H[j*d + i].real = newval.real;
//             H[j*d + i].imag = -newval.imag;
//         }
//     }


//     // // Проверка: H должно быть эрмитовым
//     // std::cout << "Hermitian matrix H:\n";
//     // for (int i = 0; i < d; i++) {
//     //     for (int j = 0; j < d; j++) {
//     //         std::cout << "(" << H[i*d+j].real << "," << H[i*d+j].imag << ") ";
//     //     }
//     //     std::cout << "\n";
//     // }

//     vslDeleteStream(&stream);
//     return H;
// }

// L_p генерируем рандомно с нулевым следом
MKL_Complex16* GenerateLp(int N, VSLStreamStatePtr stream) {
    MKL_Complex16* L_p = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);

    // генерим случайные нормальные числа
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 2*N*N, reinterpret_cast<double*>(L_p), 0., 1.);

    MKL_Complex16 tr = {0., 0.};
    for (int i = 0; i + 1 < N; i++) {
        tr.real += L_p[i*N + i].real;
        tr.imag += L_p[i*N + i].imag;
    }
    L_p[N * N - 1].real = -tr.real;
    L_p[N * N - 1].imag = -tr.imag;
    // print_matrix_rowmajor(L_p, d, "L_p (normalized)");
    return L_p;
}

MKL_Complex16* GenerateTracelessHamiltonian(int d, VSLStreamStatePtr stream) {
    int n = d * d;

    MKL_Complex16* H = (MKL_Complex16*)mkl_malloc(n * sizeof(MKL_Complex16), 64);

    // генерим случайные нормальные числа
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 2*n, reinterpret_cast<double*>(H), 0., 1.);

    double tr = 0.;
    for (int i = 0; i < d; i++) {
        // диагональ должна быть вещественной
        H[i*d + i].imag = 0.0;
        tr += H[i*d + i].real;

        for (int j = i+1; j < d; j++) {
            MKL_Complex16 a = H[i*d + j];
            MKL_Complex16 b = H[j*d + i];

            // новое значение для верхнего элемента
            MKL_Complex16 newval;
            newval.real = 0.5 * (a.real + b.real);
            newval.imag = 0.5 * (a.imag - b.imag);

            // записываем симметрично
            H[i*d + j] = newval;
            H[j*d + i].real = newval.real;
            H[j*d + i].imag = -newval.imag;
        }
    }

    H[d*d - 1].real -= tr;


    return H;
}

