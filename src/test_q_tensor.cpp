#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <chrono>

#include "mkl.h"
#include "generate_matrices.h"
#include "mkl_complex16.h"
#include "lindblad_utils.h"

using namespace std::chrono;

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

std::vector<MKL_Complex16*> CreateBasisArray(int N) {
    // 2. В цикле выделяем память для каждой матрицы и добавляем указатель в
    // вектор
    std::vector<MKL_Complex16*> basis_array;
    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            MKL_Complex16* new_matrix =
                (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            memset(new_matrix, 0, N * N * sizeof(MKL_Complex16));
            if (new_matrix == NULL) {
                std::cerr << "Ошибка выделения памяти для матрицы";
                // Освобождаем все, что успели выделить
                for (MKL_Complex16* mat : basis_array) {
                    mkl_free(mat);
                }
                // return 1;
            }

            // Инициализация

            int index = j * N + k;
            new_matrix[index] = {1. / sqrt(2), 0.};  // другой способ инициализации
            index = k * N + j;
            new_matrix[index] = {1. / sqrt(2), 0.};  // другой способ инициализации

            // Добавляем указатель на созданную матрицу в наш вектор
            basis_array.push_back(new_matrix);
        }
    }

    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            MKL_Complex16* new_matrix =
                (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            memset(new_matrix, 0, N * N * sizeof(MKL_Complex16));
            if (new_matrix == NULL) {
                std::cerr << "Ошибка выделения памяти для матрицы";
                // Освобождаем все, что успели выделить
                for (MKL_Complex16* mat : basis_array) {
                    mkl_free(mat);
                }
                // return 1;
            }

            // Инициализация

            int index = j * N + k;
            new_matrix[index] = {0., -1. / sqrt(2)};  // другой способ инициализации
            index = k * N + j;
            new_matrix[index] = {0., 1. / sqrt(2)};  // другой способ инициализации

            // Добавляем указатель на созданную матрицу в наш вектор
            basis_array.push_back(new_matrix);
        }
    }

    for (int l = 0; l < N - 1; ++l) {
        MKL_Complex16* new_matrix = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
        memset(new_matrix, 0, N * N * sizeof(MKL_Complex16));
        if (new_matrix == NULL) {
            std::cerr << "Ошибка выделения памяти для матрицы";
            // Освобождаем все, что успели выделить
            for (MKL_Complex16* mat : basis_array) {
                mkl_free(mat);
            }
            // return 1;
        }

        // Инициализация

        for (int k = 0; k < l + 1; ++k) {
            int index = k * N + k;
            new_matrix[index] = {1. / sqrt((l + 1) * (l + 2)), 0.};
        }
        int index = (l + 1) * N + (l + 1);
        new_matrix[index] = {-sqrt(l + 1) / sqrt(l + 2), 0.};
        basis_array.push_back(new_matrix);
    }

    return basis_array;
}

// Реализация нашей векторной функции f(t, v) = (Q(t) + R)v + K
// result = (Q(t) + R)v + K
void calculate_f(double t, const double* v, double* result, const double* Q, const double* R,
                 const double* K, double* workspace_A, int M) {
    double* A = workspace_A;

    // 1. Вычисляем матрицу A(t) = Q(t) + R
    double alpha = 1.0;  // Коэффициент для Q
    double beta = 1.0;   // Коэффициент для R
    mkl_domatadd('R',    // 'R' для RowMajor (C-стиль), 'C' для ColMajor (Fortran-стиль)
                 'N',    // 'N' (No-transpose) для Q
                 'N',    // 'N' (No-transpose) для R
                 M,      // Количество строк
                 M,      // Количество столбцов
                 alpha,  // Скаляр alpha (1.0)
                 Q,      // Матрица Q
                 M,      // Ведущий размер (lda) для Q
                 beta,   // Скаляр beta (1.0)
                 R,      // Матрица R
                 M,      // Ведущий размер (ldb) для R
                 A,      // Результирующая матрица A
                 M       // Ведущий размер (ldc) для A
    );

    // 2. Вычисляем A(t) * v
    // cblas_dgemv: result = alpha*A*v + beta*result
    // Мы хотим result = 1.0 * A * v + 0.0 * result
    beta = 0.0;
    cblas_dgemv(CblasRowMajor, CblasNoTrans, M, M,  // Размеры матрицы A
                alpha,                              // alpha
                A, M,                               // Матрица A и ее lda
                v, 1,                               // Вектор v и его инкремент
                beta,                               // beta
                result, 1);                         // Результирующий вектор и его инкремент

    // 3. Добавляем вектор K
    // cblas_daxpy: result = 1.0 * K + result
    cblas_daxpy(M, alpha, K, 1, result, 1);
}

std::vector<double> GetHCoef(MKL_Complex16* hamiltonian, int N) {
    std::vector<double> h_coeff;

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

double* GetVCoef(const std::vector<MKL_Complex16*>& basis_array, MKL_Complex16* rho, int N) {
    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    MKL_Complex16* result_matrix = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
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

void FillCommutator(std::vector<std::vector<MKL_Complex16*>>* commut, int N) {
    auto& commutator = *commut;
    // 1 квадрант
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = Mapping(i, j_k, N);
                size_t s_c = Mapping(j_k, l, N);
                commutator[f_c][s_c][i * N + l] = {0.5, 0};
                commutator[f_c][s_c][l * N + i] = {-0.5, 0};
            }
        }
    }

    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = Mapping(i_l, j, N);
                size_t s_c = Mapping(k, i_l, N);

                commutator[f_c][s_c][k * N + j] = {-0.5, 0};
                commutator[f_c][s_c][j * N + k] = {0.5, 0};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = Mapping(i_k, j, N);
                    size_t s_c = Mapping(i_k, l, N);

                    commutator[f_c][s_c][j * N + l] = {0.5, 0};
                    commutator[f_c][s_c][l * N + j] = {-0.5, 0};
                }
            }
        }
    }

    for (int j_l = 1; j_l < N; ++j_l) {
        for (int i = 0; i < j_l; ++i) {
            for (int k = 0; k < j_l; ++k) {
                if (i != k) {
                    size_t f_c = Mapping(i, j_l, N);
                    size_t s_c = Mapping(k, j_l, N);

                    commutator[f_c][s_c][i * N + k] = {0.5, 0};
                    commutator[f_c][s_c][k * N + i] = {-0.5, 0};
                }
            }
        }
    }

    // 2 квадрант и 4 квадрант

    // i < l
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = Mapping(i, j_k, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(j_k, l, N);
                commutator[f_c][s_c][i * N + l] = {0, -0.5};
                commutator[f_c][s_c][l * N + i] = {0, -0.5};

                commutator[s_c][f_c][i * N + l] = {0, 0.5};
                commutator[s_c][f_c][l * N + i] = {0, 0.5};
            }
        }
    }

    // k < j
    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = Mapping(i_l, j, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(k, i_l, N);

                commutator[f_c][s_c][k * N + j] = {0, 0.5};
                commutator[f_c][s_c][j * N + k] = {0, 0.5};

                commutator[s_c][f_c][k * N + j] = {0, -0.5};
                commutator[s_c][f_c][j * N + k] = {0, -0.5};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = Mapping(i_k, j, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(i_k, l, N);

                    commutator[f_c][s_c][j * N + l] = {0, -0.5};
                    commutator[f_c][s_c][l * N + j] = {0, -0.5};

                    commutator[s_c][f_c][j * N + l] = {0, 0.5};
                    commutator[s_c][f_c][l * N + j] = {0, 0.5};
                }
            }
        }
    }

    for (int j_l = 1; j_l < N; ++j_l) {
        for (int i = 0; i < j_l; ++i) {
            for (int k = 0; k < j_l; ++k) {
                if (i != k) {
                    size_t f_c = Mapping(i, j_l, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(k, j_l, N);

                    commutator[f_c][s_c][i * N + k] = {0, 0.5};
                    commutator[f_c][s_c][k * N + i] = {0, 0.5};

                    commutator[s_c][f_c][i * N + k] = {0, -0.5};
                    commutator[s_c][f_c][k * N + i] = {0, -0.5};
                }
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j_l = i_k + 1; j_l < N; ++j_l) {
            size_t f_c = Mapping(i_k, j_l, N);
            size_t s_c = N * (N - 1) / 2 + Mapping(i_k, j_l, N);

            commutator[f_c][s_c][i_k * N + i_k] = {0, 1.};
            commutator[f_c][s_c][j_l * N + j_l] = {0, -1.};

            commutator[s_c][f_c][i_k * N + i_k] = {0, -1.};
            commutator[s_c][f_c][j_l * N + j_l] = {0, 1.};
        }
    }

    // 3 квадрант и 7 квадрант
    for (int l = 0; l < N - 1; ++l) {

        // j = l + 1
        for (int i = 0; i < l + 1; ++i) {
            size_t f_c = Mapping(i, l + 1, N);
            size_t s_c = N * (N - 1) + l;
            commutator[f_c][s_c][i * N + l + 1] = {-sqrt(0.5 * (l + 2) / (l + 1)), 0};
            commutator[f_c][s_c][(l + 1) * N + i] = {sqrt(0.5 * (l + 2) / (l + 1)), 0};

            commutator[s_c][f_c][i * N + l + 1] = {sqrt(0.5 * (l + 2) / (l + 1)), 0};
            commutator[s_c][f_c][(l + 1) * N + i] = {-sqrt(0.5 * (l + 2) / (l + 1)), 0};
        }

        for (int i = 0; i < l + 1; ++i) {
            for (int j = l + 2; j < N; ++j) {
                size_t f_c = Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;
                commutator[f_c][s_c][i * N + j] = {-1. / sqrt(2. * (l + 1) * (l + 2)), 0};
                commutator[f_c][s_c][j * N + i] = {1. / sqrt(2. * (l + 1) * (l + 2)), 0};

                commutator[s_c][f_c][i * N + j] = {1. / sqrt(2. * (l + 1) * (l + 2)), 0};
                commutator[s_c][f_c][j * N + i] = {-1. / sqrt(2. * (l + 1) * (l + 2)), 0};
            }
        }

        // i = l + 1
        for (int j = l + 2; j < N; ++j) {
            size_t f_c = Mapping(l + 1, j, N);
            size_t s_c = N * (N - 1) + l;
            commutator[f_c][s_c][(l + 1) * N + j] = {sqrt(0.5 * (l + 1) / (l + 2)), 0};
            commutator[f_c][s_c][j * N + (l + 1)] = {-sqrt(0.5 * (l + 1) / (l + 2)), 0};

            commutator[s_c][f_c][(l + 1) * N + j] = {-sqrt(0.5 * (l + 1) / (l + 2)), 0};
            commutator[s_c][f_c][j * N + (l + 1)] = {sqrt(0.5 * (l + 1) / (l + 2)), 0};
        }
    }

    // 5 квадрант
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j_k, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(j_k, l, N);
                commutator[f_c][s_c][i * N + l] = {-0.5, 0};
                commutator[f_c][s_c][l * N + i] = {0.5, 0};
            }
        }
    }

    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i_l, j, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(k, i_l, N);

                commutator[f_c][s_c][k * N + j] = {0.5, 0};
                commutator[f_c][s_c][j * N + k] = {-0.5, 0};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = N * (N - 1) / 2 + Mapping(i_k, j, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(i_k, l, N);

                    commutator[f_c][s_c][j * N + l] = {0.5, 0};
                    commutator[f_c][s_c][l * N + j] = {-0.5, 0};
                }
            }
        }
    }

    for (int j_l = 1; j_l < N; ++j_l) {
        for (int i = 0; i < j_l; ++i) {
            for (int k = 0; k < j_l; ++k) {
                if (i != k) {
                    size_t f_c = N * (N - 1) / 2 + Mapping(i, j_l, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(k, j_l, N);

                    commutator[f_c][s_c][i * N + k] = {0.5, 0};
                    commutator[f_c][s_c][k * N + i] = {-0.5, 0};
                }
            }
        }
    }

    // 6, 8 квадрант

    for (int l = 0; l < N - 1; ++l) {

        // j = l + 1
        for (int i = 0; i < l + 1; ++i) {
            size_t f_c = N * (N - 1) / 2 + Mapping(i, l + 1, N);
            size_t s_c = N * (N - 1) + l;
            commutator[f_c][s_c][i * N + l + 1] = {0, sqrt(0.5 * (l + 2) / (l + 1))};
            commutator[f_c][s_c][(l + 1) * N + i] = {0, sqrt(0.5 * (l + 2) / (l + 1))};

            commutator[s_c][f_c][i * N + l + 1] = {0, -sqrt(0.5 * (l + 2) / (l + 1))};
            commutator[s_c][f_c][(l + 1) * N + i] = {0, -sqrt(0.5 * (l + 2) / (l + 1))};
        }

        for (int i = 0; i < l + 1; ++i) {
            for (int j = l + 2; j < N; ++j) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;
                commutator[f_c][s_c][i * N + j] = {0, 1. / sqrt(2. * (l + 1) * (l + 2))};
                commutator[f_c][s_c][j * N + i] = {0, 1. / sqrt(2. * (l + 1) * (l + 2))};

                commutator[s_c][f_c][i * N + j] = {0, -1. / sqrt(2. * (l + 1) * (l + 2))};
                commutator[s_c][f_c][j * N + i] = {0, -1. / sqrt(2. * (l + 1) * (l + 2))};
            }
        }

        // i = l + 1
        for (int j = l + 2; j < N; ++j) {
            size_t f_c = N * (N - 1) / 2 + Mapping(l + 1, j, N);
            size_t s_c = N * (N - 1) + l;
            commutator[f_c][s_c][(l + 1) * N + j] = {0, -sqrt(0.5 * (l + 1) / (l + 2))};
            commutator[f_c][s_c][j * N + (l + 1)] = {0, -sqrt(0.5 * (l + 1) / (l + 2))};

            commutator[s_c][f_c][(l + 1) * N + j] = {0, sqrt(0.5 * (l + 1) / (l + 2))};
            commutator[s_c][f_c][j * N + (l + 1)] = {0, sqrt(0.5 * (l + 1) / (l + 2))};
        }
    }
}

int main() {
    // Параметры
    int N = 9;
    int M = N * N - 1;

    std::vector<MKL_Complex16*> basis_array = CreateBasisArray(N);

    MKL_Complex16* end_result = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);

    // Cоздаем гамильтониан
    MKL_Complex16* hamiltonian;
    GenerateTracelessHamiltonian(N, 2, hamiltonian);

    // Cоздаем линдбладиан
    MKL_Complex16* lindbladian;
    GenerateLp(N, 2, lindbladian);

    // Cоздаем матрицу плотности
    MKL_Complex16* rho;
    GenerateDensity(N, 2, rho);

    // Вычисляем коэффициенты h
    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    std::vector<double> h_coeff = GetHCoef(hamiltonian, N);

    // Вычисляем коэффициенты l
    std::vector<MKL_Complex16> l_coeff = GetLCoef(lindbladian, N);

    std::vector<MKL_Complex16> l_coeff_conjugate(l_coeff);
    for (auto& elem : l_coeff_conjugate) {
        elem = Conjugate(elem);
    }

    // Вычисляем матрицу Коссаковски
    std::vector<MKL_Complex16> a(M * M);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            a[i * M + j] = l_coeff[i] * l_coeff_conjugate[j];
        }
    }

    auto f_tensor = GenerateTensorF(N);
    auto d_tensor = GenerateTensorD(N);

    auto z_tensor = GenerateTensorZ(f_tensor, d_tensor);

    auto cmp = [](const std::pair<std::tuple<int, int, int>, double>& left,
                  const std::pair<std::tuple<int, int, int>, double>& right) {
        if (std::get<2>(left.first) == std::get<2>(right.first)) {
            return std::get<1>(left.first) < std::get<1>(right.first);
        }
        return std::get<2>(left.first) < std::get<2>(right.first);
    };

    std::sort(f_tensor.begin(), f_tensor.end(), cmp);


    auto& f_tensor_sorted = f_tensor;
    std::vector<double> values;
    values.reserve(f_tensor_sorted.size());
    std::vector<int> row_ind;
    row_ind.reserve(f_tensor_sorted.size());
    std::vector<int> col_ind;
    col_ind.reserve(f_tensor_sorted.size());

    values.push_back(h_coeff[std::get<0>(f_tensor_sorted[0].first)] * f_tensor_sorted[0].second);
    row_ind.push_back(std::get<2>(f_tensor_sorted[0].first));
    col_ind.push_back(std::get<1>(f_tensor_sorted[0].first));

    for (size_t i = 1; i < f_tensor_sorted.size(); ++i) {
        if (std::get<2>(f_tensor_sorted[i - 1].first) == std::get<2>(f_tensor_sorted[i].first) and
            std::get<1>(f_tensor_sorted[i - 1].first) == std::get<1>(f_tensor_sorted[i].first)) {
            values.back() += h_coeff[std::get<0>(f_tensor_sorted[i].first)] * f_tensor_sorted[i].second;
        } else {
            values.push_back(h_coeff[std::get<0>(f_tensor_sorted[i].first)] * f_tensor_sorted[i].second);
            row_ind.push_back(std::get<2>(f_tensor_sorted[i].first));
            col_ind.push_back(std::get<1>(f_tensor_sorted[i].first));
        }
    }

    std::vector<std::pair<std::tuple<int, int>, double>> q_tensor;
    for(size_t i = 0; i < values.size(); ++i) {
        q_tensor.emplace_back(std::tuple(row_ind[i], col_ind[i]), values[i]);
    }


    sparse_matrix_t A_coo;

    int rows = M;
    int cols = M;
    // int nnz = NNZ;

    mkl_sparse_d_create_coo(&A_coo, SPARSE_INDEX_BASE_ZERO, rows, cols, values.size(),
                            row_ind.data(), col_ind.data(), values.data());

    sparse_matrix_t A_csr;
    mkl_sparse_convert_csr(A_coo, SPARSE_OPERATION_NON_TRANSPOSE, &A_csr);
    mkl_sparse_destroy(A_coo);

    // Спросить у Линева про этот пункт
    mkl_sparse_optimize(A_csr);

    // Общее количество элементов
    size_t total_elements = M * M * M;
    // Выделение памяти для тензора
    double* f_tensor_s = (double*)mkl_malloc(total_elements * sizeof(double), 64);

    std::vector<std::vector<MKL_Complex16*>> commutator(M);
    for (int m = 0; m < M; ++m) {
        commutator[m].resize(M);
        for (int n = 0; n < M; ++n) {
            commutator[m][n] = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            memset(commutator[m][n], 0, N * N * sizeof(MKL_Complex16));
        }
    }
    FillCommutator(&commutator, N);
    std::vector<std::pair<std::tuple<int, int, int>, double>> new_arr;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < M; ++n) {
            for (int s = 0; s < M; ++s) {
                // Вычисляем линейный индекс
                int index = m * M * M + n * M + s;

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
                            end_result,        // Матрица C (результат)
                            N                  // Ведущий размер для C.
                );

                f_tensor_s[index] = Trace(end_result, N).imag;
                if (abs(f_tensor_s[index]) > 1e-12) {
                    new_arr.emplace_back(std::tuple(m, n, s), f_tensor_s[index]);
                }
            }
        }
    }

    (void)new_arr;

    double* q_tensor_s = (double*)mkl_malloc(M * M * sizeof(double), 64);
    memset(q_tensor_s, 0, M * M * sizeof(double));

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < M; ++n) {
            for (int s = 0; s < M; ++s) {
                int q_index = s * M + n;
                int f_index = m * M * M + n * M + s;
                q_tensor_s[q_index] += h_coeff[m] * f_tensor_s[f_index];
            }
        }
    }

    std::vector<std::pair<std::tuple<int, int>, double>> new_q_tensor;
    double eps2 = 0;
    for (int n = 0; n < M; ++n) {
        for (int s = 0; s < M; ++s) {
            int q_index = s * M + n;
            if (abs(q_tensor_s[q_index]) > 1e-12) {
                new_q_tensor.emplace_back(std::tuple(s, n), q_tensor_s[q_index]);
            } else if (abs(q_tensor_s[q_index]) > 0) {
                eps2 = std::max(abs(q_tensor_s[q_index]), 0.);
            }
        }
    }
    std::sort(new_q_tensor.begin(), new_q_tensor.end());

    if (q_tensor.size() != new_q_tensor.size()) {
        std::cout << "false\n";
    }

    double eps = 0;
    for (size_t i = 0; i < q_tensor.size(); ++i) {
        eps = std::max(abs(q_tensor[i].second - new_q_tensor[i].second), eps);
        if ((q_tensor[i].first != new_q_tensor[i].first) and
            abs(q_tensor[i].second - new_q_tensor[i].second) > 1e-12) {
            std::cout << "false\n";
        }
    }
    std::cout << eps << " " << eps2;;
    (void)new_q_tensor;
}