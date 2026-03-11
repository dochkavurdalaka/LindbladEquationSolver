
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <utility>
#include <tuple>
#include "mkl.h"
#include "mkl_complex16.h"
#include "generate_matrices.h"

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

MKL_Complex16 Conjugate(MKL_Complex16 number) {
    number.imag = -number.imag;
    return number;
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

size_t Mapping(int i, int j, int N) {
    return i * (N - 1) - (i * (i - 1)) / 2 + j - i - 1;
}

// функция, которая раскладывает (e_{i}e_{i}^T + e_{j}e_{j}^T) по диагональному базису D_l
std::vector<std::pair<int, double>> GetDecomposeD_p(int i, int j, int N) {
    std::vector<std::pair<int, double>> result;

    if (i > 0) {
        int l = i - 1;
        result.emplace_back(l, -sqrt(l + 1) / sqrt(l + 2));
    }

    for (int l = i; l < j - 1; ++l) {
        result.emplace_back(l, 1 / sqrt((l + 1) * (l + 2)));
    }

    if (j != 1) {
        int l = j - 1;
        result.emplace_back(l, -l / sqrt((l + 1) * (l + 2)));
    }

    for (int l = j; l < N - 1; ++l) {
        result.emplace_back(l, 2. / sqrt((l + 1) * (l + 2)));
    }

    return result;
}

bool Check(MKL_Complex16* first, MKL_Complex16* second, int N) {
    for (int i = 0; i < N * N; ++i) {
        if (abs(first[i].real - second[i].real) > 1e-4 or
            abs(first[i].imag - second[i].imag) > 1e-4) {
            return false;
        }
    }
    return true;
}

void AntiCommutator(const MKL_Complex16* F_m, const MKL_Complex16* F_n, MKL_Complex16* dummy_result,
                    int N) {
    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    cblas_zgemm(CblasRowMajor,  // Указывает, что матрицы хранятся построчно
                                // (стандарт для C/C++)
                CblasNoTrans,   // Операция для A: не транспонировать
                CblasNoTrans,   // Операция для B: не транспонировать
                N,              // Количество строк в матрице A (и C)
                N,              // Количество столбцов в матрице B (и C)
                N,              // Количество столбцов в A и строк в B
                &alpha,         // Указатель на скаляр alpha
                F_m,            // Матрица A
                N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                // количество столбцов.
                F_n,            // Матрица B
                N,              // Ведущий размер для B.
                &beta,          // Указатель на скаляр beta
                dummy_result,   // Матрица C (результат)
                N               // Ведущий размер для C.
    );
    beta = {1.0, 0.0};
    cblas_zgemm(CblasRowMajor,  // Указывает, что матрицы хранятся построчно
                                // (стандарт для C/C++)
                CblasNoTrans,   // Операция для A: не транспонировать
                CblasNoTrans,   // Операция для B: не транспонировать
                N,              // Количество строк в матрице A (и C)
                N,              // Количество столбцов в матрице B (и C)
                N,              // Количество столбцов в A и строк в B
                &alpha,         // Указатель на скаляр alpha
                F_n,            // Матрица A
                N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                // количество столбцов.
                F_m,            // Матрица B
                N,              // Ведущий размер для B.
                &beta,          // Указатель на скаляр beta
                dummy_result,   // Матрица C (результат)
                N               // Ведущий размер для C.
    );
}

void FillAntiCommutator(std::vector<std::vector<MKL_Complex16*>>* anticommut, int N) {
    auto& anticommutator = *anticommut;

    // 1 квадрант
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = Mapping(i, j_k, N);
                size_t s_c = Mapping(j_k, l, N);
                anticommutator[f_c][s_c][i * N + l] = {0.5, 0};
                anticommutator[f_c][s_c][l * N + i] = {0.5, 0};
            }
        }
    }

    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = Mapping(i_l, j, N);
                size_t s_c = Mapping(k, i_l, N);

                anticommutator[f_c][s_c][k * N + j] = {0.5, 0};
                anticommutator[f_c][s_c][j * N + k] = {0.5, 0};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = Mapping(i_k, j, N);
                    size_t s_c = Mapping(i_k, l, N);

                    anticommutator[f_c][s_c][j * N + l] = {0.5, 0};
                    anticommutator[f_c][s_c][l * N + j] = {0.5, 0};
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

                    anticommutator[f_c][s_c][i * N + k] = {0.5, 0};
                    anticommutator[f_c][s_c][k * N + i] = {0.5, 0};
                }
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j_l = i_k + 1; j_l < N; ++j_l) {
            size_t f_c = Mapping(i_k, j_l, N);
            size_t s_c = Mapping(i_k, j_l, N);

            anticommutator[f_c][s_c][i_k * N + i_k] = {1, 0};
            anticommutator[f_c][s_c][j_l * N + j_l] = {1, 0};
        }
    }

    // 2 квадрант и 4 квадрант

    // i < l
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = Mapping(i, j_k, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(j_k, l, N);
                anticommutator[f_c][s_c][i * N + l] = {0, -0.5};
                anticommutator[f_c][s_c][l * N + i] = {0, 0.5};

                anticommutator[s_c][f_c][i * N + l] = {0, -0.5};
                anticommutator[s_c][f_c][l * N + i] = {0, 0.5};
            }
        }
    }

    // k < j
    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = Mapping(i_l, j, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(k, i_l, N);

                anticommutator[f_c][s_c][k * N + j] = {0, -0.5};
                anticommutator[f_c][s_c][j * N + k] = {0, 0.5};

                anticommutator[s_c][f_c][k * N + j] = {0, -0.5};
                anticommutator[s_c][f_c][j * N + k] = {0, 0.5};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = Mapping(i_k, j, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(i_k, l, N);

                    anticommutator[f_c][s_c][j * N + l] = {0, -0.5};
                    anticommutator[f_c][s_c][l * N + j] = {0, 0.5};

                    anticommutator[s_c][f_c][j * N + l] = {0, -0.5};
                    anticommutator[s_c][f_c][l * N + j] = {0, 0.5};
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

                    anticommutator[f_c][s_c][i * N + k] = {0, 0.5};
                    anticommutator[f_c][s_c][k * N + i] = {0, -0.5};

                    anticommutator[s_c][f_c][i * N + k] = {0, 0.5};
                    anticommutator[s_c][f_c][k * N + i] = {0, -0.5};
                }
            }
        }
    }

    // 3 квадрант и 7 квадрант
    for (int l = 0; l < N - 1; ++l) {

        // j < l + 1
        for (int j = 1; j < l + 1; ++j) {
            for (int i = 0; i < j; ++i) {
                size_t f_c = Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;
                anticommutator[f_c][s_c][i * N + j] = {sqrt(2) / sqrt((l + 1) * (l + 2)), 0};
                anticommutator[f_c][s_c][j * N + i] = {sqrt(2) / sqrt((l + 1) * (l + 2)), 0};

                anticommutator[s_c][f_c][i * N + j] = {sqrt(2) / sqrt((l + 1) * (l + 2)), 0};
                anticommutator[s_c][f_c][j * N + i] = {sqrt(2) / sqrt((l + 1) * (l + 2)), 0};
            }
        }

        // j = l + 1
        for (int i = 0; i < l + 1; ++i) {
            size_t f_c = Mapping(i, l + 1, N);
            size_t s_c = N * (N - 1) + l;
            anticommutator[f_c][s_c][i * N + l + 1] = {-l / sqrt(2 * (l + 1) * (l + 2)), 0};
            anticommutator[f_c][s_c][(l + 1) * N + i] = {-l / sqrt(2 * (l + 1) * (l + 2)), 0};

            anticommutator[s_c][f_c][i * N + l + 1] = {-l / sqrt(2 * (l + 1) * (l + 2)), 0};
            anticommutator[s_c][f_c][(l + 1) * N + i] = {-l / sqrt(2 * (l + 1) * (l + 2)), 0};
        }

        // i < l + 1 < j
        for (int i = 0; i < l + 1; ++i) {
            for (int j = l + 2; j < N; ++j) {
                size_t f_c = Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;
                anticommutator[f_c][s_c][i * N + j] = {1. / sqrt(2. * (l + 1) * (l + 2)), 0};
                anticommutator[f_c][s_c][j * N + i] = {1. / sqrt(2. * (l + 1) * (l + 2)), 0};

                anticommutator[s_c][f_c][i * N + j] = {1. / sqrt(2. * (l + 1) * (l + 2)), 0};
                anticommutator[s_c][f_c][j * N + i] = {1. / sqrt(2. * (l + 1) * (l + 2)), 0};
            }
        }

        // i = l + 1
        for (int j = l + 2; j < N; ++j) {
            size_t f_c = Mapping(l + 1, j, N);
            size_t s_c = N * (N - 1) + l;
            anticommutator[f_c][s_c][(l + 1) * N + j] = {-sqrt(0.5 * (l + 1) / (l + 2)), 0};
            anticommutator[f_c][s_c][j * N + (l + 1)] = {-sqrt(0.5 * (l + 1) / (l + 2)), 0};

            anticommutator[s_c][f_c][(l + 1) * N + j] = {-sqrt(0.5 * (l + 1) / (l + 2)), 0};
            anticommutator[s_c][f_c][j * N + (l + 1)] = {-sqrt(0.5 * (l + 1) / (l + 2)), 0};
        }
    }

    // 5 квадрант
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j_k, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(j_k, l, N);
                anticommutator[f_c][s_c][i * N + l] = {-0.5, 0};
                anticommutator[f_c][s_c][l * N + i] = {-0.5, 0};
            }
        }
    }

    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i_l, j, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(k, i_l, N);

                anticommutator[f_c][s_c][k * N + j] = {-0.5, 0};
                anticommutator[f_c][s_c][j * N + k] = {-0.5, 0};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = N * (N - 1) / 2 + Mapping(i_k, j, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(i_k, l, N);

                    anticommutator[f_c][s_c][j * N + l] = {0.5, 0};
                    anticommutator[f_c][s_c][l * N + j] = {0.5, 0};
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

                    anticommutator[f_c][s_c][i * N + k] = {0.5, 0};
                    anticommutator[f_c][s_c][k * N + i] = {0.5, 0};
                }
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j_l = i_k + 1; j_l < N; ++j_l) {
            size_t f_c = N * (N - 1) / 2 + Mapping(i_k, j_l, N);
            size_t s_c = N * (N - 1) / 2 + Mapping(i_k, j_l, N);

            anticommutator[f_c][s_c][i_k * N + i_k] = {1, 0};
            anticommutator[f_c][s_c][j_l * N + j_l] = {1, 0};
        }
    }

    // 6, 8 квадрант

    for (int l = 0; l < N - 1; ++l) {

        // j < l + 1
        for (int j = 1; j < l + 1; ++j) {
            for (int i = 0; i < j; ++i) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;
                anticommutator[f_c][s_c][i * N + j] = {0, -sqrt(2) / sqrt((l + 1) * (l + 2))};
                anticommutator[f_c][s_c][j * N + i] = {0, sqrt(2) / sqrt((l + 1) * (l + 2))};

                anticommutator[s_c][f_c][i * N + j] = {0, -sqrt(2) / sqrt((l + 1) * (l + 2))};
                anticommutator[s_c][f_c][j * N + i] = {0, sqrt(2) / sqrt((l + 1) * (l + 2))};
            }
        }

        // j = l + 1
        for (int i = 0; i < l + 1; ++i) {
            size_t f_c = N * (N - 1) / 2 + Mapping(i, l + 1, N);
            size_t s_c = N * (N - 1) + l;
            anticommutator[f_c][s_c][i * N + l + 1] = {0, l / sqrt(2 * (l + 1) * (l + 2))};
            anticommutator[f_c][s_c][(l + 1) * N + i] = {0, -l / sqrt(2 * (l + 1) * (l + 2))};

            anticommutator[s_c][f_c][i * N + l + 1] = {0, l / sqrt(2 * (l + 1) * (l + 2))};
            anticommutator[s_c][f_c][(l + 1) * N + i] = {0, -l / sqrt(2 * (l + 1) * (l + 2))};
        }

        // i < l + 1 < j
        for (int i = 0; i < l + 1; ++i) {
            for (int j = l + 2; j < N; ++j) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;

                anticommutator[f_c][s_c][i * N + j] = {0, -1. / sqrt(2. * (l + 1) * (l + 2))};
                anticommutator[f_c][s_c][j * N + i] = {0, 1. / sqrt(2. * (l + 1) * (l + 2))};

                anticommutator[s_c][f_c][i * N + j] = {0, -1. / sqrt(2. * (l + 1) * (l + 2))};
                anticommutator[s_c][f_c][j * N + i] = {0, 1. / sqrt(2. * (l + 1) * (l + 2))};
            }
        }

        // i = l + 1
        for (int j = l + 2; j < N; ++j) {
            size_t f_c = N * (N - 1) / 2 + Mapping(l + 1, j, N);
            size_t s_c = N * (N - 1) + l;
            anticommutator[f_c][s_c][(l + 1) * N + j] = {0, sqrt(0.5 * (l + 1) / (l + 2))};
            anticommutator[f_c][s_c][j * N + (l + 1)] = {0, -sqrt(0.5 * (l + 1) / (l + 2))};

            anticommutator[s_c][f_c][(l + 1) * N + j] = {0, sqrt(0.5 * (l + 1) / (l + 2))};
            anticommutator[s_c][f_c][j * N + (l + 1)] = {0, -sqrt(0.5 * (l + 1) / (l + 2))};
        }
    }

    // 9 квадрант
    for (int l = 0; l < N - 1; ++l) {
        for (int m = 0; m < N - 1; ++m) {
            size_t f_c = N * (N - 1) + l;
            size_t s_c = N * (N - 1) + m;

            if (l < m) {
                for (int k = 0; k < l + 1; ++k) {
                    int index = k * N + k;
                    anticommutator[f_c][s_c][index] = {
                        2. / (sqrt((m + 1) * (m + 2)) * sqrt((l + 1) * (l + 2))), 0.};
                }
                int index = (l + 1) * N + (l + 1);
                anticommutator[f_c][s_c][index] = {
                    -2. * sqrt(l + 1) / (sqrt((m + 1) * (m + 2)) * sqrt(l + 2)), 0.};

            } else if (l == m) {
                for (int k = 0; k < l + 1; ++k) {
                    int index = k * N + k;
                    anticommutator[f_c][s_c][index] = {2. / ((l + 1) * (l + 2)), 0.};
                }
                int index = (l + 1) * N + (l + 1);
                anticommutator[f_c][s_c][index] = {2. * (l + 1) / (l + 2), 0.};

            } else {
                for (int k = 0; k < m + 1; ++k) {
                    int index = k * N + k;
                    anticommutator[f_c][s_c][index] = {
                        2. / (sqrt((m + 1) * (m + 2)) * sqrt((l + 1) * (l + 2))), 0.};
                }
                int index = (m + 1) * N + (m + 1);
                anticommutator[f_c][s_c][index] = {
                    -2. * sqrt(m + 1) / (sqrt((l + 1) * (l + 2)) * sqrt(m + 2)), 0.};
            }
        }
    }
}

int main() {
    // Параметры
    int N = 9;
    int M = N * N - 1;

    std::vector<MKL_Complex16*> basis_array = CreateBasisArray(N);

    std::vector<std::vector<MKL_Complex16*>> anticommutator(M);
    for (int m = 0; m < M; ++m) {
        anticommutator[m].resize(M);
        for (int n = 0; n < M; ++n) {
            anticommutator[m][n] = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
        }
    }

    std::vector<std::vector<MKL_Complex16*>> new_anticommutator(M);
    for (int m = 0; m < M; ++m) {
        new_anticommutator[m].resize(M);
        for (int n = 0; n < M; ++n) {
            anticommutator[m][n] = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            new_anticommutator[m][n] =
                (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            AntiCommutator(basis_array[m], basis_array[n], new_anticommutator[m][n], N);
        }
    }

    FillAntiCommutator(&anticommutator, N);
    std::vector<std::pair<std::tuple<int, int, int>, double>> arr;

    // 1 квадрант
    // i < l
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = Mapping(i, j_k, N);
                size_t s_c = Mapping(j_k, l, N);
                arr.emplace_back(std::tuple(f_c, s_c, Mapping(i, l, N)), 1 / sqrt(2));
            }
        }
    }

    // k < j
    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = Mapping(i_l, j, N);
                size_t s_c = Mapping(k, i_l, N);

                arr.emplace_back(std::tuple(f_c, s_c, Mapping(k, j, N)), 1 / sqrt(2));
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = Mapping(i_k, j, N);
                    size_t s_c = Mapping(i_k, l, N);

                    if (j < l) {
                        arr.emplace_back(std::tuple(f_c, s_c, Mapping(j, l, N)), 1 / sqrt(2));
                    } else {
                        arr.emplace_back(std::tuple(f_c, s_c, Mapping(l, j, N)), 1 / sqrt(2));
                    }
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

                    if (i < k) {
                        arr.emplace_back(std::tuple(f_c, s_c, Mapping(i, k, N)), 1 / sqrt(2));
                    } else {
                        arr.emplace_back(std::tuple(f_c, s_c, Mapping(k, i, N)), 1 / sqrt(2));
                    }
                }
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j_l = i_k + 1; j_l < N; ++j_l) {
            size_t f_c = Mapping(i_k, j_l, N);
            size_t s_c = Mapping(i_k, j_l, N);

            auto coefficients = GetDecomposeD_p(i_k, j_l, N);
            for (auto& [index, coeff] : coefficients) {
                arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) + index), coeff);
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

                arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(i, l, N)),
                                 1 / sqrt(2));

                arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) / 2 + Mapping(i, l, N)),
                                 1 / sqrt(2));
            }
        }
    }

    // k < j
    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = Mapping(i_l, j, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(k, i_l, N);

                arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(k, j, N)),
                                 1 / sqrt(2));

                arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) / 2 + Mapping(k, j, N)),
                                 1 / sqrt(2));
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = Mapping(i_k, j, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(i_k, l, N);

                    if (j < l) {
                        arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(j, l, N)),
                                         1 / sqrt(2));
                        arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) / 2 + Mapping(j, l, N)),
                                         1 / sqrt(2));
                    } else {
                        arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(l, j, N)),
                                         -1 / sqrt(2));
                        arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) / 2 + Mapping(l, j, N)),
                                         -1 / sqrt(2));
                    }

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

                    if (i < k) {
                        arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(i, k, N)), -1 / sqrt(2));
                        arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) / 2 + Mapping(i, k, N)), -1 / sqrt(2));
                    } else {
                        arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(k, i, N)), 1 / sqrt(2));
                        arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) / 2 + Mapping(k, i, N)), 1 / sqrt(2));
                    }

                }
            }
        }
    }

    // 3 квадрант и 7 квадрант
    for (int l = 0; l < N - 1; ++l) {

        // j < l + 1
        for (int j = 1; j < l + 1; ++j) {
            for (int i = 0; i < j; ++i) {
                size_t f_c = Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;

                arr.emplace_back(std::tuple(f_c, s_c, Mapping(i, j, N)),
                                 2. / sqrt((l + 1) * (l + 2)));

                arr.emplace_back(std::tuple(s_c, f_c, Mapping(i, j, N)),
                                 2. / sqrt((l + 1) * (l + 2)));
            }
        }

        // j = l + 1
        // при l = 0 коэффициент зануляется, поэтому этот случай исключаем
        if (l != 0) {
            for (int i = 0; i < l + 1; ++i) {
                size_t f_c = Mapping(i, l + 1, N);
                size_t s_c = N * (N - 1) + l;

                arr.emplace_back(std::tuple(f_c, s_c, Mapping(i, l + 1, N)),
                                 -l / sqrt((l + 1) * (l + 2)));

                arr.emplace_back(std::tuple(s_c, f_c, Mapping(i, l + 1, N)),
                                 -l / sqrt((l + 1) * (l + 2)));
            }
        }

        // i < l + 1 < j
        for (int i = 0; i < l + 1; ++i) {
            for (int j = l + 2; j < N; ++j) {
                size_t f_c = Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;

                arr.emplace_back(std::tuple(f_c, s_c, Mapping(i, j, N)),
                                 1. / sqrt((l + 1) * (l + 2)));

                arr.emplace_back(std::tuple(s_c, f_c, Mapping(i, j, N)),
                                 1. / sqrt((l + 1) * (l + 2)));
            }
        }

        // i = l + 1
        for (int j = l + 2; j < N; ++j) {
            size_t f_c = Mapping(l + 1, j, N);
            size_t s_c = N * (N - 1) + l;

            arr.emplace_back(std::tuple(f_c, s_c, Mapping(l + 1, j, N)),
                             -sqrt(1. * (l + 1) / (l + 2)));

            arr.emplace_back(std::tuple(s_c, f_c, Mapping(l + 1, j, N)),
                             -sqrt(1. * (l + 1) / (l + 2)));
        }
    }

    // 5 квадрант
    // i < l
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j_k, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(j_k, l, N);

                arr.emplace_back(std::tuple(f_c, s_c, Mapping(i, l, N)), -1 / sqrt(2));
            }
        }
    }

    // k < j
    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i_l, j, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(k, i_l, N);

                arr.emplace_back(std::tuple(f_c, s_c, Mapping(k, j, N)), -1 / sqrt(2));
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = N * (N - 1) / 2 + Mapping(i_k, j, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(i_k, l, N);

                    if (j < l) {
                        arr.emplace_back(std::tuple(f_c, s_c, Mapping(j, l, N)), 1 / sqrt(2));
                    } else {
                        arr.emplace_back(std::tuple(f_c, s_c, Mapping(l, j, N)), 1 / sqrt(2));
                    }
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

                    if (i < k) {
                        arr.emplace_back(std::tuple(f_c, s_c, Mapping(i, k, N)), 1 / sqrt(2));
                    } else {
                        arr.emplace_back(std::tuple(f_c, s_c, Mapping(k, i, N)), 1 / sqrt(2));
                    }
                }
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j_l = i_k + 1; j_l < N; ++j_l) {
            size_t f_c = N * (N - 1) / 2 + Mapping(i_k, j_l, N);
            size_t s_c = N * (N - 1) / 2 + Mapping(i_k, j_l, N);

            auto coefficients = GetDecomposeD_p(i_k, j_l, N);
            for (auto& [index, coeff] : coefficients) {
                arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) + index), coeff);
            }
        }
    }

    // 6, 8 квадрант
    for (int l = 0; l < N - 1; ++l) {

        // j < l + 1
        for (int j = 1; j < l + 1; ++j) {
            for (int i = 0; i < j; ++i) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;

                arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(i, j, N)),
                                 2. / sqrt((l + 1) * (l + 2)));

                arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) / 2 + Mapping(i, j, N)),
                                 2. / sqrt((l + 1) * (l + 2)));
            }
        }

        // j = l + 1
        // при l = 0 коэффициент зануляется, поэтому этот случай исключаем
        if (l != 0) {
            for (int i = 0; i < l + 1; ++i) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, l + 1, N);
                size_t s_c = N * (N - 1) + l;

                arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(i, l + 1, N)),
                                 -l / sqrt((l + 1) * (l + 2)));

                arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) / 2 + Mapping(i, l + 1, N)),
                                 -l / sqrt((l + 1) * (l + 2)));
            }
        }

        // i < l + 1 < j
        for (int i = 0; i < l + 1; ++i) {
            for (int j = l + 2; j < N; ++j) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;

                arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(i, j, N)),
                                 1. / sqrt((l + 1) * (l + 2)));

                arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) / 2 + Mapping(i, j, N)),
                                 1. / sqrt((l + 1) * (l + 2)));
            }
        }

        // i = l + 1
        for (int j = l + 2; j < N; ++j) {
            size_t f_c = N * (N - 1) / 2 + Mapping(l + 1, j, N);
            size_t s_c = N * (N - 1) + l;

            arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(l + 1, j, N)),
                             -sqrt(1. * (l + 1) / (l + 2)));

            arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) / 2 + Mapping(l + 1, j, N)),
                             -sqrt(1. * (l + 1) / (l + 2)));
        }
    }

    // 9 квадрант
    // рассматриваем m < l
    // [l][m] = [m][l] по симметрии антикоммутатора

    for (int m = 0; m + 1 < N - 1; ++m) {
        size_t f_c = N * (N - 1) + m;
        for (int l = m + 1; l < N - 1; ++l) {
            size_t s_c = N * (N - 1) + l;
            arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) + m), 2. / sqrt((l + 1) * (l + 2)));
            arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) + m), 2. / sqrt((l + 1) * (l + 2)));
        }
    }

    // рассматриваем l = m
    for (int l_m = 0; l_m < N - 1; ++l_m) {
        size_t f_c = N * (N - 1) + l_m;
        if (l_m != 0) {
            arr.emplace_back(std::tuple(f_c, f_c, N * (N - 1) + l_m),
                             -2. * l_m / sqrt((l_m + 1) * (l_m + 2)));
        }
        for (int s = l_m + 1; s < N - 1; ++s) {
            arr.emplace_back(std::tuple(f_c, f_c, N * (N - 1) + s), 2. / sqrt((s + 1) * (s + 2)));
        }
    }

    std::sort(arr.begin(), arr.end());
    (void)arr;

    std::vector<std::pair<std::tuple<int, int, int>, double>> new_arr;

    MKL_Complex16* end_result = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < M; ++n) {
            for (int s = 0; s < M; ++s) {
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
                            new_anticommutator[m][n],  // Матрица B
                            N,                     // Ведущий размер для B.
                            &beta,                 // Указатель на скаляр beta
                            end_result,            // Матрица C (результат)
                            N                      // Ведущий размер для C.
                );

                double temp = Trace(end_result, N).real;
                if (abs(temp) > 1e-12) {
                    new_arr.emplace_back(std::tuple(m, n, s), temp);
                }
            }
        }
    }

    std::sort(new_arr.begin(), new_arr.end());

    if (arr.size() != new_arr.size()) {
        std::cout << "false\n";
    }

    double eps = 0;
    for (size_t i = 0; i < arr.size(); ++i) {
        eps = std::max(abs(arr[i].second - new_arr[i].second), eps);
        if ((arr[i].first != new_arr[i].first) or abs(arr[i].second - new_arr[i].second) > 1e-12) {
            std::cout << "false\n";
        }
    }
    std::cout << eps;

    // print_matrix_rowmajor(commutator[2][13], N, "one");
    // print_matrix_rowmajor(new_commutator[2][13], N, "two");
}