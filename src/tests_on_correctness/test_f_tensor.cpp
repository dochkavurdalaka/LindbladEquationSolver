#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>

#include "mkl.h"
#include "mkl_complex16.h"
#include "original.h"
#include "lindblad_utils.h"


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

    std::vector<std::vector<MKL_Complex16*>> commutator(M);
    for (int m = 0; m < M; ++m) {
        commutator[m].resize(M);
        for (int n = 0; n < M; ++n) {
            commutator[m][n] = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            memset(commutator[m][n], 0, N * N * sizeof(MKL_Complex16));
        }
    }

    FillCommutator(&commutator, N);


    std::vector<std::pair<std::tuple<int, int, int>, double>> arr = GenerateTensorF(N);

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
                            commutator[m][n],  // Матрица B
                            N,                 // Ведущий размер для B.
                            &beta,             // Указатель на скаляр beta
                            end_result,        // Матрица C (результат)
                            N                  // Ведущий размер для C.
                );

                double temp = Trace(end_result, N).imag;
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

    std::cout << eps << "\n";


    // Освобождение памяти
    for (auto& el: basis_array) {
        mkl_free(el);
    }
    for (int m = 0; m < N * N - 1; ++m) {
        for (int n = 0; n < N * N - 1; ++n) {
            mkl_free(commutator[m][n]);
        }
    }

    mkl_free(end_result);
}