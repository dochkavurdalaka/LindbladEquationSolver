#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>

#include "mkl.h"
#include "original.h"

size_t Mapping(int i, int j, int N) {
    return i * (N - 1) - (i * (i - 1)) / 2 + j - i - 1;
}

bool Check(MKL_Complex16* first, MKL_Complex16* second, int N) {
    for (int i = 0; i < N * N; ++i) {
        if (abs(first[i].real - second[i].real) > 1e-12 or
            abs(first[i].imag - second[i].imag) > 1e-12) {
            return false;
        }
    }
    return true;
}

int main() {
    // Параметры
    int N = 9;
    int M = N * N - 1;

    std::vector<MKL_Complex16*> basis_array = CreateBasisArray(N);

    std::vector<std::vector<MKL_Complex16*>> commutator(M);
    std::vector<std::vector<MKL_Complex16*>> new_commutator(M);
    for (int m = 0; m < M; ++m) {
        commutator[m].resize(M);
        new_commutator[m].resize(M);
        for (int n = 0; n < M; ++n) {
            commutator[m][n] = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            new_commutator[m][n] = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            memset(new_commutator[m][n], 0, N * N * sizeof(MKL_Complex16));
            Commutator(basis_array[m], basis_array[n], commutator[m][n], N);
        }
    }

    // 1 квадрант
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = Mapping(i, j_k, N);
                size_t s_c = Mapping(j_k, l, N);
                new_commutator[f_c][s_c][i * N + l] = {0.5, 0};
                new_commutator[f_c][s_c][l * N + i] = {-0.5, 0};
            }
        }
    }

    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = Mapping(i_l, j, N);
                size_t s_c = Mapping(k, i_l, N);

                new_commutator[f_c][s_c][k * N + j] = {-0.5, 0};
                new_commutator[f_c][s_c][j * N + k] = {0.5, 0};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = Mapping(i_k, j, N);
                    size_t s_c = Mapping(i_k, l, N);

                    new_commutator[f_c][s_c][j * N + l] = {0.5, 0};
                    new_commutator[f_c][s_c][l * N + j] = {-0.5, 0};
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

                    new_commutator[f_c][s_c][i * N + k] = {0.5, 0};
                    new_commutator[f_c][s_c][k * N + i] = {-0.5, 0};
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
                new_commutator[f_c][s_c][i * N + l] = {0, -0.5};
                new_commutator[f_c][s_c][l * N + i] = {0, -0.5};

                new_commutator[s_c][f_c][i * N + l] = {0, 0.5};
                new_commutator[s_c][f_c][l * N + i] = {0, 0.5};
            }
        }
    }

    // k < j
    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = Mapping(i_l, j, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(k, i_l, N);

                new_commutator[f_c][s_c][k * N + j] = {0, 0.5};
                new_commutator[f_c][s_c][j * N + k] = {0, 0.5};

                new_commutator[s_c][f_c][k * N + j] = {0, -0.5};
                new_commutator[s_c][f_c][j * N + k] = {0, -0.5};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = Mapping(i_k, j, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(i_k, l, N);

                    new_commutator[f_c][s_c][j * N + l] = {0, -0.5};
                    new_commutator[f_c][s_c][l * N + j] = {0, -0.5};

                    new_commutator[s_c][f_c][j * N + l] = {0, 0.5};
                    new_commutator[s_c][f_c][l * N + j] = {0, 0.5};
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

                    new_commutator[f_c][s_c][i * N + k] = {0, 0.5};
                    new_commutator[f_c][s_c][k * N + i] = {0, 0.5};

                    new_commutator[s_c][f_c][i * N + k] = {0, -0.5};
                    new_commutator[s_c][f_c][k * N + i] = {0, -0.5};
                }
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j_l = i_k + 1; j_l < N; ++j_l) {
            size_t f_c = Mapping(i_k, j_l, N);
            size_t s_c = N * (N - 1) / 2 + Mapping(i_k, j_l, N);

            new_commutator[f_c][s_c][i_k * N + i_k] = {0, 1.};
            new_commutator[f_c][s_c][j_l * N + j_l] = {0, -1.};

            new_commutator[s_c][f_c][i_k * N + i_k] = {0, -1.};
            new_commutator[s_c][f_c][j_l * N + j_l] = {0, 1.};
        }
    }

    // 3 квадрант и 7 квадрант
    for (int l = 0; l < N - 1; ++l) {

        // j = l + 1
        for (int i = 0; i < l + 1; ++i) {
            size_t f_c = Mapping(i, l + 1, N);
            size_t s_c = N * (N - 1) + l;
            new_commutator[f_c][s_c][i * N + l + 1] = {-sqrt(0.5 * (l + 2) / (l + 1)), 0};
            new_commutator[f_c][s_c][(l + 1) * N + i] = {sqrt(0.5 * (l + 2) / (l + 1)), 0};

            new_commutator[s_c][f_c][i * N + l + 1] = {sqrt(0.5 * (l + 2) / (l + 1)), 0};
            new_commutator[s_c][f_c][(l + 1) * N + i] = {-sqrt(0.5 * (l + 2) / (l + 1)), 0};
        }

        for (int i = 0; i < l + 1; ++i) {
            for (int j = l + 2; j < N; ++j) {
                size_t f_c = Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;
                new_commutator[f_c][s_c][i * N + j] = {-1. / sqrt(2. * (l + 1) * (l + 2)), 0};
                new_commutator[f_c][s_c][j * N + i] = {1. / sqrt(2. * (l + 1) * (l + 2)), 0};

                new_commutator[s_c][f_c][i * N + j] = {1. / sqrt(2. * (l + 1) * (l + 2)), 0};
                new_commutator[s_c][f_c][j * N + i] = {-1. / sqrt(2. * (l + 1) * (l + 2)), 0};
            }
        }

        // i = l + 1
        for (int j = l + 2; j < N; ++j) {
            size_t f_c = Mapping(l + 1, j, N);
            size_t s_c = N * (N - 1) + l;
            new_commutator[f_c][s_c][(l + 1) * N + j] = {sqrt(0.5 * (l + 1) / (l + 2)), 0};
            new_commutator[f_c][s_c][j * N + (l + 1)] = {-sqrt(0.5 * (l + 1) / (l + 2)), 0};

            new_commutator[s_c][f_c][(l + 1) * N + j] = {-sqrt(0.5 * (l + 1) / (l + 2)), 0};
            new_commutator[s_c][f_c][j * N + (l + 1)] = {sqrt(0.5 * (l + 1) / (l + 2)), 0};
        }
    }

    // 5 квадрант
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j_k, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(j_k, l, N);
                new_commutator[f_c][s_c][i * N + l] = {-0.5, 0};
                new_commutator[f_c][s_c][l * N + i] = {0.5, 0};
            }
        }
    }

    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i_l, j, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(k, i_l, N);

                new_commutator[f_c][s_c][k * N + j] = {0.5, 0};
                new_commutator[f_c][s_c][j * N + k] = {-0.5, 0};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = N * (N - 1) / 2 + Mapping(i_k, j, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(i_k, l, N);

                    new_commutator[f_c][s_c][j * N + l] = {0.5, 0};
                    new_commutator[f_c][s_c][l * N + j] = {-0.5, 0};
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

                    new_commutator[f_c][s_c][i * N + k] = {0.5, 0};
                    new_commutator[f_c][s_c][k * N + i] = {-0.5, 0};
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
            new_commutator[f_c][s_c][i * N + l + 1] = {0, sqrt(0.5 * (l + 2) / (l + 1))};
            new_commutator[f_c][s_c][(l + 1) * N + i] = {0, sqrt(0.5 * (l + 2) / (l + 1))};

            new_commutator[s_c][f_c][i * N + l + 1] = {0, -sqrt(0.5 * (l + 2) / (l + 1))};
            new_commutator[s_c][f_c][(l + 1) * N + i] = {0, -sqrt(0.5 * (l + 2) / (l + 1))};
        }

        for (int i = 0; i < l + 1; ++i) {
            for (int j = l + 2; j < N; ++j) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;
                new_commutator[f_c][s_c][i * N + j] = {0, 1. / sqrt(2. * (l + 1) * (l + 2))};
                new_commutator[f_c][s_c][j * N + i] = {0, 1. / sqrt(2. * (l + 1) * (l + 2))};

                new_commutator[s_c][f_c][i * N + j] = {0, -1. / sqrt(2. * (l + 1) * (l + 2))};
                new_commutator[s_c][f_c][j * N + i] = {0, -1. / sqrt(2. * (l + 1) * (l + 2))};
            }
        }

        // i = l + 1
        for (int j = l + 2; j < N; ++j) {
            size_t f_c = N * (N - 1) / 2 + Mapping(l + 1, j, N);
            size_t s_c = N * (N - 1) + l;
            new_commutator[f_c][s_c][(l + 1) * N + j] = {0, -sqrt(0.5 * (l + 1) / (l + 2))};
            new_commutator[f_c][s_c][j * N + (l + 1)] = {0, -sqrt(0.5 * (l + 1) / (l + 2))};

            new_commutator[s_c][f_c][(l + 1) * N + j] = {0, sqrt(0.5 * (l + 1) / (l + 2))};
            new_commutator[s_c][f_c][j * N + (l + 1)] = {0, sqrt(0.5 * (l + 1) / (l + 2))};
        }
    }

    // если вывод пустой, значит все корректно работает
    for (int m = 0; m < N * N - 1; ++m) {
        for (int n = 0; n < N * N - 1; ++n) {
            if (not Check(commutator[m][n], new_commutator[m][n], N)) {
                std::cout << m << " " << n << "\n";
            }
        }
    }

    // освобождение памяти
    for (auto& el: basis_array) {
        mkl_free(el);
    }
    for (int m = 0; m < N * N - 1; ++m) {
        for (int n = 0; n < N * N - 1; ++n) {
            mkl_free(commutator[m][n]);
            mkl_free(new_commutator[m][n]);
        }
    }


}