#pragma once

#include <algorithm>
#include <cmath>
#include <array>
#include <cstring>
#include <vector>
#include "mkl.h"
#include "mkl_complex16.h"

size_t Mapping(int i, int j, int N) {
    return i * (N - 1) - (i * (i - 1)) / 2 + j - i - 1;
}

// функция, которая раскладывает (e_{i}e_{i}^T - e_{j}e_{j}^T) по диагональному базису D_l
std::vector<std::pair<int, double>> GetDecomposeD_m(int i, int j, int) {
    std::vector<std::pair<int, double>> result;

    if (i > 0) {
        int l = i - 1;
        result.emplace_back(l, -sqrt(l + 1) / sqrt(l + 2));
    }

    for (int l = i; l < j - 1; ++l) {
        result.emplace_back(l, 1 / sqrt((l + 1) * (l + 2)));
    }

    int l = j - 1;
    result.emplace_back(l, sqrt(l + 2) / sqrt(l + 1));

    return result;
}

std::vector<std::pair<std::tuple<int, int, int>, double>> GenerateTensorF(int N) {
    // 1 квадрант
    // i < l
    std::vector<std::pair<std::tuple<int, int, int>, double>> arr;

    // (N³ - 3N² + 2N) / 6 = C(N, 3) - столько здесь добавляется элементов
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = Mapping(i, j_k, N);
                size_t s_c = Mapping(j_k, l, N);

                arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(i, l, N)),
                                 1 / sqrt(2));
            }
        }
    }

    // k < j
    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = Mapping(i_l, j, N);
                size_t s_c = Mapping(k, i_l, N);

                arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(k, j, N)),
                                 -1 / sqrt(2));
            }
        }
    }

    // (N³ - 3N² + 2N) / 3 = 2 * C(N, 3) - столько здесь добавляется элементов
    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = Mapping(i_k, j, N);
                    size_t s_c = Mapping(i_k, l, N);

                    if (j < l) {
                        arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(j, l, N)),
                                         1 / sqrt(2));
                    } else {
                        arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(l, j, N)),
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
                    size_t s_c = Mapping(k, j_l, N);

                    if (i < k) {
                        arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(i, k, N)),
                                         1 / sqrt(2));
                    } else {
                        arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(k, i, N)),
                                         -1 / sqrt(2));
                    }
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

                arr.emplace_back(std::tuple(f_c, s_c, Mapping(i, l, N)), -1 / sqrt(2));

                arr.emplace_back(std::tuple(s_c, f_c, Mapping(i, l, N)), 1 / sqrt(2));
            }
        }
    }

    // k < j
    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = Mapping(i_l, j, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(k, i_l, N);

                arr.emplace_back(std::tuple(f_c, s_c, Mapping(k, j, N)), 1 / sqrt(2));

                arr.emplace_back(std::tuple(s_c, f_c, Mapping(k, j, N)), -1 / sqrt(2));
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
                        arr.emplace_back(std::tuple(f_c, s_c, Mapping(j, l, N)), -1 / sqrt(2));
                        arr.emplace_back(std::tuple(s_c, f_c, Mapping(j, l, N)), 1 / sqrt(2));
                    } else {
                        arr.emplace_back(std::tuple(f_c, s_c, Mapping(l, j, N)), -1 / sqrt(2));
                        arr.emplace_back(std::tuple(s_c, f_c, Mapping(l, j, N)), 1 / sqrt(2));
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
                        arr.emplace_back(std::tuple(f_c, s_c, Mapping(i, k, N)), 1 / sqrt(2));
                        arr.emplace_back(std::tuple(s_c, f_c, Mapping(i, k, N)), -1 / sqrt(2));
                    } else {
                        arr.emplace_back(std::tuple(f_c, s_c, Mapping(k, i, N)), 1 / sqrt(2));
                        arr.emplace_back(std::tuple(s_c, f_c, Mapping(k, i, N)), -1 / sqrt(2));
                    }
                }
            }
        }
    }

    //  2 * ((N³ + 3N² - 10N + 6) / 6) - столько здесь добавляется элементов
    // слева домножается на 2, потому что здесь обрабатываются две диагонали
    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j_l = i_k + 1; j_l < N; ++j_l) {
            size_t f_c = Mapping(i_k, j_l, N);
            size_t s_c = N * (N - 1) / 2 + Mapping(i_k, j_l, N);

            auto coefficients = GetDecomposeD_m(i_k, j_l, N);
            for (auto& [index, coeff] : coefficients) {
                arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) + index), coeff);
            }

            for (auto& [index, coeff] : coefficients) {
                arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) + index), -coeff);
            }
        }
    }

    // 3 квадрант и 7 квадрант
    for (int l = 0; l < N - 1; ++l) {

        // j = l + 1
        for (int i = 0; i < l + 1; ++i) {
            size_t f_c = Mapping(i, l + 1, N);
            size_t s_c = N * (N - 1) + l;

            arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(i, l + 1, N)),
                             -sqrt(1. * (l + 2) / (l + 1)));

            arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) / 2 + Mapping(i, l + 1, N)),
                             sqrt(1. * (l + 2) / (l + 1)));
        }

        for (int i = 0; i < l + 1; ++i) {
            for (int j = l + 2; j < N; ++j) {
                size_t f_c = Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;

                arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(i, j, N)),
                                 -1. / sqrt((l + 1) * (l + 2)));

                arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) / 2 + Mapping(i, j, N)),
                                 1. / sqrt((l + 1) * (l + 2)));
            }
        }

        // i = l + 1
        for (int j = l + 2; j < N; ++j) {
            size_t f_c = Mapping(l + 1, j, N);
            size_t s_c = N * (N - 1) + l;

            arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(l + 1, j, N)),
                             sqrt(1. * (l + 1) / (l + 2)));

            arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) / 2 + Mapping(l + 1, j, N)),
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

                arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(i, l, N)),
                                 -1 / sqrt(2));
            }
        }
    }

    // k < j
    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i_l, j, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(k, i_l, N);

                arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(k, j, N)),
                                 1 / sqrt(2));
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
                        arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(j, l, N)),
                                         1 / sqrt(2));
                    } else {
                        arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(l, j, N)),
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
                    size_t f_c = N * (N - 1) / 2 + Mapping(i, j_l, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(k, j_l, N);

                    if (i < k) {
                        arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(i, k, N)),
                                         1 / sqrt(2));
                    } else {
                        arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(k, i, N)),
                                         -1 / sqrt(2));
                    }
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

            arr.emplace_back(std::tuple(f_c, s_c, Mapping(i, l + 1, N)),
                             sqrt(1. * (l + 2) / (l + 1)));

            arr.emplace_back(std::tuple(s_c, f_c, Mapping(i, l + 1, N)),
                             -sqrt(1. * (l + 2) / (l + 1)));
        }

        for (int i = 0; i < l + 1; ++i) {
            for (int j = l + 2; j < N; ++j) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;

                arr.emplace_back(std::tuple(f_c, s_c, Mapping(i, j, N)),
                                 1. / sqrt((l + 1) * (l + 2)));

                arr.emplace_back(std::tuple(s_c, f_c, Mapping(i, j, N)),
                                 -1. / sqrt((l + 1) * (l + 2)));
            }
        }

        // i = l + 1
        for (int j = l + 2; j < N; ++j) {
            size_t f_c = N * (N - 1) / 2 + Mapping(l + 1, j, N);
            size_t s_c = N * (N - 1) + l;

            arr.emplace_back(std::tuple(f_c, s_c, Mapping(l + 1, j, N)),
                             -sqrt(1. * (l + 1) / (l + 2)));

            arr.emplace_back(std::tuple(s_c, f_c, Mapping(l + 1, j, N)),
                             sqrt(1. * (l + 1) / (l + 2)));
        }
    }

    return arr;
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

std::vector<std::pair<std::tuple<int, int, int>, double>> GenerateTensorD(int N) {
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
                        arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(i, k, N)),
                                         -1 / sqrt(2));
                        arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) / 2 + Mapping(i, k, N)),
                                         -1 / sqrt(2));
                    } else {
                        arr.emplace_back(std::tuple(f_c, s_c, N * (N - 1) / 2 + Mapping(k, i, N)),
                                         1 / sqrt(2));
                        arr.emplace_back(std::tuple(s_c, f_c, N * (N - 1) / 2 + Mapping(k, i, N)),
                                         1 / sqrt(2));
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

    return arr;
}

// принимает на вход две отсортированные последовательности f и d и merge их
std::vector<std::pair<std::tuple<int, int, int>, MKL_Complex16>> GenerateTensorZ(
    const std::vector<std::pair<std::tuple<int, int, int>, double>>& f_tensor,
    const std::vector<std::pair<std::tuple<int, int, int>, double>>& d_tensor) {
    std::vector<std::pair<std::tuple<int, int, int>, MKL_Complex16>> z_tensor;
    z_tensor.reserve(f_tensor.size() + d_tensor.size());

    for (size_t i = 0; i < f_tensor.size(); ++i) {
        z_tensor.emplace_back(f_tensor[i].first, MKL_Complex16{f_tensor[i].second, 0});
    }

    for (size_t i = 0; i < d_tensor.size(); ++i) {
        z_tensor.emplace_back(d_tensor[i].first, MKL_Complex16{0, d_tensor[i].second});
    }

    return z_tensor;
}

// пердполагается, что в эту функцию передается уже отсортированный тензор f
// sparse_matrix_t GenerateSparseMatrixQ(
//     const std::vector<std::pair<std::tuple<int, int, int>, double>>& f_tensor_sorted,
//     const std::vector<double>& h_coeff) {

//     std::vector<double> values;
//     values.reserve(f_tensor_sorted.size());
//     std::vector<int> row_ind;
//     row_ind.reserve(f_tensor_sorted.size());
//     std::vector<int> col_ind;
//     col_ind.reserve(f_tensor_sorted.size());

//     values.push_back(h_coeff[std::get<0>(f_tensor_sorted[0].first)] * f_tensor_sorted[0].second);
//     row_ind.push_back(std::get<2>(f_tensor_sorted[0].first));
//     col_ind.push_back(std::get<1>(f_tensor_sorted[0].first));

//     for (size_t i = 1; i < f_tensor_sorted.size(); ++i) {
//         if (std::get<2>(f_tensor_sorted[i - 1].first) == std::get<2>(f_tensor_sorted[i].first)
//         and
//             std::get<1>(f_tensor_sorted[i - 1].first) == std::get<1>(f_tensor_sorted[i].first)) {
//             values.back() +=
//                 h_coeff[std::get<0>(f_tensor_sorted[i].first)] * f_tensor_sorted[i].second;
//         } else {
//             values.push_back(h_coeff[std::get<0>(f_tensor_sorted[i].first)] *
//                              f_tensor_sorted[i].second);
//             row_ind.push_back(std::get<2>(f_tensor_sorted[i].first));
//             col_ind.push_back(std::get<1>(f_tensor_sorted[i].first));
//         }
//     }

//     sparse_matrix_t q_coo;

//     int rows = static_cast<int>(h_coeff.size());
//     int cols = static_cast<int>(h_coeff.size());

//     mkl_sparse_d_create_coo(&q_coo, SPARSE_INDEX_BASE_ZERO, rows, cols, values.size(),
//                             row_ind.data(), col_ind.data(), values.data());

//     sparse_matrix_t q_csr;
//     mkl_sparse_convert_csr(q_coo, SPARSE_OPERATION_NON_TRANSPOSE, &q_csr);
//     mkl_sparse_destroy(q_coo);

//     // Спросить у Линева про этот пункт
//     mkl_sparse_optimize(q_csr);
//     return q_csr;
// }

double* GenerateVectorK(const std::vector<MKL_Complex16>& a,
                        const std::vector<std::pair<std::tuple<int, int, int>, double>>& f_tensor,
                        int N) {
    int M = N * N - 1;
    double* k_vector = (double*)mkl_malloc(M * sizeof(double), 64);
    memset(k_vector, 0, M * sizeof(double));
    for (const auto& [ind, value] : f_tensor) {
        const auto& [m, n, s] = ind;
        k_vector[s] += -1. * (a[m * M + n] * value).imag / N;
    }

    return k_vector;
}

double* GenerateVectorKWithFunctor(
    auto kossakovski_func,
    const std::vector<std::pair<std::tuple<int, int, int>, double>>& f_tensor, int N) {
    int M = N * N - 1;
    double* k_vector = (double*)mkl_malloc(M * sizeof(double), 64);
    memset(k_vector, 0, M * sizeof(double));

    for (const auto& [ind, value] : f_tensor) {
        const auto& [m, n, s] = ind;
        k_vector[s] += -1. * (kossakovski_func(m, n) * value).imag;
    }

    for (int s = 0; s < M; ++s) {
        k_vector[s] /= N;
    }

    return k_vector;
}

// в этой функции ключевым условием для правильности работы является отсортированность тензора f по
// возрастанию (s, n)
std::vector<std::pair<std::tuple<int, int>, double>> GenerateCOOMatrixQ(
    std::vector<std::pair<std::tuple<int, int, int>, double>>* f_tens,
    const std::vector<double>& h_coeff, bool need_sort = true) {
    auto& f_tensor = *f_tens;

    if (need_sort) {
        auto cmp = [](const std::pair<std::tuple<int, int, int>, double>& left,
                      const std::pair<std::tuple<int, int, int>, double>& right) {
            if (std::get<2>(left.first) == std::get<2>(right.first)) {
                return std::get<1>(left.first) < std::get<1>(right.first);
            }
            return std::get<2>(left.first) < std::get<2>(right.first);
        };
        std::sort(f_tensor.begin(), f_tensor.end(), cmp);
    }

    std::vector<std::pair<std::tuple<int, int>, double>> q_matrix;

    int cur_n, cur_s;
    cur_n = cur_s = -1;

    for (const auto& [f_ind, value] : f_tensor) {
        const auto& [m, n, s] = f_ind;
        if (cur_n == n and cur_s == s) {
            q_matrix.back().second += h_coeff[m] * value;
        } else {
            q_matrix.emplace_back(std::tuple(s, n), h_coeff[m] * value);
            cur_n = n;
            cur_s = s;
        }
    }

    return q_matrix;
}

double* GenerateMatrixR(const std::vector<MKL_Complex16>& l_coeff,
                        const std::vector<MKL_Complex16>& l_coeff_conjugate,
                        std::vector<std::pair<std::tuple<int, int, int>, double>>* f_tens,
                        std::vector<std::pair<std::tuple<int, int, int>, MKL_Complex16>>* z_tens,
                        int N, bool need_sort = true) {

    auto& f_tensor = *f_tens;
    auto& z_tensor = *z_tens;

    int M = N * N - 1;

    auto cmp = []<typename T>(const std::pair<std::tuple<int, int, int>, T>& left,
                              const std::pair<std::tuple<int, int, int>, T>& right) {
        if (std::get<1>(left.first) == std::get<1>(right.first)) {
            return std::get<2>(left.first) < std::get<2>(right.first);
        }
        return std::get<1>(left.first) < std::get<1>(right.first);
    };

    // сортировка f_tensor и z_tensor по второму и третьему индексу
    if (need_sort) {
        std::sort(f_tensor.begin(), f_tensor.end(), cmp);
        std::sort(z_tensor.begin(), z_tensor.end(), cmp);
    }

    std::vector<std::pair<std::tuple<int, int>, std::array<MKL_Complex16, 2>>> f_tensor_ss;
    f_tensor_ss.reserve(f_tensor.size());
    int cur_n, cur_s;
    cur_n = cur_s = -1;

    for (size_t i = 0; i < f_tensor.size(); ++i) {
        const auto& [f_ind, value] = f_tensor[i];
        const auto& [m, n, s] = f_ind;
        if (cur_n == n and cur_s == s) {
            f_tensor_ss.back().second[0] += l_coeff_conjugate[m] * value;
            f_tensor_ss.back().second[1] += l_coeff[m] * value;
        } else {
            f_tensor_ss.emplace_back(std::tuple(n, s),
                                     std::array{l_coeff_conjugate[m] * value, l_coeff[m] * value});
            cur_n = n;
            cur_s = s;
        }
    }

    std::vector<std::pair<std::tuple<int, int>, std::array<MKL_Complex16, 2>>> z_tensor_ss;
    z_tensor_ss.reserve(z_tensor.size());
    cur_n = cur_s = -1;

    for (size_t i = 0; i < z_tensor.size(); ++i) {
        const auto& [z_ind, value] = z_tensor[i];
        const auto& [m, n, s] = z_ind;
        if (cur_n == n and cur_s == s) {
            z_tensor_ss.back().second[0] += l_coeff[m] * value;
            z_tensor_ss.back().second[1] += l_coeff_conjugate[m] * Conjugate(value);
        } else {
            z_tensor_ss.emplace_back(
                std::tuple(n, s),
                std::array{l_coeff[m] * value, l_coeff_conjugate[m] * Conjugate(value)});
            cur_n = n;
            cur_s = s;
        }
    }

    double* r_tensor = (double*)mkl_malloc(M * M * sizeof(double), 64);

    memset(r_tensor, 0, M * M * sizeof(double));

    // Добавляем страж-элемент для упрощения проверки границ в циклах while
    // Чтобы ниже в циклах не проверять end_z < z_tensor_ss.size() и end_f < f_tensor_ss.size()
    z_tensor_ss.emplace_back(std::tuple(M, 0),
                             std::array{MKL_Complex16{0, 0}, MKL_Complex16{0, 0}});
    f_tensor_ss.emplace_back(std::tuple(M, 0),
                             std::array{MKL_Complex16{0, 0}, MKL_Complex16{0, 0}});

    size_t start_z = 0;
    size_t start_f = 0;
    for (int l = 0; l < M; ++l) {
        size_t end_z = start_z;
        size_t end_f = start_f;

        while (std::get<0>(z_tensor_ss[end_z].first) == l) {
            ++end_z;
        }

        while (std::get<0>(f_tensor_ss[end_f].first) == l) {
            ++end_f;
        }

        for (size_t i = start_z; i < end_z; ++i) {
            for (size_t j = start_f; j < end_f; ++j) {
                int s = std::get<1>(f_tensor_ss[j].first);
                int n = std::get<1>(z_tensor_ss[i].first);

                r_tensor[s * M + n] +=
                    -0.25 * (f_tensor_ss[j].second[0] * z_tensor_ss[i].second[0] +
                             f_tensor_ss[j].second[1] * z_tensor_ss[i].second[1])
                                .real;
            }
        }

        start_z = end_z;
        start_f = end_f;
    }

    return r_tensor;
}


double* GenerateMatrixR(const std::vector<std::vector<MKL_Complex16>>& l_coeff,
                        const std::vector<std::vector<MKL_Complex16>>& l_coeff_conjugate,
                        std::vector<std::pair<std::tuple<int, int, int>, double>>* f_tens,
                        std::vector<std::pair<std::tuple<int, int, int>, MKL_Complex16>>* z_tens,
                        int N, bool need_sort = true) {

    size_t P = l_coeff.size();

    auto& f_tensor = *f_tens;
    auto& z_tensor = *z_tens;

    int M = N * N - 1;

    // сортировка по второму и третьему индексу
    auto cmp = []<typename T>(const std::pair<std::tuple<int, int, int>, T>& left,
                              const std::pair<std::tuple<int, int, int>, T>& right) {
        if (std::get<1>(left.first) == std::get<1>(right.first)) {
            return std::get<2>(left.first) < std::get<2>(right.first);
        }
        return std::get<1>(left.first) < std::get<1>(right.first);
    };

    // сортировка f_tensor и z_tensor по второму и третьему индексу
    if (need_sort) {
        std::sort(f_tensor.begin(), f_tensor.end(), cmp);
        std::sort(z_tensor.begin(), z_tensor.end(), cmp);
    }

    double* r_tensor = (double*)mkl_malloc(M * M * sizeof(double), 64);
    memset(r_tensor, 0, M * M * sizeof(double));

    std::vector<std::pair<std::tuple<int, int>, std::array<MKL_Complex16, 2>>> f_tensor_ss;
    f_tensor_ss.reserve(f_tensor.size());
    std::vector<std::pair<std::tuple<int, int>, std::array<MKL_Complex16, 2>>> z_tensor_ss;
    z_tensor_ss.reserve(z_tensor.size());

    for (size_t cnt = 0; cnt < P; ++cnt) {

        int cur_n, cur_s;
        cur_n = cur_s = -1;

        for (size_t i = 0; i < f_tensor.size(); ++i) {
            const auto& [f_ind, value] = f_tensor[i];
            const auto& [m, n, s] = f_ind;
            if (cur_n == n and cur_s == s) {
                f_tensor_ss.back().second[0] += l_coeff_conjugate[cnt][m] * value;
                f_tensor_ss.back().second[1] += l_coeff[cnt][m] * value;
            } else {
                f_tensor_ss.emplace_back(
                    std::tuple(n, s),
                    std::array{l_coeff_conjugate[cnt][m] * value, l_coeff[cnt][m] * value});
                cur_n = n;
                cur_s = s;
            }
        }

        cur_n = cur_s = -1;

        for (size_t i = 0; i < z_tensor.size(); ++i) {
            const auto& [z_ind, value] = z_tensor[i];
            const auto& [m, n, s] = z_ind;
            if (cur_n == n and cur_s == s) {
                z_tensor_ss.back().second[0] += l_coeff[cnt][m] * value;
                z_tensor_ss.back().second[1] += l_coeff_conjugate[cnt][m] * Conjugate(value);
            } else {
                z_tensor_ss.emplace_back(std::tuple(n, s),
                                         std::array{l_coeff[cnt][m] * value,
                                                    l_coeff_conjugate[cnt][m] * Conjugate(value)});
                cur_n = n;
                cur_s = s;
            }
        }

        // Добавляем страж-элемент для упрощения проверки границ в циклах while
        // Чтобы ниже в циклах не проверять end_z < z_tensor_ss.size() и end_f <
        // f_tensor_ss.size()
        z_tensor_ss.emplace_back(std::tuple(M, 0),
                                 std::array{MKL_Complex16{0, 0}, MKL_Complex16{0, 0}});
        f_tensor_ss.emplace_back(std::tuple(M, 0),
                                 std::array{MKL_Complex16{0, 0}, MKL_Complex16{0, 0}});

        size_t start_z = 0;
        size_t start_f = 0;
        for (int l = 0; l < M; ++l) {
            size_t end_z = start_z;
            size_t end_f = start_f;

            while (std::get<0>(z_tensor_ss[end_z].first) == l) {
                ++end_z;
            }

            while (std::get<0>(f_tensor_ss[end_f].first) == l) {
                ++end_f;
            }

            for (size_t i = start_z; i < end_z; ++i) {
                for (size_t j = start_f; j < end_f; ++j) {
                    int s = std::get<1>(f_tensor_ss[j].first);
                    int n = std::get<1>(z_tensor_ss[i].first);

                    r_tensor[s * M + n] +=
                        -0.25 * (f_tensor_ss[j].second[0] * z_tensor_ss[i].second[0] +
                                 f_tensor_ss[j].second[1] * z_tensor_ss[i].second[1])
                                    .real;
                }
            }

            start_z = end_z;
            start_f = end_f;
        }

        f_tensor_ss.clear();
        z_tensor_ss.clear();
    }

    return r_tensor;
}


// вычисление R через матрицу Коссаковски
// будет накапливаться большая численная погрешность
// не надо так делать
double* GenerateMatrixR(auto kossakovski_func,
                        std::vector<std::pair<std::tuple<int, int, int>, double>>* f_tens,
                        std::vector<std::pair<std::tuple<int, int, int>, MKL_Complex16>>* z_tens,
                        int N, bool need_sort = true) {

    auto& f_tensor = *f_tens;
    auto& z_tensor = *z_tens;

    int M = N * N - 1;

    auto cmp = []<typename T>(const std::pair<std::tuple<int, int, int>, T>& left,
                              const std::pair<std::tuple<int, int, int>, T>& right) {
        return std::get<1>(left.first) < std::get<1>(right.first);
    };

    // сортировка f_tensor и z_tensor по второму и третьему индексу
    if (need_sort) {
        std::sort(f_tensor.begin(), f_tensor.end(), cmp);
        std::sort(z_tensor.begin(), z_tensor.end(), cmp);
    }

    double* r_tensor = (double*)mkl_malloc(M * M * sizeof(double), 64);

    memset(r_tensor, 0, M * M * sizeof(double));

    // Добавляем страж-элемент для упрощения проверки границ в циклах while
    // Чтобы ниже в циклах не проверять end_z < z_tensor_ss.size() и end_f < f_tensor_ss.size()
    z_tensor.emplace_back(std::tuple(0, M, 0), MKL_Complex16{0, 0});
    f_tensor.emplace_back(std::tuple(0, M, 0), 0);

    size_t start_z = 0;
    size_t start_f = 0;
    for (int l = 0; l < M; ++l) {
        size_t end_z = start_z;
        size_t end_f = start_f;

        while (std::get<1>(z_tensor[end_z].first) == l) {
            ++end_z;
        }

        while (std::get<1>(f_tensor[end_f].first) == l) {
            ++end_f;
        }

        for (size_t i_cnt = start_z; i_cnt < end_z; ++i_cnt) {
            for (size_t j_cnt = start_f; j_cnt < end_f; ++j_cnt) {

                int s = std::get<2>(f_tensor[j_cnt].first);
                int n = std::get<2>(z_tensor[i_cnt].first);

                int j = std::get<0>(z_tensor[i_cnt].first);
                int k = std::get<0>(f_tensor[j_cnt].first);

                r_tensor[s * M + n] += -0.25 * (kossakovski_func(j, k) * f_tensor[j_cnt].second *
                                                    z_tensor[i_cnt].second +
                                                kossakovski_func(k, j) * f_tensor[j_cnt].second *
                                                    Conjugate(z_tensor[i_cnt].second))
                                                   .real;
            }
        }

        start_z = end_z;
        start_f = end_f;
    }

    f_tensor.pop_back();
    z_tensor.pop_back();

    return r_tensor;
}
