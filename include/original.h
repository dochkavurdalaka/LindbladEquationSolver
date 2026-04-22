#pragma once

#include <stdexcept>
#include "mkl.h"


void Commutator(const MKL_Complex16* A, const MKL_Complex16* B, MKL_Complex16* result, int N) {
    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    cblas_zgemm(CblasRowMajor,  // Указывает, что матрицы хранятся построчно
                                // (стандарт для C/C++)
                CblasNoTrans,   // Операция для A: не транспонировать
                CblasNoTrans,   // Операция для B: не транспонировать
                N,              // Количество строк в матрице A (и C)
                N,              // Количество столбцов в матрице B (и C)
                N,              // Количество столбцов в A и строк в B
                &alpha,         // Указатель на скаляр alpha
                A,            // Матрица A
                N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                // количество столбцов.
                B,            // Матрица B
                N,              // Ведущий размер для B.
                &beta,          // Указатель на скаляр beta
                result,         // Матрица C (результат)
                N               // Ведущий размер для C.
    );
    alpha = {-1.0, 0.0};
    beta = {1.0, 0.0};
    cblas_zgemm(CblasRowMajor,  // Указывает, что матрицы хранятся построчно
                                // (стандарт для C/C++)
                CblasNoTrans,   // Операция для A: не транспонировать
                CblasNoTrans,   // Операция для B: не транспонировать
                N,              // Количество строк в матрице A (и C)
                N,              // Количество столбцов в матрице B (и C)
                N,              // Количество столбцов в A и строк в B
                &alpha,         // Указатель на скаляр alpha
                B,            // Матрица A
                N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                // количество столбцов.
                A,            // Матрица B
                N,              // Ведущий размер для B.
                &beta,          // Указатель на скаляр beta
                result,         // Матрица C (результат)
                N               // Ведущий размер для C.
    );
}

void AntiCommutator(const MKL_Complex16* A, const MKL_Complex16* B, MKL_Complex16* result,
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
                A,            // Матрица A
                N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                // количество столбцов.
                B,            // Матрица B
                N,              // Ведущий размер для B.
                &beta,          // Указатель на скаляр beta
                result,         // Матрица C (результат)
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
                B,            // Матрица A
                N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                // количество столбцов.
                A,            // Матрица B
                N,              // Ведущий размер для B.
                &beta,          // Указатель на скаляр beta
                result,         // Матрица C (результат)
                N               // Ведущий размер для C.
    );
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
                throw std::runtime_error("not enough memory");
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
                throw std::runtime_error("not enough memory");
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
            throw std::runtime_error("not enough memory");
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
