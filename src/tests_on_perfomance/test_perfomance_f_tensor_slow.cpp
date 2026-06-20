#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "mkl.h"
#include "mkl_complex16.h"
#include "original.h"
#include "timer.h"


int main(int argc, char* argv[]) {
    // Параметры
    int N = 10;
    // на случай если передаем размер N в параметрах командной строки
    if (argc == 2) {
        N = std::atoi(argv[1]);
    }
    int M = N * N - 1;
    RAMMeter meter;
    Timer timer;
    MKL_Complex16* end_result = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    std::vector<MKL_Complex16*> basis_array = CreateBasisArray(N);

    std::vector<std::vector<MKL_Complex16*>> commutator(M);
    for (int m = 0; m < M; ++m) {
        commutator[m].resize(M);
        for (int n = 0; n < M; ++n) {
            commutator[m][n] = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            Commutator(basis_array[m], basis_array[n], commutator[m][n], N);
        }
    }

    // Общее количество элементов
    size_t total_elements = M * M * M;

    // Выделение памяти для тензора
    double *f_tensor = (double *)mkl_malloc(total_elements * sizeof(double), 64);

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
                            N,                     // Ведущий размер для B.
                            &beta,                 // Указатель на скаляр beta
                            end_result,            // Матрица C (результат)
                            N                      // Ведущий размер для C.
                );

                int index = m * M * M + n * M + s;
                f_tensor[index] = Trace(end_result, N).imag;
            }
        }
    }

    timer.stop();
    meter.tick();
    std::cout << f_tensor[M] << "\n";

    // Освобождение памяти
    for (auto& el: basis_array) {
        mkl_free(el);
    }
    mkl_free(f_tensor);
    mkl_free(end_result);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < M; ++n) {
            mkl_free(commutator[m][n]);
        }
    }

}