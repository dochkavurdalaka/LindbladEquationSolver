#include <iostream>
#include <vector>

#include "timer.h"
#include "mkl.h"
#include "generate_matrices.h"
#include "mkl_complex16.h"
#include "lindblad_utils_dimer.h"

// #include "mkl_spblas.h"


// здесь и далее попробуем решить систему для константного H
// тогда матрица Q тоже будет константной

void print_first_element(sparse_matrix_t A) {
    if (!A) {
        std::cout << "Matrix is null" << std::endl;
        return;
    }
    
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT* row_start = nullptr;
    MKL_INT* row_end = nullptr;
    MKL_INT* col_indx = nullptr;
    double* values = nullptr;
    
    sparse_status_t status = mkl_sparse_d_export_csr(A, &indexing, &rows, &cols,
                                                      &row_start, &row_end, 
                                                      &col_indx, (double**)&values);
    
    if (status == SPARSE_STATUS_SUCCESS && row_start && row_end && values) {
        MKL_INT nnz = row_end[0] - row_start[0];
        if (nnz > 0) {
            std::cout << "Matrix " << rows << "x" << cols 
                      << ", first element: value=" << values[0] 
                      << " at column=" << col_indx[0] << std::endl;
        } else {
            std::cout << "Matrix is empty" << std::endl;
        }
    } else {
        std::cout << "Failed to export matrix" << std::endl;
    }
}


int main(int argc, char* argv[]) {
    // Параметры
    int N = 50;
    if (argc == 2) {
        N = std::atoi(argv[1]);
    }


    auto theta = [T = 2 * std::numbers::pi](double t) {
        t = fmod(t, T);
        if (t < T / 2) {
            return 1.;
        }
        return -1.;
    };

    DimerModel<decltype(theta)> model(theta, N);

    std::vector<MKL_Complex16> l_coeff = model.GetLCoefDimer();

    std::vector<MKL_Complex16> l_coeff_conjugate(l_coeff);
    for (auto& elem : l_coeff_conjugate) {
        elem = Conjugate(elem);
    }


    auto f_tensor = GenerateTensorF(N);
    auto d_tensor = GenerateTensorD(N);
    auto z_tensor = GenerateTensorZ(f_tensor, d_tensor);

    RAMMeter meter;
    Timer timer;
    SparseRBuilder r_builder(l_coeff, l_coeff_conjugate, &f_tensor, &z_tensor, N);
    sparse_matrix_t r_matrix = r_builder.GetMatrix();
    timer.stop();
    meter.tick();

    // филлерный код, чтобы компилятор не выкинул вышенаписанный код ничего
    print_first_element(r_matrix);

}
