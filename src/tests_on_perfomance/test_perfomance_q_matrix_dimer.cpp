#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>

#include "mkl.h"
#include "lindblad_utils_dimer.h"
#include "mkl_complex16.h"
#include "timer.h"


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

    std::vector<double> h_coeff = model.GetBaseHCoefDimer(0.);

    auto f_tensor = GenerateTensorF(N);

    RAMMeter meter;
    Timer timer;
    SparseQBuilder q_builder(&f_tensor, h_coeff, N);
    sparse_matrix_t q_matrix = q_builder.GetMatrix();
    timer.stop();
    meter.tick();


    h_coeff[0] = -1;
    timer.start();
    q_builder.UpdateValues(h_coeff);
    timer.stop();

    // филлерный код, чтобы компилятор не выкинул вышенаписанный код ничего
    print_first_element(q_matrix);

}