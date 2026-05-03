#include <vector>
#include <cmath>
#include "mkl.h"
#include "mkl_complex16.h"

std::vector<double> GetHCoef(MKL_Complex16* hamiltonian, size_t N) {
    std::vector<double> h_coeff;
    h_coeff.reserve(N * N - 1);

    for (size_t j = 0; j < N; ++j) {
        for (size_t k = j + 1; k < N; ++k) {
            size_t index = j * N + k;
            h_coeff.push_back(sqrt(2.) * hamiltonian[index].real);
        }
    }

    for (size_t j = 0; j < N; ++j) {
        for (size_t k = j + 1; k < N; ++k) {
            size_t index = k * N + j;
            h_coeff.push_back(sqrt(2.) * hamiltonian[index].imag);
        }
    }

    std::vector<double> hamiltonian_diag_prefixsum(N - 1);
    for (size_t k = 0; k + 1 < N; ++k) {
        hamiltonian_diag_prefixsum[k] = hamiltonian[k * N + k].real;
    }
    for (size_t k = 1; k + 1 < N; ++k) {
        hamiltonian_diag_prefixsum[k] += hamiltonian_diag_prefixsum[k - 1];
    }

    for (size_t l = 0; l + 1 < N; ++l) {
        double coeff = hamiltonian_diag_prefixsum[l] / sqrt((l + 1) * (l + 2));
        size_t index = (l + 1) * N + (l + 1);
        coeff += -sqrt(l + 1) * hamiltonian[index].real / sqrt(l + 2);

        h_coeff.push_back(coeff);
    }

    return h_coeff;
}

std::vector<MKL_Complex16> GetLCoef(MKL_Complex16* lindbladian, size_t N) {
    std::vector<MKL_Complex16> l_coeff;
    l_coeff.reserve(N * N - 1);

    for (size_t j = 0; j < N; ++j) {
        for (size_t k = j + 1; k < N; ++k) {
            MKL_Complex16 coeff = (lindbladian[j * N + k] + lindbladian[k * N + j]) / sqrt(2);
            l_coeff.push_back(coeff);
        }
    }

    for (size_t j = 0; j < N; ++j) {
        for (size_t k = j + 1; k < N; ++k) {

            size_t ind_one = j * N + k;
            size_t ind_two = k * N + j;
            MKL_Complex16 coeff;
            coeff.real = lindbladian[ind_two].imag - lindbladian[ind_one].imag;
            coeff.imag = lindbladian[ind_one].real - lindbladian[ind_two].real;
            l_coeff.push_back(coeff / sqrt(2));
        }
    }

    std::vector<MKL_Complex16> lindbladian_diag_prefixsum(N - 1);
    for (size_t k = 0; k + 1 < N; ++k) {
        lindbladian_diag_prefixsum[k] = lindbladian[k * N + k];
    }
    for (size_t k = 1; k + 1 < N; ++k) {
        lindbladian_diag_prefixsum[k] += lindbladian_diag_prefixsum[k - 1];
    }

    for (size_t l = 0; l + 1 < N; ++l) {
        MKL_Complex16 coeff = lindbladian_diag_prefixsum[l] / sqrt((l + 1) * (l + 2));
        size_t index = (l + 1) * N + (l + 1);
        coeff += -sqrt(l + 1) * lindbladian[index] / sqrt(l + 2);

        l_coeff.push_back(coeff);
    }

    return l_coeff;
}

double* GetVCoef(MKL_Complex16* rho, size_t N) {
    double* v_coeff = (double*)mkl_malloc((N * N - 1) * sizeof(double), 64);
    size_t ind = 0;
    for (size_t j = 0; j < N; ++j) {
        for (size_t k = j + 1; k < N; ++k) {
            size_t index = j * N + k;
            v_coeff[ind] = sqrt(2.) * rho[index].real;
            ++ind;
        }
    }

    for (size_t j = 0; j < N; ++j) {
        for (size_t k = j + 1; k < N; ++k) {
            size_t index = k * N + j;
            v_coeff[ind] = sqrt(2.) * rho[index].imag;
            ++ind;
        }
    }

    std::vector<double> rho_diag_prefixsum(N - 1);
    for (size_t k = 0; k + 1 < N; ++k) {
        rho_diag_prefixsum[k] = rho[k * N + k].real;
    }
    for (size_t k = 1; k + 1 < N; ++k) {
        rho_diag_prefixsum[k] += rho_diag_prefixsum[k - 1];
    }

    for (size_t l = 0; l + 1 < N; ++l) {
        double coeff = rho_diag_prefixsum[l] / sqrt((l + 1) * (l + 2));
        size_t index = (l + 1) * N + (l + 1);
        coeff += -sqrt(l + 1) * rho[index].real / sqrt(l + 2);

        v_coeff[ind] = coeff;
        ++ind;
    }

    return v_coeff;
}

MKL_Complex16* GetDensityBySUDecomposition(double* v, size_t N) {
    MKL_Complex16* rho_final = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
    size_t ind = 0;
    size_t offset = N * (N - 1) / 2;
    for (size_t j = 0; j < N; ++j) {
        for (size_t k = j + 1; k < N; ++k) {
            rho_final[j * N + k].real = v[ind] / sqrt(2);
            rho_final[k * N + j].real = v[ind] / sqrt(2);

            rho_final[j * N + k].imag = -v[offset + ind] / sqrt(2);
            rho_final[k * N + j].imag = v[offset + ind] / sqrt(2);

            ++ind;
        }
    }

    offset = N * (N - 1);
    std::vector<double> diag_inverse_prefix_sum(N - 1);
    for (size_t k = 0; k < N - 1; ++k) {
        diag_inverse_prefix_sum[k] = v[offset + k] / sqrt((k + 1) * (k + 2));
    }

    for (size_t k = 1; k < N - 1; ++k) {
        size_t k_inv = N - 2 - k;
        diag_inverse_prefix_sum[k_inv] += diag_inverse_prefix_sum[k_inv + 1];
    }

    for (size_t l = 0; l + 1 < N; ++l) {
        rho_final[l * N + l] = {0, 0};
        rho_final[l * N + l].real = diag_inverse_prefix_sum[l];
    }

    rho_final[N * N - 1] = {0., 0.};

    for (size_t l = 0; l + 1 < N; ++l) {
        rho_final[(l + 1) * N + l + 1].real += -v[offset + l] * sqrt(1. * (l + 1) / (l + 2));
    }

    for (size_t k = 0; k < N; ++k) {
        rho_final[k * N + k].real += 1. / N;
    }

    return rho_final;
}