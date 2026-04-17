#include <vector>
#include <cmath>
#include "mkl.h"
#include "mkl_complex16.h"

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


double* GetVCoef(MKL_Complex16* rho, int N) {
    double* v_coeff = (double*)mkl_malloc((N * N - 1) * sizeof(double), 64);
    size_t ind = 0;
    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            int index = j * N + k;
            v_coeff[ind] = sqrt(2.) * rho[index].real;
            ++ind;
        }
    }

    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            int index = k * N + j;
            v_coeff[ind] = sqrt(2.) * rho[index].imag;
            ++ind;
        }
    }

    for (int l = 0; l < N - 1; ++l) {
        double coeff = 0.;

        for (int k = 0; k < l + 1; ++k) {
            int index = k * N + k;
            coeff += rho[index].real / sqrt((l + 1) * (l + 2));
        }

        int index = (l + 1) * N + (l + 1);
        coeff += -sqrt(l + 1) * rho[index].real / sqrt(l + 2);

        v_coeff[ind] = coeff;
        ++ind;
    }

    return v_coeff;
}