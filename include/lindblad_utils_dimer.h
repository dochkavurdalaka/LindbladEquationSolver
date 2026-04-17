#pragma once

#include <vector>
#include <cmath>
#include <tuple>
#include <unordered_map>
#include <functional>
#include <numbers>

#include "mkl.h"
#include "lindblad_utils.h"

template <class Functor>
struct DimerModel {
    double J = -1.0;
    double U = 2.2;
    double E = -1.0;
    double A = -1.5;
    Functor theta;

    double gamma = 0.1;
    int N;
    explicit DimerModel(Functor f, int N) : theta(std::move(f)), N(N) {
    }

    std::vector<double> GetBaseHCoefDimer(double t) {
        std::vector<double> h_coeff(N * N - 1, 0);

        for (int n = 0; n + 1 < N; ++n) {
            int index = Mapping(n, n + 1, N);
            h_coeff[index] = -J * sqrt(2. * (n + 1) * (N - 1 - n));
        }

        std::vector<double> hamiltonian_diag(N, 0);
        for (int n = 0; n < N; ++n) {
            hamiltonian_diag[n] = (2.0 * U / (N - 1)) * (n * (n - 1) + (N - 1 - n) * (N - 2 - n));
            hamiltonian_diag[n] += (E + A * theta(t)) * (N - 1 - 2 * n);
        }

        for (int l = 0; l < N - 1; ++l) {
            double coeff = 0.;

            for (int k = 0; k < l + 1; ++k) {
                coeff += hamiltonian_diag[k] / sqrt((l + 1) * (l + 2));
            }

            coeff += -sqrt(l + 1) * hamiltonian_diag[l + 1] / sqrt(l + 2);

            h_coeff[N * (N - 1) + l] = coeff;
        }

        return h_coeff;
    }

    void GetUpdateHCoefDimer(std::vector<double>* h, double t) {

        std::fill(h->begin() + N * (N - 1), h->end(), 0);

        auto& h_coeff = *h;

        std::vector<double> hamiltonian_diag(N, 0);
        for (int n = 0; n < N; ++n) {
            hamiltonian_diag[n] = (2.0 * U / (N - 1)) * (n * (n - 1) + (N - 1 - n) * (N - 2 - n));
            hamiltonian_diag[n] += (E + A * theta(t)) * (N - 1 - 2 * n);
        }

        for (int l = 0; l < N - 1; ++l) {
            double coeff = 0.;

            for (int k = 0; k < l + 1; ++k) {
                coeff += hamiltonian_diag[k] / sqrt((l + 1) * (l + 2));
            }

            coeff += -sqrt(l + 1) * hamiltonian_diag[l + 1] / sqrt(l + 2);

            h_coeff[N * (N - 1) + l] = coeff;
        }
    }

    std::vector<MKL_Complex16> GetLCoefDimer() {
        double mult_coeff = sqrt(gamma / (N - 1));
        std::vector<MKL_Complex16> l_coeff(N * N - 1, {0, 0});
        for (int n = 0; n + 1 < N; ++n) {
            int index = Mapping(n, n + 1, N);
            l_coeff[index].real = mult_coeff * sqrt(2. * (n + 1) * (N - 1 - n));
        }
        return l_coeff;
    }
};

template <size_t sort_ind = 1>
struct SparseQBuilder {
    SparseQBuilder(std::vector<std::pair<std::tuple<int, int, int>, double>>* f_tensor,
                   const std::vector<double>& h_coeff, int N);

    void update_values(const std::vector<double>& h_coeff);

    sparse_matrix_t get_matrix() {
        return q_;
    }

    ~SparseQBuilder() {
        if (q_) {
            mkl_sparse_destroy(q_);
        }
    }

private:
    std::vector<char> check_non_zero_hcoeff_;

    int rows_;
    int cols_;

    std::vector<MKL_INT> row_ptr_;
    std::vector<MKL_INT> col_ind_;

    struct Contribution {
        int value_ind;
        int h_ind;
        double f_tensor_value;
    };
    std::vector<Contribution> contributions_;
    std::vector<double> values_;

    sparse_matrix_t q_;
};

template <size_t sort_ind>
SparseQBuilder<sort_ind>::SparseQBuilder(
    std::vector<std::pair<std::tuple<int, int, int>, double>>* f_tensor,
    const std::vector<double>& h_coeff, int N) {

    check_non_zero_hcoeff_.resize(N * N - 1, false);
    for (int n = 0; n + 1 < N; ++n) {
        int index = Mapping(n, n + 1, N);
        // если true элемент ненулевой
        check_non_zero_hcoeff_[index] = true;
    }

    for (int index = N * (N - 1); index < N * N - 1; ++index) {
        check_non_zero_hcoeff_[index] = true;
    }

    auto cmp = [](const std::pair<std::tuple<int, int, int>, double>& left,
                  const std::pair<std::tuple<int, int, int>, double>& right) {
        if (std::get<2>(left.first) == std::get<2>(right.first)) {
            return std::get<1>(left.first) < std::get<1>(right.first);
        }
        return std::get<2>(left.first) < std::get<2>(right.first);
    };

        // сортировка f_tensor по второму и третьему индексу
    if constexpr (sort_ind == 1) {
        std::sort(f_tensor->begin(), f_tensor->end(), cmp);
    } else if constexpr (sort_ind == 2) {
        CountSortTensorSN(f_tensor, N);
    }

    auto& f_tensor_sorted = *f_tensor;

    col_ind_.reserve(f_tensor_sorted.size());

    // здесь будут храниться индексы элементов в массиве values и индексы которые им соотвествуют
    // спросить у Линева про этот пункт
    contributions_.reserve(f_tensor_sorted.size() / N);

    // размерность q : (N^2 - 1)*(N^2 - 1)
    rows_ = N * N - 1;
    row_ptr_.resize(rows_ + 1, 0);
    cols_ = N * N - 1;

    int cur_row = -1;
    int cur_col = -1;
    int current_contribution = -1;

    for (const auto& [f_ind, value] : f_tensor_sorted) {
        const auto& [m, col, row] = f_ind;
        if (check_non_zero_hcoeff_[m]) {
            if (not(col == cur_col and row == cur_row)) {
                col_ind_.push_back(col);
                ++row_ptr_[row + 1];
                cur_row = row;
                cur_col = col;
                ++current_contribution;
            }

            contributions_.push_back({current_contribution, m, value});
        }
    }

    for (size_t i = 1; i < row_ptr_.size(); ++i) {
        row_ptr_[i] += row_ptr_[i - 1];
    }

    values_.resize(row_ptr_.back(), 0);

    for (auto [value_ind, h_ind, f_tensor_value] : contributions_) {
        values_[value_ind] += h_coeff[h_ind] * f_tensor_value;
    }

    mkl_sparse_d_create_csr(&q_, SPARSE_INDEX_BASE_ZERO, rows_, cols_, row_ptr_.data(),
                            row_ptr_.data() + 1, col_ind_.data(), values_.data());

    mkl_sparse_optimize(q_);  // вызываем только один раз!
}

template <size_t sort_ind>
void SparseQBuilder<sort_ind>::update_values(const std::vector<double>& h_coeff) {

    std::fill(values_.begin(), values_.end(), 0);
    for (auto [value_ind, h_ind, f_tensor_value] : contributions_) {
        values_[value_ind] += h_coeff[h_ind] * f_tensor_value;
    }
}

template <size_t sort_ind_f = 1, size_t sort_ind_z = 1>
struct SparseRBuilder {
    SparseRBuilder(const std::vector<MKL_Complex16>& l_coeff,
                   const std::vector<MKL_Complex16>& l_coeff_conjugate,
                   std::vector<std::pair<std::tuple<int, int, int>, double>>* f_tens,
                   std::vector<std::pair<std::tuple<int, int, int>, MKL_Complex16>>* z_tens, int N);

    sparse_matrix_t get_matrix() {
        return r_;
    }

    ~SparseRBuilder() {
        if (r_) {
            mkl_sparse_destroy(r_);
        }
    }

private:
    int rows_;
    int cols_;

    std::vector<MKL_INT> row_ptr_;
    std::vector<MKL_INT> col_ind_;

    std::vector<double> values_;

    sparse_matrix_t r_;
};

template <size_t sort_ind_f, size_t sort_ind_z>
SparseRBuilder<sort_ind_f, sort_ind_z>::SparseRBuilder(
    const std::vector<MKL_Complex16>& l_coeff, const std::vector<MKL_Complex16>& l_coeff_conjugate,
    std::vector<std::pair<std::tuple<int, int, int>, double>>* f_tens,
    std::vector<std::pair<std::tuple<int, int, int>, MKL_Complex16>>* z_tens, int N) {

    auto& f_tensor = *f_tens;
    auto& z_tensor = *z_tens;

    // сортировка f_tensor по второму и третьему индексу
    auto cmp = []<typename T>(const std::pair<std::tuple<int, int, int>, T>& left,
                              const std::pair<std::tuple<int, int, int>, T>& right) {
        if (std::get<1>(left.first) == std::get<1>(right.first)) {
            return std::get<2>(left.first) < std::get<2>(right.first);
        }
        return std::get<1>(left.first) < std::get<1>(right.first);
    };

    // сортировка f_tensor по второму и третьему индексу
    if constexpr (sort_ind_f == 1) {
        std::sort(f_tensor.begin(), f_tensor.end(), cmp);
    } else if constexpr (sort_ind_f == 2) {
        CountSortTensorNS(f_tens, N);
    }

    // сортировка z_tensor по второму и третьему индексу
    if constexpr (sort_ind_z == 1) {
        std::sort(z_tensor.begin(), z_tensor.end(), cmp);
    } else if constexpr (sort_ind_z == 2) {
        CountSortTensorNS(z_tens, N);
    }

    std::vector<std::pair<std::tuple<int, int>, std::array<MKL_Complex16, 2>>> f_tensor_ss;
    int cur_n, cur_s;
    cur_n = cur_s = -1;

    for (size_t i = 0; i < f_tensor.size(); ++i) {
        const auto& [f_ind, value] = f_tensor[i];
        const auto& [m, n, s] = f_ind;

        // если l = 0, тогда l сопряженное тоже = 0
        if (l_coeff_conjugate[m] != 0) {
            if (cur_n == n and cur_s == s) {
                f_tensor_ss.back().second[0] += l_coeff_conjugate[m] * value;
                f_tensor_ss.back().second[1] += l_coeff[m] * value;
            } else {
                f_tensor_ss.emplace_back(
                    std::tuple(n, s), std::array{l_coeff_conjugate[m] * value, l_coeff[m] * value});
                cur_n = n;
                cur_s = s;
            }
        }
    }

    cur_n = cur_s = -1;

    std::vector<std::pair<std::tuple<int, int>, std::array<MKL_Complex16, 2>>> z_tensor_ss;

    for (size_t i = 0; i < z_tensor.size(); ++i) {
        const auto& [z_ind, value] = z_tensor[i];
        const auto& [m, n, s] = z_ind;
        // если l = 0, тогда l сопряженное тоже = 0
        if (l_coeff[m] != 0) {
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
    }

    int M = N * N - 1;
    std::vector<std::unordered_map<int, double>> r_sparse(M);
    for (int i = 0; i < M; ++i) {
        r_sparse[i].reserve(N);
    }

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

                r_sparse[s][n] += -0.25 * (f_tensor_ss[j].second[0] * z_tensor_ss[i].second[0] +
                                           f_tensor_ss[j].second[1] * z_tensor_ss[i].second[1])
                                              .real;
            }
        }

        start_z = end_z;
        start_f = end_f;
    }

    rows_ = M;
    cols_ = M;

    // Подсчет nnz
    size_t nnz = 0;
    for (const auto& row_map : r_sparse) {
        nnz += row_map.size();
    }

    row_ptr_.resize(rows_ + 1);

    row_ptr_[0] = 0;
    for (size_t i = 0; i < r_sparse.size(); ++i) {
        row_ptr_[i + 1] = row_ptr_[i] + r_sparse[i].size();
    }

    col_ind_.reserve(nnz);
    values_.reserve(nnz);

    std::vector<std::pair<int, double>> arr_for_sort;
    arr_for_sort.reserve(M);

    for (const auto& mas : r_sparse) {
        for (auto& elem : mas) {
            arr_for_sort.push_back(elem);
        }
        std::sort(arr_for_sort.begin(), arr_for_sort.end());

        for (auto& [ind, value] : arr_for_sort) {
            col_ind_.push_back(ind);
            values_.push_back(value);
        }

        arr_for_sort.clear();
    }

    mkl_sparse_d_create_csr(&r_, SPARSE_INDEX_BASE_ZERO, rows_, cols_, row_ptr_.data(),
                            row_ptr_.data() + 1, col_ind_.data(), values_.data());

    mkl_sparse_optimize(r_);  // вызываем только один раз!
}
