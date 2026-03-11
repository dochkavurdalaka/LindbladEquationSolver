#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <unordered_map>
#include <functional>
#include <numbers>

#include "mkl.h"
#include "lindblad_utils.h"

struct DimerModel {
    double J = -1.0;
    double U = 2.2;
    double E = -1.0;
    double A = -1.5;
    std::function<double(double)> theta;
    // вот здесь нужно будет поставить нормальный pi
    double T = 2 * std::numbers::pi;
    double gamma = 0.1;
    int N;
};

// Показать Линеву, tuple нужно скорее всего будет заменить на struct
struct SparseQBuilder {
    SparseQBuilder(std::vector<std::pair<std::tuple<int, int, int>, double>>* f_tensor,
                   const std::vector<double>& h_coeff, int N);

    void update_values(const std::vector<double>& h_coeff);

    sparse_matrix_t get_matrix();

    ~SparseQBuilder() {
        if (q_) {
            mkl_sparse_destroy(q_);
        }
    }

private:
    // спросить у Линева про этот способ и можно ли так делать
    // если true !=0, если false = 0
    bool CheckNonZeroHCoeff(int index, int N) {
        if (index < N * (N - 1) / 2) {
            return check[index];
        } else if (index >= N * (N - 1)) {
            return true;
        }
        return false;
    }

    std::vector<char> check;

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

SparseQBuilder::SparseQBuilder(std::vector<std::pair<std::tuple<int, int, int>, double>>* f_tensor,
                               const std::vector<double>& h_coeff, int N) {

    check.resize(N * (N - 1) / 2, false);
    for (int n = 0; n + 1 < N; ++n) {
        int index = Mapping(n, n + 1, N);
        // если true элемент ненулевой
        check[index] = true;
    }

    auto cmp = [](const std::pair<std::tuple<int, int, int>, double>& left,
                  const std::pair<std::tuple<int, int, int>, double>& right) {
        if (std::get<2>(left.first) == std::get<2>(right.first)) {
            return std::get<1>(left.first) < std::get<1>(right.first);
        }
        return std::get<2>(left.first) < std::get<2>(right.first);
    };

    std::sort(f_tensor->begin(), f_tensor->end(), cmp);

    auto& f_tensor_sorted = *f_tensor;

    col_ind_.reserve(f_tensor_sorted.size());

    // здесь будут храниться индексы элементов в массиве values и индексы которые им соотвествуют
    // спросить у Линева про этот пункт
    contributions_.reserve(f_tensor_sorted.size() / N);

    // размерность q : (N^2 - 1)*(N^2 - 1)
    rows_ = N * N - 1;
    row_ptr_.resize(rows_ + 1, 0);
    cols_ = N * N - 1;

    int current_row = -1;
    int current_col = -1;
    int current_contribution = -1;

    for (size_t i = 0; i < f_tensor_sorted.size(); ++i) {
        if (CheckNonZeroHCoeff(std::get<0>(f_tensor_sorted[i].first), N)) {
            int row = std::get<2>(f_tensor_sorted[i].first);
            int col = std::get<1>(f_tensor_sorted[i].first);

            if (not(row == current_row and col == current_col)) {
                col_ind_.push_back(col);
                ++row_ptr_[row + 1];
                current_row = row;
                current_col = col;
                ++current_contribution;
            }

            contributions_.push_back({current_contribution, std::get<0>(f_tensor_sorted[i].first),
                                      f_tensor_sorted[i].second});
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

sparse_matrix_t SparseQBuilder::get_matrix() {
    return q_;
}

// в целях оптимизации нужно будет здесь добавить локальную аккумуляцию
void SparseQBuilder::update_values(const std::vector<double>& h_coeff) {

    std::fill(values_.begin(), values_.end(), 0);
    for (auto [value_ind, h_ind, f_tensor_value] : contributions_) {
        values_[value_ind] += h_coeff[h_ind] * f_tensor_value;
    }
}

// Показать Линеву, tuple нужно скорее всего будет заменить на struct
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

SparseRBuilder::SparseRBuilder(
    const std::vector<MKL_Complex16>& l_coeff, const std::vector<MKL_Complex16>& l_coeff_conjugate,
    std::vector<std::pair<std::tuple<int, int, int>, double>>* f_tens,
    std::vector<std::pair<std::tuple<int, int, int>, MKL_Complex16>>* z_tens, int N) {

    auto& f_tensor = *f_tens;
    auto& z_tensor = *z_tens;

    int M = N * N - 1;

    std::vector<std::unordered_map<int, double>> r_sparse(M);
    for (auto& m : r_sparse) {
        m.reserve(N);
    }

    // сортировка f_tensor по второму и третьему индексу
    auto cmp = []<typename T>(const std::pair<std::tuple<int, int, int>, T>& left,
                              const std::pair<std::tuple<int, int, int>, T>& right) {
        if (std::get<1>(left.first) == std::get<1>(right.first)) {
            return std::get<2>(left.first) < std::get<2>(right.first);
        }
        return std::get<1>(left.first) < std::get<1>(right.first);
    };

    std::sort(f_tensor.begin(), f_tensor.end(), cmp);

    auto f_tensor_cmplx = DoubleToComplexTensor(f_tensor);

    // double* r_tensor = (double*)mkl_malloc(M * M * sizeof(double), 64);
    // memset(r_tensor, 0, M * M * sizeof(double));

    std::pair<std::tuple<int, int>, std::array<MKL_Complex16, 2>> elem;
    std::vector<std::pair<std::tuple<int, int>, std::array<MKL_Complex16, 2>>> f_tensor_ss;
    int current_get_1 = -1;
    int current_get_2 = -1;

    for (size_t i = 0; i < f_tensor_cmplx.size(); ++i) {

        // если l = 0, тогда l сопряженное тоже = 0
        if (l_coeff_conjugate[std::get<0>(f_tensor_cmplx[i].first)] != 0) {

            if (current_get_1 == std::get<1>(f_tensor_cmplx[i].first) and
                current_get_2 == std::get<2>(f_tensor_cmplx[i].first)) {

                f_tensor_ss.back().second[0] +=
                    l_coeff_conjugate[std::get<0>(f_tensor_cmplx[i].first)] *
                    f_tensor_cmplx[i].second;

                f_tensor_ss.back().second[1] +=
                    l_coeff[std::get<0>(f_tensor_cmplx[i].first)] * f_tensor_cmplx[i].second;

            } else {
                elem = {std::tuple(std::get<1>(f_tensor_cmplx[i].first),
                                   std::get<2>(f_tensor_cmplx[i].first)),
                        {l_coeff_conjugate[std::get<0>(f_tensor_cmplx[i].first)] *
                             f_tensor_cmplx[i].second,
                         l_coeff[std::get<0>(f_tensor_cmplx[i].first)] * f_tensor_cmplx[i].second}};

                f_tensor_ss.push_back(elem);

                current_get_1 = std::get<1>(f_tensor_cmplx[i].first);
                current_get_2 = std::get<2>(f_tensor_cmplx[i].first);
            }
        }
    }

    std::sort(z_tensor.begin(), z_tensor.end(), cmp);

    current_get_1 = -1;
    current_get_2 = -1;

    std::vector<std::pair<std::tuple<int, int>, std::array<MKL_Complex16, 2>>> z_tensor_ss;

    for (size_t i = 0; i < z_tensor.size(); ++i) {

        // если l = 0, тогда l сопряженное тоже = 0
        if (l_coeff[std::get<0>(z_tensor[i].first)] != 0) {
            if (current_get_1 == std::get<1>(z_tensor[i].first) and
                current_get_2 == std::get<2>(z_tensor[i].first)) {
                z_tensor_ss.back().second[0] +=
                    l_coeff[std::get<0>(z_tensor[i].first)] * z_tensor[i].second;
                z_tensor_ss.back().second[1] += l_coeff_conjugate[std::get<0>(z_tensor[i].first)] *
                                                Conjugate(z_tensor[i].second);
            } else {
                elem = {std::tuple(std::get<1>(z_tensor[i].first), std::get<2>(z_tensor[i].first)),
                        {l_coeff[std::get<0>(z_tensor[i].first)] * z_tensor[i].second,
                         l_coeff_conjugate[std::get<0>(z_tensor[i].first)] *
                             Conjugate(z_tensor[i].second)}};

                z_tensor_ss.push_back(elem);

                current_get_1 = std::get<1>(z_tensor[i].first);
                current_get_2 = std::get<2>(z_tensor[i].first);
            }
        }
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

    int rows_ = M;
    int cols_ = M;

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

double* GenerateVectorKWithFunctor(
    int N, auto kossakovski_func,
    const std::vector<std::pair<std::tuple<int, int, int>, double>>& f_tensor) {
    int M = N * N - 1;
    double* k_vector = (double*)mkl_malloc(M * sizeof(double), 64);
    memset(k_vector, 0, M * sizeof(double));

    for (const auto& el : f_tensor) {
        MKL_Complex16 a = kossakovski_func(std::get<0>(el.first), std::get<1>(el.first));

        // большая часть a у нас будут 0
        if (a.real != 0 or a.imag != 0) {
            k_vector[std::get<2>(el.first)] += -1. * (a * el.second).imag / N;
        }
    }

    return k_vector;
}

