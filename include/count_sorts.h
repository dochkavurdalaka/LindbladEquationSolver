#include <vector>
#include <tuple>


template <class NumberType>
void CountSortTensor(std::vector<std::pair<std::tuple<int, int, int>, NumberType>>* items,
                     size_t N) {
    auto& tensor = *items;

    // Фаза 1: сортировка по s
    std::vector<int> count(N * N - 1, 0);
    for (const auto& item : tensor) {
        count[std::get<2>(item.first)]++;
    }

    // Префиксные суммы для s
    for (size_t i = 1; i < count.size(); i++) {
        count[i] += count[i - 1];
    }

    std::vector<std::pair<std::tuple<int, int, int>, double>> temp_tensor(tensor.size());
    // Распределение в обратном порядке для stable сортировки
    for (ptrdiff_t i = tensor.size() - 1; i >= 0; i--) {
        const auto& item = tensor[i];
        int pos = --count[std::get<2>(item.first)];
        temp_tensor[pos] = tensor[i];
    }

    // Фаза 2: сортировка по n
    std::fill(count.begin(), count.end(), 0);
    for (const auto& item : temp_tensor) {
        count[std::get<1>(item.first)]++;
    }

    for (size_t i = 1; i < count.size(); i++) {
        count[i] += count[i - 1];
    }

    for (ptrdiff_t i = temp_tensor.size() - 1; i >= 0; i--) {
        const auto& item = temp_tensor[i];
        int pos = --count[std::get<1>(item.first)];
        tensor[pos] = temp_tensor[i];
    }

    // Фаза 3: сортировка по m
    std::fill(count.begin(), count.end(), 0);
    for (const auto& item : tensor) {
        count[std::get<0>(item.first)]++;
    }

    for (size_t i = 1; i < count.size(); i++) {
        count[i] += count[i - 1];
    }

    for (ptrdiff_t i = temp_tensor.size() - 1; i >= 0; i--) {
        const auto& item = tensor[i];
        int pos = --count[std::get<0>(item.first)];
        temp_tensor[pos] = item;
    }
    tensor = std::move(temp_tensor);
}

void CountSortTensorSN(std::vector<std::pair<std::tuple<int, int, int>, double>>* items, size_t N) {
    auto& tensor = *items;

    // Фаза 1: сортировка по n
    std::vector<int> count(N * N - 1, 0);
    for (const auto& item : tensor) {
        count[std::get<1>(item.first)]++;
    }

    // Префиксные суммы для n
    for (size_t i = 1; i < count.size(); i++) {
        count[i] += count[i - 1];
    }

    std::vector<std::pair<std::tuple<int, int, int>, double>> temp_tensor(tensor.size());
    // Распределение в обратном порядке для stable сортировки
    for (ptrdiff_t i = tensor.size() - 1; i >= 0; i--) {
        const auto& item = tensor[i];
        int pos = --count[std::get<1>(item.first)];
        temp_tensor[pos] = tensor[i];
    }

    // Фаза 2: сортировка по s
    std::fill(count.begin(), count.end(), 0);
    for (const auto& item : temp_tensor) {
        count[std::get<2>(item.first)]++;
    }

    for (size_t i = 1; i < count.size(); i++) {
        count[i] += count[i - 1];
    }

    for (ptrdiff_t i = temp_tensor.size() - 1; i >= 0; i--) {
        const auto& item = temp_tensor[i];
        int pos = --count[std::get<2>(item.first)];
        tensor[pos] = temp_tensor[i];
    }
}

template <class NumberType>
void CountSortTensorNS(std::vector<std::pair<std::tuple<int, int, int>, NumberType>>* items, size_t N) {
    auto& tensor = *items;

    // Фаза 1: сортировка по s
    std::vector<int> count(N * N - 1, 0);
    for (const auto& item : tensor) {
        count[std::get<2>(item.first)]++;
    }

    // Префиксные суммы для s
    for (size_t i = 1; i < count.size(); i++) {
        count[i] += count[i - 1];
    }

    std::vector<std::pair<std::tuple<int, int, int>, NumberType>> temp_tensor(tensor.size());
    // Распределение в обратном порядке для stable сортировки
    for (ptrdiff_t i = tensor.size() - 1; i >= 0; i--) {
        const auto& item = tensor[i];
        int pos = --count[std::get<2>(item.first)];
        temp_tensor[pos] = tensor[i];
    }

    // Фаза 2: сортировка по n
    std::fill(count.begin(), count.end(), 0);
    for (const auto& item : temp_tensor) {
        count[std::get<1>(item.first)]++;
    }

    for (size_t i = 1; i < count.size(); i++) {
        count[i] += count[i - 1];
    }

    for (ptrdiff_t i = temp_tensor.size() - 1; i >= 0; i--) {
        const auto& item = temp_tensor[i];
        int pos = --count[std::get<1>(item.first)];
        tensor[pos] = temp_tensor[i];
    }
}