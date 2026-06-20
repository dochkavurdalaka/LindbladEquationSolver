#include <iostream>
#include <vector>
#include <format>
#include <fstream>

#include "lindblad_utils.h"

int main() {
    // Параметры
    int N = 5;

    auto arr = GenerateTensorD(N);
    char name = 'd';

    std::ofstream file(std::format("{}_tensor_file{}.txt", name, N));
    if (!file.is_open()) {
        std::cerr << std::format("Cannot open file: {}_tensor_file{}.txt", name, N) << std::endl;
        return 0;
    }
    
    for (const auto& [a, _] : arr) {
        const auto& [m, n, s] = a;
        file << m << " " << n << " " << s << "\n";
    }
    
    file.close();
}