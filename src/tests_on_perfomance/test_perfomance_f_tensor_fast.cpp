#include <iostream>

#include "lindblad_utils.h"
#include "timer.h"

int main(int argc, char* argv[]) {
    // Параметры
    int N = 4;
    // на случай если передаем размер N в параметрах командной строки
    if (argc == 2) {
        N = std::atoi(argv[1]);
    }

    RAMMeter meter;
    Timer timer("");

    std::vector<std::pair<std::tuple<int, int, int>, double>> arr = GenerateTensorF(N);

    timer.stop();
    meter.tick();
    std::cout << "\n";
    std::cout << arr[1].second << "\n\n";
}