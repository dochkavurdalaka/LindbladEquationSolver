#include <iostream>
#include <mkl.h>
#include <omp.h>
#include <chrono>
#include <vector>
#include <iomanip>

class MKLThreadChecker {
private:
    int mkl_version;
    bool is_parallel;
    int max_threads;
    
public:
    MKLThreadChecker() {
        check_mkl_version();
        check_threading_mode();
    }
    
    void check_mkl_version() {
        MKLVersion version;
        mkl_get_version(&version);
        
        std::cout << "=== Intel MKL Version Information ===" << std::endl;
        std::cout << "Major version: " << version.MajorVersion << std::endl;
        std::cout << "Minor version: " << version.MinorVersion << std::endl;
        std::cout << "Update version: " << version.UpdateVersion << std::endl;
        std::cout << "Product status: " << version.ProductStatus << std::endl;
        std::cout << "Processor: " << version.Processor << std::endl;
        std::cout << std::endl;
    }
    
    void check_threading_mode() {
        std::cout << "=== MKL Threading Configuration ===" << std::endl;
        
        // Проверяем, какая версия MKL используется
        #ifdef MKL_ILP64
            std::cout << "MKL Interface: ILP64 (64-bit integers)" << std::endl;
        #else
            std::cout << "MKL Interface: LP64 (32-bit integers)" << std::endl;
        #endif
        
        #ifdef MKL_DIRECT_CALL
            std::cout << "MKL Direct Call: Enabled" << std::endl;
        #endif
        
        // Проверяем, какая библиотека потоков используется
        std::cout << "MKL Threading Layer: ";
        
        // Проверяем через переменные окружения
        const char* mkl_threading = std::getenv("MKL_THREADING_LAYER");
        if (mkl_threading) {
            std::cout << mkl_threading << " (from environment)" << std::endl;
        } else {
            // Пытаемся определить
            #ifdef _OPENMP
                std::cout << "Intel OpenMP (likely)" << std::endl;
            #else
                std::cout << "Sequential (no threading)" << std::endl;
            #endif
        }
        
        // Количество потоков
        max_threads = mkl_get_max_threads();
        std::cout << "MKL max threads: " << max_threads << std::endl;
        std::cout << "MKL dynamic adjustment: " << (mkl_get_dynamic() ? "Enabled" : "Disabled") << std::endl;
        std::cout << std::endl;
    }
    
    void check_library_links() {
        std::cout << "=== Library Linkage Check ===" << std::endl;
        
        #ifdef __INTEL_COMPILER
            std::cout << "Compiler: Intel Compiler" << std::endl;
        #elif defined(__GNUC__)
            std::cout << "Compiler: GCC" << std::endl;
        #elif defined(_MSC_VER)
            std::cout << "Compiler: MSVC" << std::endl;
        #endif
        
        #ifdef _OPENMP
            std::cout << "OpenMP: Enabled (version " << _OPENMP << ")" << std::endl;
            std::cout << "OpenMP max threads: " << omp_get_max_threads() << std::endl;
        #else
            std::cout << "OpenMP: Disabled" << std::endl;
        #endif
        
        std::cout << std::endl;
    }
    
    void print_recommendations() {
        if (!is_parallel) {
            std::cout << "=== Recommendations for enabling MKL parallelism ===" << std::endl;
            std::cout << "1. Link with parallel MKL library:" << std::endl;
            std::cout << "   - Use: -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core" << std::endl;
            std::cout << "   - Instead of: -lmkl_intel_lp64 -lmkl_sequential -lmkl_core" << std::endl;
            std::cout << std::endl;
            std::cout << "2. Set environment variables:" << std::endl;
            std::cout << "   export MKL_NUM_THREADS=<num_cores>" << std::endl;
            std::cout << "   export OMP_NUM_THREADS=<num_cores>" << std::endl;
            std::cout << std::endl;
            std::cout << "3. In CMake, use:" << std::endl;
            std::cout << "   find_package(MKL REQUIRED)" << std::endl;
            std::cout << "   target_link_libraries(${target} MKL::MKL)" << std::endl;
            std::cout << "   (not MKL::MKLSequential)" << std::endl;
        }
    }
    
    bool is_multithreaded() const {
        return is_parallel;
    }
};

// 2. Простой тест для быстрой проверки
void quick_mkl_test() {
    std::cout << "\n=== Quick MKL Threading Test ===" << std::endl;
    
    const int N = 3000;
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Инициализация
    for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0;
        B[i] = 1.0;
    }
    
    // Сохраняем текущие настройки
    int original_threads = mkl_get_max_threads();
    
    // Тест с 1 потоком
    mkl_set_num_threads(1);
    auto start = std::chrono::high_resolution_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, A.data(), N, B.data(), N, 0.0, C.data(), N);
    auto end = std::chrono::high_resolution_clock::now();
    double time_1 = std::chrono::duration<double>(end - start).count();
    
    // Тест с максимальным количеством потоков
    int max_threads = original_threads;
    mkl_set_num_threads(max_threads);
    start = std::chrono::high_resolution_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, A.data(), N, B.data(), N, 0.0, C.data(), N);
    end = std::chrono::high_resolution_clock::now();
    double time_n = std::chrono::duration<double>(end - start).count();
    
    // Восстанавливаем настройки
    mkl_set_num_threads(original_threads);
    
    std::cout << "Time with 1 thread: " << time_1 << " seconds" << std::endl;
    std::cout << "Time with " << max_threads << " threads: " << time_n << " seconds" << std::endl;
    std::cout << "Speedup: " << (time_1 / time_n) << "x" << std::endl;
    
    if (time_n < time_1 * 0.8) {  // Если ускорение > 20%
        std::cout << "✅ MKL is using multiple threads!" << std::endl;
    } else {
        std::cout << "❌ MKL appears to be running in sequential mode!" << std::endl;
    }
    std::cout << std::endl;
}

// 3. Проверка, какая библиотека слинкована
void check_mkl_library_type() {
    std::cout << "=== MKL Library Type Detection ===" << std::endl;
    
    // Проверяем через функцию mkl_get_max_threads
    // В sequential версии эта функция всегда возвращает 1
    int max_threads = mkl_get_max_threads();
    
    if (max_threads == 1) {
        std::cout << "⚠️  mkl_get_max_threads() returns 1" << std::endl;
        std::cout << "This suggests you might be using MKL sequential version" << std::endl;
    } else {
        std::cout << "mkl_get_max_threads() returns " << max_threads << std::endl;
        std::cout << "MKL parallel version is likely linked" << std::endl;
    }
    
    // Пытаемся определить через переменные окружения
    const char* mkl_threading = std::getenv("MKL_THREADING_LAYER");
    if (mkl_threading) {
        std::cout << "MKL_THREADING_LAYER=" << mkl_threading << std::endl;
        if (std::string(mkl_threading) == "sequential") {
            std::cout << "❌ MKL is forced to sequential mode via environment" << std::endl;
        }
    }
    
    std::cout << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   Intel MKL Multithreading Checker" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    MKLThreadChecker checker;
    checker.check_library_links();
    checker.check_mkl_version();
    checker.check_threading_mode();
    checker.print_recommendations();
    
    quick_mkl_test();
    check_mkl_library_type();
    
    
    return 0;
}