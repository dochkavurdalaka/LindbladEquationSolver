#include <iostream>
#include <chrono>
#include <unistd.h>
#include <fstream>

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::string name;

public:
    Timer(const std::string& timer_name = "Timer") : name(timer_name) {
        start();
    }

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << name << ": " << duration.count() << " мкс" << std::endl;
    }

    long long elapsed_microseconds() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    }
};

class RAMMeter {
private:
    double vsz_current_memory;
    double rss_current_memory;
    size_t page_size;

public:
    RAMMeter() {
        size_t vsz_pages = 0;
        size_t rss_pages = 0;
        std::ifstream statmFile("/proc/self/statm");

        // Первое число пропускаем, второе сохраняем в rssPages
        statmFile >> vsz_pages >> rss_pages;
        statmFile.close();

        // Размер страницы в Linux обычно 4096 байт. Узнаем его точно через sysconf()
        page_size = sysconf(_SC_PAGESIZE);

        size_t memory_usage_bytes = rss_pages * page_size;
        rss_current_memory = memory_usage_bytes / (1024.0 * 1024);
        memory_usage_bytes = vsz_pages * page_size;
        vsz_current_memory = memory_usage_bytes / (1024.0 * 1024);

        std::cout << "Используется RAM (VSZ, RSS): " << vsz_current_memory << " MB, " << rss_current_memory << " MB" << std::endl;
    }

    void tick() {
        size_t vsz_pages = 0;
        size_t rss_pages = 0;
        std::ifstream statmFile("/proc/self/statm");

        // Первое число пропускаем, второе сохраняем в rssPages
        statmFile >> vsz_pages >> rss_pages;
        statmFile.close();

        size_t memory_usage_bytes = rss_pages * page_size;
        double rss_memory_usage_mb = memory_usage_bytes / (1024.0 * 1024);
        memory_usage_bytes = vsz_pages * page_size;
        double vsz_memory_usage_mb = memory_usage_bytes / (1024.0 * 1024);


        std::cout << "Используется RAM (VSZ, RSS): " << vsz_memory_usage_mb << " MB, " << rss_memory_usage_mb << " MB" << std::endl;
        std::cout << "Выделилось новой RAM с момента прошлого вызова RAMMeter (VSZ, RSS): " << vsz_memory_usage_mb - vsz_current_memory << " MB, " << rss_memory_usage_mb - rss_current_memory << " MB" << std::endl;
        rss_current_memory = rss_memory_usage_mb;
        vsz_current_memory = vsz_memory_usage_mb;
    }

};