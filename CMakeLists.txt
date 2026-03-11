cmake_minimum_required(VERSION 3.15)
project(MKLProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Включаем папку include для всех таргетов
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Определяем список программ
set(PROGRAMS hardcore original main main_dimer_test main_dimer main_with_lindbladians main_RK_control test_commutator test_f_tensor test_anticommutator test_d_tensor test_q_tensor q_tensor_experiments test_r_tensor test_k_tensor)

# Добавляем путь к нашим модулям CMake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Поиск MKL
find_package(MKL REQUIRED)


# Функция для настройки санитайзеров
function(setup_sanitizers target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Asan>:-g>
            $<$<CONFIG:Asan>:-fsanitize=address>
            $<$<CONFIG:Asan>:-fsanitize=undefined>
            $<$<CONFIG:Asan>:-fno-omit-frame-pointer>
            $<$<CONFIG:Asan>:-O2>
        )
        
        # Критически важно: добавляем флаги санитайзеров и при линковке!
        target_link_options(${target} PRIVATE
            $<$<CONFIG:Asan>:-fsanitize=address>
            $<$<CONFIG:Asan>:-fsanitize=undefined>
        )
    endif()
endfunction()

# Создаем каждую программу
foreach(program IN LISTS PROGRAMS)
    add_executable(${program} src/${program}.cpp)
    
    # Общие флаги (передаем отдельно!)
    target_compile_options(${program} PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
    
    # Конфигурационные флаги для Debug (отдельные флаги)
    target_compile_options(${program} PRIVATE
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-Og>
        $<$<CONFIG:Debug>:-DDEBUG>
    )
    
    # Конфигурационные флаги для Release (отдельные флаги)
    target_compile_options(${program} PRIVATE
        $<$<CONFIG:Release>:-O2>
        $<$<CONFIG:Release>:-march=native>
    )

    # Конфигурационные флаги для Asan (отдельные флаги)
    target_compile_options(${program} PRIVATE
        $<$<CONFIG:Asan>:-g>
        $<$<CONFIG:Asan>:-fsanitize=address>
        $<$<CONFIG:Asan>:-fsanitize=undefined>
        $<$<CONFIG:Asan>:-O2>
    )
    
    setup_sanitizers(${program})

    # Связывание с MKL и OpenMP
    target_link_libraries(${program} 
        MKL::MKL
    )
    
    # Организация выходных файлов
    set_target_properties(${program} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
    )
endforeach()