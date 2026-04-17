cmake_minimum_required(VERSION 3.15)
project(MKLProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Включаем папку include для всех таргетов
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


file(GLOB TEST_SOURCES "src/*.cpp")

# Преобразуем пути в имена программ
set(PROGRAMS "")
foreach(source_file ${TEST_SOURCES})
    get_filename_component(program_name ${source_file} NAME_WE)
    list(APPEND PROGRAMS ${program_name})
endforeach()

# Определяем список программ
# set(PROGRAMS test_perfomance_f_tensor_slow)

# Добавляем путь к нашим модулям CMake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Поиск OpenMP
find_package(OpenMP REQUIRED)


# find_package(MKLParallel REQUIRED)
find_package(MKLSequential REQUIRED)

set(MKL_TARGET MKL::MKL)


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

    # Определяем путь к исходному файлу
    set(SOURCE_FILE "")
    
    # Проверяем наличие файла в разных директориях
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/${program}.cpp)
        set(SOURCE_FILE src/${program}.cpp)
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/tests_on_correctness/${program}.cpp)
        set(SOURCE_FILE src/tests_on_correctness/${program}.cpp)
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/tests_on_perfomance/${program}.cpp)
        set(SOURCE_FILE src/tests_on_perfomance/${program}.cpp)
    else()
        message(WARNING "Source file for ${program} not found in src/ or src/tests_on_correctness/ or src/tests_on_perfomance/")
        continue()  # Пропускаем эту программу
    endif()

    add_executable(${program} ${SOURCE_FILE})
    
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
        ${MKL_TARGET}
        OpenMP::OpenMP_CXX
    )
    
    # Организация выходных файлов
    set_target_properties(${program} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
    )
endforeach()