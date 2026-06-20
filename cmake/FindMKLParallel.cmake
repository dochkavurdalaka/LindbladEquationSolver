# cmake/FindMKL.cmake
# Находит Intel Math Kernel Library (MKL) с многопоточной поддержкой

# Возможные пути установки
set(MKL_SEARCH_PATHS
    /opt/intel/oneapi/mkl/latest
    /opt/intel/mkl
    /usr/local/mkl
    $ENV{MKLROOT}
    $ENV{MKL_ROOT}
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
    "C:/Program Files (x86)/Intel/mkl"
)

# Поиск заголовочных файлов
find_path(MKL_INCLUDE_DIR
    NAMES mkl.h
    PATHS /opt/intel/oneapi/mkl/latest/include
          /opt/intel/mkl/include
          $ENV{MKLROOT}/include
)

# Поиск библиотек для параллельной версии
find_library(MKL_CORE_LIB mkl_core
    PATHS /opt/intel/oneapi/mkl/latest/lib/intel64
          /opt/intel/mkl/lib/intel64
)

find_library(MKL_INTEL_LP64_LIB mkl_intel_lp64
    PATHS /opt/intel/oneapi/mkl/latest/lib/intel64
          /opt/intel/mkl/lib/intel64
)

# Выбор threading библиотеки в зависимости от компилятора
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    find_library(MKL_THREAD_LIB mkl_intel_thread
        PATHS /opt/intel/oneapi/mkl/latest/lib/intel64
              /opt/intel/mkl/lib/intel64
    )
else()
    # Для GCC, Clang и других компиляторов используем GNU OpenMP
    find_library(MKL_THREAD_LIB mkl_gnu_thread
        PATHS /opt/intel/oneapi/mkl/latest/lib/intel64
              /opt/intel/mkl/lib/intel64
    )
endif()

# Поиск OpenMP библиотеки
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    find_library(OPENMP_LIB iomp5
        PATHS /opt/intel/oneapi/compiler/latest/lib/intel64
              /opt/intel/compiler/lib/intel64
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    find_library(OPENMP_LIB gomp
        PATHS /usr/lib/x86_64-linux-gnu
              /usr/lib64
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    find_library(OPENMP_LIB omp
        PATHS /usr/lib/x86_64-linux-gnu
              /usr/lib64
    )
endif()

if(MKL_INCLUDE_DIR AND MKL_CORE_LIB AND MKL_INTEL_LP64_LIB AND MKL_THREAD_LIB)
    set(MKL_FOUND TRUE)
    
    if(NOT TARGET MKL::MKL)
        add_library(MKL::MKL INTERFACE IMPORTED)
        target_include_directories(MKL::MKL INTERFACE ${MKL_INCLUDE_DIR})
        
        # Собираем список библиотек для линковки
        set(MKL_LIBRARIES
            ${MKL_INTEL_LP64_LIB}
            ${MKL_THREAD_LIB}
            ${MKL_CORE_LIB}
        )
        
        # Добавляем OpenMP библиотеку
        if(OPENMP_LIB)
            list(APPEND MKL_LIBRARIES ${OPENMP_LIB})
        endif()
        
        # Добавляем системные библиотеки
        list(APPEND MKL_LIBRARIES pthread m dl)
        
        target_link_libraries(MKL::MKL INTERFACE ${MKL_LIBRARIES})
        
        # Добавляем определение для использования OpenMP в коде
        target_compile_definitions(MKL::MKL INTERFACE MKL_USE_OPENMP)
    endif()
    
    message(STATUS "Found MKL (parallel): ${MKL_INCLUDE_DIR}")
    message(STATUS "  Threading library: ${MKL_THREAD_LIB}")
    if(OPENMP_LIB)
        message(STATUS "  OpenMP library: ${OPENMP_LIB}")
    endif()
else()
    message(FATAL_ERROR "MKL not found. Install: sudo apt install intel-oneapi-mkl-devel")
endif()