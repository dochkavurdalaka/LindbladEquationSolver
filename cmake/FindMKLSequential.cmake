# cmake/FindMKL.cmake
# Находит Intel Math Kernel Library (MKL)

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

# cmake/FindMKL.cmake - sequential версия
find_path(MKL_INCLUDE_DIR
    NAMES mkl.h
    PATHS /opt/intel/oneapi/mkl/latest/include
          /opt/intel/mkl/include
          $ENV{MKLROOT}/include
)

# Ищем sequential библиотеки (без threading)
find_library(MKL_CORE_LIB mkl_core
    PATHS /opt/intel/oneapi/mkl/latest/lib/intel64
          /opt/intel/mkl/lib/intel64
)

find_library(MKL_INTEL_LP64_LIB mkl_intel_lp64
    PATHS /opt/intel/oneapi/mkl/latest/lib/intel64
          /opt/intel/mkl/lib/intel64
)

find_library(MKL_SEQUENTIAL_LIB mkl_sequential
    PATHS /opt/intel/oneapi/mkl/latest/lib/intel64
          /opt/intel/mkl/lib/intel64
)

if(MKL_INCLUDE_DIR AND MKL_CORE_LIB AND MKL_INTEL_LP64_LIB AND MKL_SEQUENTIAL_LIB)
    set(MKL_FOUND TRUE)
    
    if(NOT TARGET MKL::MKL)
        add_library(MKL::MKL INTERFACE IMPORTED)
        target_include_directories(MKL::MKL INTERFACE ${MKL_INCLUDE_DIR})
        target_link_libraries(MKL::MKL INTERFACE
            ${MKL_INTEL_LP64_LIB}
            ${MKL_SEQUENTIAL_LIB}
            ${MKL_CORE_LIB}
            pthread m dl
        )
    endif()
    
    message(STATUS "Found MKL (sequential): ${MKL_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "MKL not found. Install: sudo apt install intel-oneapi-mkl-devel")
endif()