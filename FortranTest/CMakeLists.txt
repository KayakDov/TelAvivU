cmake_minimum_required(VERSION 3.18)

# --- Force NVHPC Fortran compiler early ---
set(CMAKE_Fortran_COMPILER "/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nvfortran" CACHE FILEPATH "Fortran compiler" FORCE)

project(FortranCudaProject LANGUAGES Fortran CXX)

# Where Fortran modules go
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod)

# Paths to Shroud wrapper + static C++ library
set(SHROUD_WRAPF "${CMAKE_SOURCE_DIR}~/Documents/CudaBanded/TelAvivU/shroud_project/wrapffortranbindings.f90")
set(CUDA_LIB "${CMAKE_SOURCE_DIR}/../TelAvivU/cmake-build-debug/libCudaBandedLib.a")

if(NOT EXISTS "${SHROUD_WRAPF}")
    message(FATAL_ERROR "Cannot find Shroud wrapper: ${SHROUD_WRAPF}")
endif()

if(NOT EXISTS "${CUDA_LIB}")
    message(FATAL_ERROR "Cannot find CUDA static library: ${CUDA_LIB}")
endif()

# Build the Fortran executable
add_executable(fortran_app
        main.f90
        ${SHROUD_WRAPF}
)

# --- NVHPC Fortran + CUDA compile options ---
# This enables device code compilation
target_compile_options(fortran_app PRIVATE -cuda -Minfo)

# Include NVHPC + CUDA directories
target_include_directories(fortran_app PRIVATE
        /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/include
        /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/include/cuda
        /usr/local/cuda/targets/x86_64-linux/include
)

# Link libraries
find_package(CUDAToolkit REQUIRED)

# Use nvfortran as linker so device runtime symbols are resolved
set_target_properties(fortran_app PROPERTIES LINKER_LANGUAGE Fortran)

# Link C++ static library and CUDA runtimes after Fortran objects
target_link_libraries(fortran_app PRIVATE
        ${CUDA_LIB}          # your C++ static library
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        stdc++
        # REMOVE THE MISSING LIBRARY LINE HERE
)

# Use LINK_OPTIONS to implicitly include the Fortran CUDA runtime (equivalent to -lcudafor)
target_link_options(fortran_app PRIVATE -cudalib)

# Disable PIE to avoid relocation errors
set_target_properties(fortran_app PROPERTIES POSITION_INDEPENDENT_CODE OFF)
