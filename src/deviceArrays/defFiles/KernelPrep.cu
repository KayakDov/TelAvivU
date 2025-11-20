#include "deviceArrays/headers/KernelPrep.cuh"
// --- Private Calculation Methods ---

void KernelPrep::calculate1D(size_t count) {
    if (count == 0) {
        blockDim = dim3(0, 0, 0);
        gridDim = dim3(0, 0, 0);
        return;
    }
    
    // Set block size for 1D
    blockDim = dim3(DEFAULT_BLOCK_SIZE_1D, 1, 1);
    
    // Calculate required grid size
    gridDim = dim3(ceiling_division(count, DEFAULT_BLOCK_SIZE_1D), 1, 1);
}

void KernelPrep::calculate2D(size_t width, size_t height) {
    if (width == 0 || height == 0) {
        blockDim = dim3(0, 0, 0);
        gridDim = dim3(0, 0, 0);
        return;
    }
    
    // Use a square block size (16x16 = 256 threads)
    blockDim = dim3(DEFAULT_BLOCK_SIZE_2D_SIDE, DEFAULT_BLOCK_SIZE_2D_SIDE, 1);
    
    // Calculate required grid size
    gridDim = dim3(
        ceiling_division(width, DEFAULT_BLOCK_SIZE_2D_SIDE),
        ceiling_division(height, DEFAULT_BLOCK_SIZE_2D_SIDE),
        1
    );
}

void KernelPrep::calculate3D(size_t width, size_t height, size_t depth) {
    if (width == 0 || height == 0 || depth == 0) {
        blockDim = dim3(0, 0, 0);
        gridDim = dim3(0, 0, 0);
        return;
    }

    constexpr size_t SIDE = 8;
    
    blockDim = dim3(SIDE, SIDE, SIDE);

    gridDim = dim3(
        ceiling_division(width, SIDE),
        ceiling_division(height, SIDE),
        ceiling_division(depth, SIDE)
    );
}

// --- Public Constructors (Delegating to private helpers) ---

KernelPrep::KernelPrep(size_t count) {
    calculate1D(count);
}

KernelPrep::KernelPrep(size_t width, size_t height) {
    calculate2D(width, height);
}

KernelPrep::KernelPrep(size_t width, size_t height, size_t depth) {
    calculate3D(width, height, depth);
}