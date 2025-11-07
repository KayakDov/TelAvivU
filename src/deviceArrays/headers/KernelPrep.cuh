#ifndef KERNELPREP_H
#define KERNELPREP_H

#include <cuda_runtime.h>
#include <stdexcept>

/**
 * @brief Utility class to calculate optimal CUDA grid and block dimensions for kernel launches.
 * * Provides methods to set up dim3 structures for 1D, 2D, and 3D data processing.
 */
class KernelPrep {
public:
    // --- Public Fields ---

    /** @brief The calculated grid dimensions (number of blocks). */
    dim3 gridDim;

    /** @brief The calculated block dimensions (threads per block). */
    dim3 blockDim;

    // --- Constructors / Configuration Methods ---

    /**
     * @brief Constructs a KernelPrep object and calculates 1D grid/block dimensions.
     * @param count The total number of elements to process.
     */
    explicit KernelPrep(size_t count);

    /**
     * @brief Constructs a KernelPrep object and calculates 2D grid/block dimensions.
     * @param width The width (X dimension) of the data.
     * @param height The height (Y dimension) of the data.
     */
    KernelPrep(size_t width, size_t height);

    /**
     * @brief Constructs a KernelPrep object and calculates 3D grid/block dimensions.
     * @param width The width (X dimension) of the data.
     * @param height The height (Y dimension) of the data.
     * @param depth The depth (Z dimension) of the data.
     */
    KernelPrep(size_t width, size_t height, size_t depth);

    /**
     * @brief Utility function to perform ceiling division (a / b).
     * @param numerator The total size (a).
     * @param denominator The block size (b).
     * @return The number of blocks needed (ceil(a / b)).
     */
    static size_t ceiling_division(size_t numerator, size_t denominator) {
        return (numerator + denominator - 1) / denominator;
    }

private:
    // --- Private Helper Constants and Methods ---

    // Default threads per block for 1D operations. Must be power of 2.
    static constexpr size_t DEFAULT_BLOCK_SIZE_1D = 256;

    // Default threads per side for 2D/3D blocks. 16*16 = 256 threads.
    static constexpr size_t DEFAULT_BLOCK_SIZE_2D_SIDE = 16;

    /**
     * @brief Core calculation logic for 1D dimensions.
     * @param count The total number of elements.
     */
    void calculate1D(size_t count);

    /**
     * @brief Core calculation logic for 2D dimensions.
     * @param width The X dimension.
     * @param height The Y dimension.
     */
    void calculate2D(size_t width, size_t height);

    /**
     * @brief Core calculation logic for 3D dimensions.
     * @param width The X dimension.
     * @param height The Y dimension.
     * @param depth The Z dimension.
     */
    void calculate3D(size_t width, size_t height, size_t depth);
};

#endif // KERNELPREP_H
