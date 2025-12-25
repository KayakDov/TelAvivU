//
// Created by usr on 12/24/25.
//

#ifndef CUDABANDED_POISSONLHS_H
#define CUDABANDED_POISSONLHS_H
#include <array>
#include <cuda_runtime.h>
#include "BandedMat.h"

constexpr  size_t numDiagonals = 7;

struct AdjacencyInd {
    /**
     * The column in the banded matrix.
     */
    const size_t col;
    /**
     * The index of the diagonal that is held by that column.
     */
    const int32_t diag;
    __device__ __host__ AdjacencyInd(const size_t col, const int32_t diag) : col(col), diag(diag) {
    }
};

template <typename T>
class ToeplitzLaplacian{
    const std::array<AdjacencyInd, numDiagonals> adjInds;//here, up, down, left, right, back, front;
    const GridDim dim;

    void loadMapRowToDiag(Vec<int32_t> diags, cudaStream_t stream);
public:
    /**
     * Creates the LHS matrix of the linear system used for solving the Poisson equation.
     * @param dim The dimensions of the Poisson grid.
     */
    ToeplitzLaplacian(GridDim dim);

    /**
     * @brief Launch kernel that assembles A in banded/dense storage.
     *
     * @param numInds The number of indices.
     * @param mindices device pointer to the int32_t offsets array (length numNonZeroDiags).
     * @param handle contains the stream to run on.
     * @param preAlocatedForA Provide prealocated memory here to be written to, numDiagonals x _b.size().
     * @param dim
     */
    BandedMat<T> setA(cudaStream_t stream, Mat<T> &preAlocatedForA, Vec<int32_t> &preAlocatedForIndices);
};

#endif //CUDABANDED_POISSONLHS_H