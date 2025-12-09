
#ifndef CUDABANDED_DIRECTSOLVER_CUH
#define CUDABANDED_DIRECTSOLVER_CUH

#include "Poisson.h"
#include "deviceArrays/headers/BandedMat.h"
#include "BiCGSTAB/BiCGSTAB.cuh"

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
class DirectSolver : public Poisson<T> {

public:

    const AdjacencyInd here, up, down, left, right, back, front;

private:
    const BandedMat<T> A;

    /**
     * @brief Launch kernel that assembles A in banded/dense storage.
     *
     * @param numInds The number of indices.
     * @param mindices device pointer to the int32_t offsets array (length numNonZeroDiags).
     * @param handle contains the stream to run on.
     * @param preAlocatedForA Provide prealocated memory here to be written to, numDiagonals x _b.size().
     */
    BandedMat<T> setA(cudaStream_t& stream, Mat<T>& preAlocatedForA, Vec<int32_t>& preAlocatedForIndices);

    void loadMapRowToDiag(Vec<int32_t> diags, const cudaStream_t stream) const ;
public:
    /**
    * @brief Constructs the PoissonFDM solver object.
    *
    * Initializes the boundary condition matrices and the dimensions of the interior grid.
    * It assumes the RHS vector $\mathbf{b}$ is pre-loaded with the source term $f$.
    *
    * @param boundary The boundary conditions.
    * @param[in] f The initial right-hand side vector, pre-loaded with the source term $f$.  This will be overwritten.
    * This vector is modified by the solver to include boundary contributions.
    * @param prealocatedForIndices
    */
    DirectSolver(const CubeBoundary<T>& boundary, Vec<T>& f, Mat<T>& preAlocatedForBandedA, Vec<int32_t>& prealocatedForIndices, cudaStream_t stream);

    /**
     * @brief Solves the Poisson equation for the grid.
     *
     * Automatically dispatches to the 2D or 3D solver based on whether the number of layers is 1.
     *
     * @param[out] x Pre-allocated memory that the solution will be written to.
     * @param A Pre-allocated memory that will be used to compute the solution.  It should be numDiagonals rows and _b.size() columns.
     * @param[in] hand The CUDA handle (stream/context) to manage the computation.
     */
    void solve(Mat<T> prealocatedForBiCGSTAB);
};

void testPoisson(size_t dim, Handle& hand);
void testPoisson(size_t height, size_t width, size_t depth, Handle& hand);


#endif //CUDABANDED_DIRECTSOLVER_CUH