
#ifndef CUDABANDED_DIRECTSOLVER_CUH
#define CUDABANDED_DIRECTSOLVER_CUH

#include "PoissonRHS.h"
#include "deviceArrays/headers/BandedMat.h"
#include "solvers/BiCGSTAB.cuh"


template <typename T>
class DirectSolver : public PoissonRHS<T> {


private:
    const BandedMat<T> A;

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