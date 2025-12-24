
#include "DirectSolver.cuh"
//TODO: Combine this class  with RunDirectSolver class, and then make into an hpp or split into .cuh and .cu.

 /**
 * @brief Device-side functor to set off-diagonal entries of the system matrix A to 0 or NAN.
 *
 * This is used inside the setAKernel3d kernel to handle the six neighbors for each interior
 * grid point. It ensures that entries corresponding to boundary connections are either
 * correctly set to 0 (non-existent internal connection) or marked as NAN (which typically
 * signals an element outside the valid band storage).
 *
 * @tparam T Floating-point type (float or double).
 */
template <typename T>
class Set0 {
private:
    DeviceData2d<T>& a;
    const size_t idGrid;
public:
    /**
     * @brief Constructs the Set0 functor.
     *
     * @param[in,out] a Pointer to the banded matrix data on the device.
     * @param[in] idGrid The flat index of the current grid point (row in A).
     */
    __device__ Set0(DeviceData2d<T>& a, const size_t idGrid) :
        a(a), idGrid(idGrid) {}

    /**
     * @brief Sets the corresponding off-diagonal entry to 0 or NAN based on boundary condition logic.
     *
     * This operator is called to check if a specific off-diagonal entry (corresponding to a neighbor)
     * should be set to 0 (internal point) or NAN (outside band storage).
     *
     * @param[in] aInd The index of the diagonal corresponding to the neighbor being checked.
     */
    __device__ void operator()(const AdjacencyInd aInd) {

        const size_t rowInd = modPos(static_cast<int32_t>(idGrid) + min(aInd.diag, 0), static_cast<int32_t>(a.rows));
        if (rowInd < a.rows - abs(aInd.diag)) a(rowInd, aInd.col) = static_cast<T>(0);
        else a(rowInd, aInd.col) = NAN;
    }
};

/**
 * @brief CUDA kernel to set up the system matrix A for the 3D Poisson FDM problem.
 *
 * Each thread handles one unknown point $(gRow, gCol, gLayer)$ in the interior grid,
 * setting the main diagonal entry ($A_{i,i} = -6$) and using the Set0 functor to handle
 * the 6 off-diagonal entries (neighbors) and enforce boundary conditions by setting
 * unused band elements to NAN.
 *
 * @tparam T Floating-point type (float or double).
 *
 */
template <typename T>
__global__ void setAKernel(DeviceData2d<T> a,
    const GridDim g,
    const AdjacencyInd here, const AdjacencyInd up, const AdjacencyInd down, const AdjacencyInd left,
    const AdjacencyInd right, const AdjacencyInd front, const AdjacencyInd back
    ) {
    const GridInd3d ind;

    if (ind >= g) return;

    const size_t idGrid = g[ind];

    a(idGrid, here.col) = -6;
    Set0<T> set0(a, idGrid);

    if (ind.row == 0) set0(up);
    else if (ind.row == g.rows - 1) set0(down);

    if (ind.col == 0) set0(left);
    else if (ind.col == g.cols - 1) set0(right);

    if (ind.layer == 0) set0(front);
    else if (ind.layer == g.layers - 1) set0(back);
}

/**
 * @brief Solves the 3D Poisson equation $\nabla^2 u = f$ using the Finite Difference Method (FDM)
 * and the BiCGSTAB iterative solver on the resulting linear system $A\mathbf{x} = \mathbf{b}$.
 *
 * This class handles the construction of the FDM linear system for a 3D grid,
 * including setting up the system matrix $A$ (as a banded matrix), calculating
 * the right-hand side vector $\mathbf{b}$ (incorporating boundary conditions),
 * and leveraging the BiCGSTAB solver for the solution. The class assumes a uniform
 * grid spacing (which is absorbed into the matrix $A$ coefficients).
 *
 * @tparam T Floating-point type used for the computation (typically float or double).
 */
template <typename T>
BandedMat<T>  DirectSolver<T>::setA(cudaStream_t& stream, Mat<T>& preAlocatedForA, Vec<int32_t>& preAlocatedForIndices, const GridDim& dim, const std::array<AdjacencyInd, numDiagonals>  adjInds) {

        preAlocatedForA.subMat(0, 1, preAlocatedForA._rows, preAlocatedForA._cols - 1).fill(1, stream);

        const KernelPrep kp = dim.kernelPrep();
        setAKernel<T><<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
            preAlocatedForA.toKernel2d(), dim,
            adjInds[0], adjInds[1], adjInds[2], adjInds[3], adjInds[4], adjInds[5], adjInds[6]
        );

        CHECK_CUDA_ERROR(cudaGetLastError());

        loadMapRowToDiag(preAlocatedForIndices, adjInds, stream);

        return BandedMat<T> (preAlocatedForA, preAlocatedForIndices);
    }

template <typename T>
void  DirectSolver<T>::loadMapRowToDiag(Vec<int32_t> diags, const std::array<AdjacencyInd, numDiagonals> adjInds, const cudaStream_t stream) {
        int32_t diagsCpu[numDiagonals];
        for (size_t i = 0; i < numDiagonals; i++) diagsCpu[adjInds[i].col] = adjInds[i].diag;
        diags.set(diagsCpu, stream);
    }

template <typename T>
DirectSolver<T>::DirectSolver(const CubeBoundary<T>& boundary, Vec<T>& f, Mat<T>& preAlocatedForBandedA, Vec<int32_t>& prealocatedForIndices, cudaStream_t stream):
        PoissonRHS<T>(boundary, f, stream),
        adjInds{
            AdjacencyInd(0, 0),
            AdjacencyInd(1, -1),
            AdjacencyInd(2, 1),
            AdjacencyInd(3, -this->dim.rows * this->dim.layers),
            AdjacencyInd(4, this->dim.rows * this->dim.layers),
            AdjacencyInd(5, -this->dim.rows),
            AdjacencyInd(6, this->dim.rows)
        },
        A(setA(stream, preAlocatedForBandedA, prealocatedForIndices, this->dim, adjInds))
    {}
template <typename T>
void  DirectSolver<T>::solve(Mat<T> prealocatedForBiCGSTAB) {
        cudaDeviceSynchronize();
        BiCGSTAB<T>::solve(A, this->_b, &prealocatedForBiCGSTAB);
    }

/**
 * Creates and solved an example Poisson class on a cube with the given side length.
 * @param dimLength The length of an edge of the grid.  //up to 325 works on Dov's computer.  After that the size of
 * the initally allocated memory exceeds the available memory on the gpu.
 */
void testPoisson(const size_t height, size_t width, size_t depth, Handle& hand) {

    auto boundary = CubeBoundary<double>::ZeroTo1(height, width, depth, hand);

    auto longVecs = Mat<double>::create(boundary.internalSize(), 1 + numDiagonals + 7);
    auto b = longVecs.col(0);
    b.fill(0, hand);

    // std::cout << "RunDirectSolver testPoisson b: " << b.size() << std::endl << GpuOut<double>(b, hand) << std::endl;

    auto A = longVecs.subMat(0, 1, boundary.internalSize(), numDiagonals);
    auto prealocatedForBiCGSTAB = longVecs.subMat(0, 1 + numDiagonals, boundary.internalSize(), 7);

    auto diagonalInds = Vec<int32_t>::create(numDiagonals);

    DirectSolver<double> solver(boundary, b, A, diagonalInds, hand);

    boundary.freeMem();

    solver.solve(prealocatedForBiCGSTAB);

    // std::cout << "x = \n" << GpuOut<double>(x.tensor(height, depth), hand) << std::endl;

}

/**
 * benchmarks  the BiCGSTAV algorithm.
 * @param dim The size of a dimension
 * @param hand
 */
void testPoisson(size_t dim, Handle& hand) {
    testPoisson(dim, dim, dim, hand);
}

template class DirectSolver<float>;
template class DirectSolver<double>;