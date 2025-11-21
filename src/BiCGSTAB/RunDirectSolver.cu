
#include <iostream>

#include "deviceArrays/headers/handle.h"
#include "deviceArrays/headers/Mat.h"
#include "deviceArrays/headers/Streamable.h"
#include "Poisson/CubeBoundary.h"
#include "Poisson/DirectSolver.cu"
constexpr  size_t numDiagonals = 7;
/**
 * Creates and solved an example Poisson class on a cube with the given side length.
 * @param dimLength The length of an edge of the grid.  //up to 325 works on Dov's computer.  After that the size of
 * the initally allocated memory exceeds the available memory on the gpu.
 */
void testPoisson(const size_t height, size_t width, size_t depth, Handle& hand) {

    auto boundary = CubeBoundary<double>::ZeroTo1(height, width, depth, hand);

    auto longVecs = Mat<double>::create(boundary.internalSize(), 2 + numDiagonals + 7);
    auto b = longVecs.col(0);
    b.fill(0, hand);
    auto x = longVecs.col(1);
    auto A = longVecs.subMat(0, 2, boundary.internalSize(), numDiagonals);
    auto prealocatedForBiCGSTAB = longVecs.subMat(0, 2 + numDiagonals, boundary.internalSize(), 7);

    auto diagonalInds = Vec<int32_t>::create(numDiagonals);

    DirectSolver<double> solver(boundary, b, A, diagonalInds, hand);

    boundary.freeMem();

    solver.solve(x, prealocatedForBiCGSTAB);

    std::cout << "x = \n" << GpuOut<double>(x, hand) << std::endl;

}

/**
 * @brief Main execution function for the Poisson FDM solver example.
 *
 * Sets up a simple 3D grid problem with constant boundary conditions,
 * initializes the RHS and solution vectors, and calls the solver.
 *
 * @param[in] argc Argument count (unused).
 * @param[in] argv Argument vector (unused).
 * @return 0 on successful execution.
 *///TODO:Similarly for any memory allocated by bicgstab.
int main(int argc, char *argv[]) {

    Handle hand;

    testPoisson(2, 2, 2, hand);

    // std::cout << "dimension size, number of iterations, total time" << std::endl;
    // for (size_t i = 2; i < 350; ++i) {
    // std::cout << i << ", ";
    // testPoisson(i, hand);
    // cudaDeviceSynchronize();

    // }



    return 0;
}