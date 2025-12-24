
#include <chrono>
#include "poisson/CubeBoundary.h"
#include "solvers/EigenDecompSolver.h"
#include "poisson/DirectSolver.cuh"
// #include "FortranBindings.hpp"
#include "wrapFortranBindings.h"
using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

template<typename T>
void benchMarkEigenDecompSolver(size_t height, size_t width, size_t depth, std::array<Handle, 3>& hand3) {
    const auto boundary = CubeBoundary<double>::ZeroTo1(height, width, depth, hand3[0]);

    auto memAlocFX = Mat<double>::create(boundary.internalSize(), 2);


    size_t maxDim = std::max(width, std::max(height, depth));

    Mat<T> eigenStorage = Mat<T>::create(maxDim, width + height + depth + 3);
    SquareMat<T> eX = eigenStorage.sqSubMat(0, 0, width),
            eY = eigenStorage.sqSubMat(0, width, height),
            eZ = eigenStorage.sqSubMat(0, width + height, depth);

    Mat<T> vals = eigenStorage.subMat(0, width + height + depth, maxDim, 3);

    auto x = memAlocFX.col(0);
    auto f = memAlocFX.col(1);

    f.fill(0, hand3[0]);

    cudaDeviceSynchronize();

    TimePoint start = std::chrono::steady_clock::now();
    PoissonRHS<double> poisson(boundary, f, hand3[0]);
    EigenDecompSolver<double> fdm(x, f, eX, eY, eZ, vals, hand3);
    fdm.solve(x, f, hand3[0]);
    cudaDeviceSynchronize();
    TimePoint end = std::chrono::steady_clock::now();


    double iterationTime = (static_cast<std::chrono::duration<double, std::milli>>(end - start)).count();

    std::cout << iterationTime << ", ";

    // std::cout << "x = \n" << GpuOut<double>(x.tensor(height, depth), hand3[0]) << std::endl;
}

template<typename T>
void benchMarkEigenDecompSolver(size_t dim, std::array<Handle, 3>& hand3) {
    benchMarkEigenDecompSolver<T>(dim, dim, dim, hand3);
}

/**
 * @brief Main entry point to demonstrate the FastDiagonalizationMethod
 *        for a 2x2x2 grid.
 */
int main() {
    std::array<Handle, 3> hand3;

    constexpr size_t maxDimensions = 5;

    std::cout << "dim, time" << std::endl;

    for (size_t dim = 1; dim < maxDimensions; dim++) {
        std::cout << dim << ", ";
        benchMarkEigenDecompSolver<double>(dim, hand3);
        cudaDeviceSynchronize();

        testPoisson(dim, hand3[0]);
        std::cout << std::endl;
    }


    //let's choose a gridi

    return 0;
}


