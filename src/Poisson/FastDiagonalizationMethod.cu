
#include "deviceArrays/headers/vec.h"
#include "GridDim.h"
#include "deviceArrays/headers/squareMat.h"
#include "deviceArrays/headers/singleton.h"
#include <cmath>

template <typename T>
__device__ __host__ constexpr T PI = static_cast<T>(3.14159265358979323846);

template <typename T>
__global__ void eiganMatLKernel(
    T* __restrict__ eVecs,
    const size_t height, const size_t ld
) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < height)
        eVecs[col * ld + row] = std::sqrt(2/static_cast<T>(height + 1)) * std::sin(col * row * PI<T> / static_cast<T>(height + 1));
}
template <typename T>
__global__ void eiganValLKernel(
    T* __restrict__ eVals,
    const size_t n
) {

    if (const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n) {
        T sinExpr = std::sin(idx * PI<T> / (2*n + 1));
        eVals[idx] = -4 * sinExpr * sinExpr;
    }
}

template <typename T>
class FastDiagonalizationMethod:  GridDim{
private:

    /**
     * Sets the eigen matrix for L[i].
     * @param i The index of the desired matrix, 1,2, or 3.
     * @param stream
     */
    void eigenL(size_t i, cudaStream_t stream) { //TODO: this method could run both kernels on different streams.  Also different streams for each index.
        constexpr dim3 blockDim(16, 16);

        const dim3 gridDim3d(
            (eVecs[i]._cols + blockDim.x - 1) / blockDim.x,
            (eVecs[i]._rows + blockDim.y - 1) / blockDim.y
        );
        eiganMatLKernel<T><<<gridDim3d, blockDim, 0, stream>>>(eVecs[i].data(), eVecs[i]._rows, eVecs[i]._ld);

        constexpr size_t gridDim1d = 256;

        eiganValLKernel<T><<<(eVecs[i]._cols + gridDim1d - 1) / gridDim1d, gridDim1d, 0, stream>>>(eVals[i].data(), eVals[i].size());
    }

    std::array<SquareMat<T>, 3> I;//TODO: I may not need I's.
    std::array<SquareMat<T>, 3> L;
    std::array<SquareMat<T>, 3> eVecs;//Note: transpose is the inverse for these matrices.
    std::array<Vec<T>, 3> eVals;

    void setL(size_t i, cudaStream_t stream) {//TODO: try parallelizing this data
        L[i].fill(static_cast<T>(0), stream);
        L[i].diag(0).fill(static_cast<T>(-2), stream);
        L[i].diag(1).fill(static_cast<T>(1), stream);
        L[i].diag(-1).fill(static_cast<T>(1), stream);
    }

    void setI(size_t i, cudaStream_t stream) {
        I[i].setToIdentity(stream);
    }

    /**
     * The kronecker product of a, b, and c
     * @param a
     * @param b
     * @param c
     * @param temp should be a.rows * b.rows x a.cols * b.cols
     * @param result should be a.rows * b.rows * c.rows x a.cols * b.cols * c.cols
     * @param stream
     */
    void kronecker3(const Mat<T>& a, const Mat<T>& b, const Mat<T>& c, Mat<T>& temp, Mat<T>& result, cudaStream_t stream) {
        temp.fill(0, stream);
        a.multKronecker(b, temp, stream);
        temp.multKronecker(c, result, stream);
    }

    std::array<SquareMat<T>, 3> generateSquareMat() const{
        return {SquareMat<T>::create(_cols), SquareMat<T>::create(_rows),  SquareMat<T>::create(_layers)};
    }
public:

    FastDiagonalizationMethod(size_t rows, size_t cols, size_t layers) :
        GridDim(rows, cols, layers),
        I(generateSquareMat()),
        L(generateSquareMat()),
        eVecs(generateSquareMat()),
        eVals({Vec<T>::create(_cols), Vec<T>::create(_rows), Vec<T>::create(_layers)})
    {
        Handle hand;
        Mat<T> temp = SquareMat<T>::create(std::max(_rows, std::max(cols, layers)));

        for (size_t i = 0; i < 3; ++i) {
            setL(i, hand.stream);

            eigenL(i, hand.stream);
            auto maybeI = SquareMat<T>::create(L[i]._cols);
            eVecs[i].mult(eVecs[i], &maybeI, &hand, &(Singleton<T>::ONE), &(Singleton<T>::ZERO), false, true);
            eVecs[i].get(std::cout, true, false, &hand.stream);

        }


    }
};


/**
 * @brief Main entry point to demonstrate the FastDiagonalizationMethod for a 2x2x2 grid.
 */
int main() {

    Handle handle;

    auto a = Mat<double>::create(2, 2);
    auto b = Mat<double>::create(2, 2);
    auto c = Mat<double>::create(2, 2);
    a.fill(1, handle.stream);
    b.fill(2, handle.stream);

    a.get(std::cout << "\nA = \n", true, false, &handle);
    b.get(std::cout << "\nB = \n", true, false, &handle);

    a.mult(b, &c, &handle, &Singleton<double>::ONE, &Singleton<double>::ZERO, true,true);

    c.get(std::cout << "\nC = \n", true, false, &handle);

    // constexpr size_t dim = 2;
    // constexpr  size_t rows = dim;
    // constexpr  size_t cols = dim;
    // constexpr  size_t layers = dim;
    //
    // std::cout << "Starting Fast Diagonalization Method setup for a "
    //           << rows << "x" << cols << "x" << layers << " grid (Total size: "
    //           << rows * cols * layers << ")\n\n";
    //
    // FastDiagonalizationMethod<double> fdm(rows, cols, layers);

    return 0;
}