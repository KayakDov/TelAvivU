
 #ifndef BICGSTAB_POISSONFDM_CUH
#define BICGSTAB_POISSONFDM_CUH
#include "deviceArrays/deviceArrays.h"


template <typename T>
class Segmenter {
private:
    const bool seperateA, seperateB;
public:
    Segmenter(bool seperateA, bool seperateB) : seperateA(seperateA), seperateB(seperateB) {}
    T operator()(T val0, T val1, T val2) {
        return seperateA ? val0 : (seperateB ? val1 : val2);
    }
};


/**
 *
 * @tparam T float or double
 * @param b The right hand side of the equation.  Should start with the column major solution to the Poisson function.  Will be modified.
 * @param frontBack Two matrices, the firs tis the front boundary values of the grid and the second is the back boundary values of the grid.  Indexed 0 and 1.  The front face is indexed as a matrix with the top up and the bottom down.  The back matrix is aligned.
 * @param fbLd The leading dimension of the front and back matrices.
 * @param fbSize The number of elements in the front matrix, which is equal to the number of elements in the back matrix.
 * @param leftRight indexed 2 and 3.  The face values are indexed similarly to the front and back.
 * @param lrLd
 * @param lrSize
 * @param topBottom Indexed 4 and 5.  The top face values are indexed with the back up and the front down.
 * @param tbLd
 * @param tbSize
 * @param height The number of elements in the grid, not including the boundary values.
 * @param width The width of the grid, not including boundary values.
 * @param depth The depth of the grid, not including boundary values.
 */
template <typename T>
__global__ void setRHSKernel3D(T* __restrict__ b,
                               const T* __restrict__ frontBack, const size_t fbLd, size_t fbSize,
                               const T* __restrict__ leftRight, const size_t lrLd, size_t lrSize,
                               const T* __restrict__ topBottom, const size_t tbLd, size_t tbSize,
                               const size_t height, const size_t width, const size_t depth) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (fbSize + lrSize + tbSize) * 2) return;

    Segmenter<T> seg3(idx < 2*fbSize, idx < 2*fbSize + 2*lrSize);
    const size_t layer = seg3(
        idx < fbSize ? 0 : depth - 1,
        depth - 1 - ((idx - 2*fbSize) / height) % depth,
        depth - 1 - ((idx - 2 * fbSize + 2 * lrSize) % depth)
        );
    const size_t row = seg3(
        idx  % height,
        (idx - 2 * fbSize) % height,
        idx - 2 * fbSize - 2 * lrSize < tbSize ? 0 : height - 1
        );
    const size_t col = seg3(
        (idx / height) % width,
         idx - 2 * fbSize < lrSize ? 0 : width - 1,
         ((idx - 2 * fbSize - 2 * lrSize) / depth) % width
        );


    b[row + (col + layer * width) * height] +=
          (row == 0           ? topBottom[col*tbLd + layer]                    : 0)
        + (row == height - 1  ? topBottom[col*tbLd + layer + tbSize]           : 0)
        + (col == 0           ? leftRight[(depth - layer)*lrLd + row]          : 0)
        + (col == width - 1   ? leftRight[(depth - layer)*lrLd + row + lrSize] : 0)
        + (layer == 0         ? frontBack[col*fbLd + row]                      : 0)
        + (layer == depth - 1 ? frontBack[col*fbLd + row + fbSize]             : 0);
}

template <typename T>
class PoissonFDM {
private:
    const Mat<T> _frontBack, _leftRight, _topBottom;
    const Vec<T> _b;
    const size_t _rows, _cols, _layers;

    template <typename T>
    void setB3d(Vec<T>& b, cudaStream_t stream) {
        const size_t totalThreadsNeeded = (_frontBack.size() + _leftRight.size() + _topBottom.size());

        const int threadsPerBlock = 256;
        const int gridDim = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

        setRHSKernel3D<<<gridDim, threadsPerBlock, 0, stream>>>(
            b.data(),
            _frontBack.data(), _frontBack.getLD(), _frontBack.size()/2,
            _leftRight.data(), _leftRight.getLD(), _leftRight.size()/2,
            _topBottom.data(), _topBottom.getLD(), _topBottom.size()/2,
            _rows, _cols, _layers);

        CHECK_CUDA_ERROR(cudaGetLastError());
    }


    void solve2d(Vec<T>& x) {

    }
    void solve3d(Vec<T>& x, Handle& handle) {
        size_t numNonZeroDiags = 7;
        Mat<T> A(numNonZeroDiags, _b.size());
        Vec<T> indices(numNonZeroDiags);
        indices.set({0, 1, -1, _cols, -_cols, layerSize(), -layerSize()}, handle.stream);

        A.row(0).fill(-6, handle.stream);
        for (size_t i = 1; i < numNonZeroDiags; ++i) A.row(i).fill(1, handle.stream);

        setB3d(b, handle.stream);

    }







public:
    PoissonFDM(Tensor<T>& grid, size_t rows, size_t cols, size_t layers, const Tensor<T>& _f): _grid(grid), _rows(rows), _cols(cols), _layers(layers), _f(_f) {}

    void solve(Vec<T>& x) {
        if (_layers == 1) solve2d(x);
        else solve3d(x);

    }

};


#endif //BICGSTAB_POISSONFDM_CUH

// __global__ void setRHSKernel3D(T* __restrict__ b,
//                                const T* __restrict__ f, const size_t fld,
//                                const T* __restrict__ grid, const size_t gld, // holds boundary values
//                                const size_t height, const size_t width, const size_t depth) {
//
//     size_t row = blockIdx.x * blockDim.x + threadIdx.x + 1;
//     size_t col = blockIdx.y * blockDim.y + threadIdx.y + 1;
//     size_t layer = blockIdx.z * blockDim.z + threadIdx.z + 1;
//
//     if (row >= height - 1 || col >= width - 1 || layer >= depth - 1) return;
//
//
//     size_t gIdx = (layer * width + col) * gld + row;
//     size_t bIdx = row - 1 + ((col - 1) + (layer - 1) * (width - 2)) * (height - 2);
//
//     b[bIdx] = f[(layer * width + col) * fld + row]
//                 + (row - 1 == 0 ? grid[gIdx - 1]: 0)
//                 + (row + 1 == height - 1 ? grid[gIdx + 1]: 0)
//                 + (col - 1 == 0 ? grid[gIdx - gld]: 0)
//                 + (col + 1 == width - 1 ? grid[gIdx + gld]: 0)
//                 + (layer - 1 == 0 ? grid[gIdx - gld * width]: 0)
//                 + (layer + 1 == depth - 1 ? grid[gIdx + gld * width]: 0);
// }