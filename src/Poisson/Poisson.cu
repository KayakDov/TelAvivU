
#include "Poisson.h"

#include <optional>

#include "../deviceArrays/headers/GridDim.cuh"


template <typename T>
__device__ T bVal(const GridInd3d& ind, const GridDim& g, const DeviceCubeBoundary<T>& dcb) {
    size_t distFromBack = g.layers - 1 - ind.layer;
    return (ind.row == 0             ? dcb.bottom(distFromBack, ind.col) : 0)
        + (ind.row == g.rows - 1     ? dcb.top(distFromBack, ind.col)    : 0)
        + (ind.col == 0              ? dcb.left(ind.row, distFromBack)   : 0)
        + (ind.col == g.cols - 1     ? dcb.right(ind.row, distFromBack)  : 0)
        + (ind.layer == 0            ? dcb.front(ind.row, ind.col)           : 0)
        + (ind.layer == g.layers - 1 ? dcb.back(ind.row, ind.col)            : 0);
}

template <typename T>
__global__ void setRHSKernel3D(DeviceData1d<T> b, const DeviceCubeBoundary<T> dcb, GridDim grid) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t bound;
    GridInd3d ind;
    if (idx < (bound = 2 * dcb.front.size())) {
        size_t layer;
        if (idx >= dcb.front.size()) {
            idx -= dcb.front.size();
            layer = grid.layers - 1;
        } else layer = 0;
        ind = GridInd3d(dcb.front.row(idx), dcb.front.col(idx), layer);
    }
    else if ( (idx -= bound) < (bound = 2 * dcb.left.size())) {
        size_t col;
        if (idx >= dcb.left.size()) {
            idx -= dcb.left.size();
            col = grid.cols - 1;
        } else col = 0;

        ind = GridInd3d(dcb.left.row(idx), col, dcb.left.col(idx));
    }
    else if ((idx -= bound) < (bound = 2* dcb.top.size())) {
        size_t row;
        if (idx >= dcb.top.size()) {
            idx -= dcb.top.size();
            row = grid.rows - 1;
        } else row = 0;
        ind = GridInd3d(row, dcb.top.col(idx), dcb.top.row(idx));
    }
    else return;
    // printf("idx = %d, index (row, col, layer) = (%lu, %lu, %lu)\n", blockIdx.x * blockDim.x + threadIdx.x, ind.row, ind.col, ind.layer);
    b[grid[ind]] -= bVal(ind, grid, dcb);
}

template <typename T>
size_t Poisson<T>::size() const {
    return dim.size();
}

template <typename T>
void Poisson<T>::setB(const CubeBoundary<T>& boundary, cudaStream_t stream) {

    KernelPrep kp(boundary.size());

    setRHSKernel3D<<<kp.gridDim, kp.blockDim, 0, stream>>>(_b.toKernel1d(), boundary.deviceCubeBoundary(), dim);

    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename T>
Poisson<T>::Poisson(const CubeBoundary<T>& boundary, Vec<T>& f, const cudaStream_t stream) :
    dim(boundary.frontBack._rows/2, boundary.frontBack._cols, boundary.topBottom._rows/2),
    _b(f) {

    setB(boundary, stream);

}


template class Poisson<float>;
template class Poisson<double>;