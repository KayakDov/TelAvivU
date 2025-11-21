#include "EigenDecompSolver.h"

template <typename T>
__device__ constexpr T PI = static_cast<T>(3.14159265358979323846);

// ============================================================================
//                                    Kernels
// ============================================================================

template <typename T>
__global__ void eigenMatLKernel(DeviceData2d<T> eVecs) {
    if (const GridInd2d ind; ind < eVecs) {
        eVecs[ind] = std::sqrt(2/static_cast<T>(eVecs.rows + 1)) *
            std::sin((ind.col + 1)*(ind.row + 1)*PI<T> / static_cast<T>(eVecs.rows + 1));
    }
}

template <typename T>
__global__ void eigenValLKernel(DeviceData1d<T> eVals) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < eVals.cols) {
        T s = std::sin((idx+1)*PI<T> / (2*(eVals.cols + 1)));
        eVals[idx] = -4 * s * s;
    }
}

template <typename T>
__global__ void setUTildeKernel(DeviceData3d<T> uTilde,
                                const DeviceData2d<T> eVals,
                                const DeviceData3d<T> fTilde) {
    if (GridInd3d ind; ind < uTilde) {
        uTilde[ind] =
            fTilde[ind] /
            ( eVals(ind.col,0)
            + eVals(ind.row,1)
            + eVals(ind.layer,2) );
    }
}

// ============================================================================
//                         EigenDecompSolver<T> Methods
// ============================================================================

template <typename T>
void EigenDecompSolver<T>::eigenL(size_t i, cudaStream_t stream) {
    KernelPrep kpVec = eVecs[i].kernelPrep();
    eigenMatLKernel<T><<<kpVec.gridDim, kpVec.blockDim, 0, stream>>>(
        eVecs[i].toKernel2d());

    size_t n = eVecs[i]._cols;
    KernelPrep kpVal(n);
    eigenValLKernel<T><<<kpVal.gridDim, kpVal.blockDim, 0, stream>>>(
        eVals.col(i).subVec(0, n, 1).toKernel1d());
}

template <typename T>
void EigenDecompSolver<T>::setUTilde(const Tensor<T>& f,
                                     Tensor<T>& u,
                                     Handle& hand) {
    KernelPrep kp = f.kernelPrep();
    setUTildeKernel<T><<<kp.gridDim, kp.blockDim, 0, hand>>>(
        u.toKernel3d(),
        eVals.toKernel2d(),
        f.toKernel3d());
}


template <typename T>
void EigenDecompSolver<T>::multE(size_t i,
                                 bool transposeEigen,
                                 bool transpose,
                                 const Mat<T>& a1,
                                 Mat<T>& dst1,
                                 size_t stride,
                                 Handle& hand,
                                 size_t batchCount) {

    Mat<T>::batchMult(
        transpose ? a1      : eVecs[i], transpose ? stride : 0,
        transpose ? eVecs[i]: a1,       transpose ? 0      : stride,
        dst1, stride,
        transpose ? false : transposeEigen,
        transpose ? transposeEigen : false,
        hand, batchCount
    );
}

template <typename T>
void EigenDecompSolver<T>::multEX(const Mat<T>& src, Mat<T>& dst, Handle& hand, bool transposeE) {
    multE(0, transposeE, true, src, dst, src._rows, hand, this->dim.layers);
}

template <typename T>
void EigenDecompSolver<T>::multEY(const Mat<T>& src, Mat<T>& dst, Handle& hand, bool transposeE) {
    multE(1, transposeE, false, src, dst, src._rows, hand, this->dim.layers);
}

template <typename T>
void EigenDecompSolver<T>::multEZ(const Mat<T>& src, Mat<T>& dst, Handle& hand, bool transposeE) {
    multE(2, transposeE, true, src, dst, this->dim.layers * this->dim.rows, hand, this->dim.cols);
}

template <typename T>
void EigenDecompSolver<T>::multiplyEF(Handle& hand, Tensor<T>& src, Tensor<T>& dst, bool transposeE) {

    auto xF  = src.layerRowCol(0), dx1 = dst.layerRowCol(0),
        yF  = dst.layerRowCol(0), dyF = src.layerRowCol(0),
        zS  = src.layerColDepth(0), dzS  = dst.layerColDepth(0);

    if (transposeE) {
        multEX(xF, dx1, hand, transposeE);
        multEY(yF, dyF, hand, transposeE);
        multEZ(zS, dzS, hand, transposeE);
    } else {
        multEZ(zS, dzS, hand, transposeE);
        multEY(yF, dyF, hand, transposeE);
        multEX(xF, dx1, hand, transposeE);
    }
}

template <typename T>
EigenDecompSolver<T>::EigenDecompSolver(const CubeBoundary<T>& boundary,
                                        Vec<T>& x, Vec<T>& f,
                                        Handle& hand)
    : Poisson<T>(boundary, f, hand),
      eVecs({SquareMat<T>::create(this->dim.cols),
             SquareMat<T>::create(this->dim.rows),
             SquareMat<T>::create(this->dim.layers)}),
      eVals(Mat<T>::create(
          std::max(this->dim.rows, std::max(this->dim.cols, this->dim.layers)),
          3))
{
    for (size_t i = 0; i < 3; ++i) eigenL(i, hand);

    auto fT  = f.tensor(this->dim.rows, this->dim.layers);
    auto fTd = x.tensor(this->dim.rows, this->dim.layers);

    multiplyEF(hand, fT, fTd, true);

    setUTilde(fTd, fT, hand);

    auto xT = x.tensor(this->dim.rows, this->dim.layers);
    multiplyEF(hand, fT, xT, false);
}


template class EigenDecompSolver<double>;
template class EigenDecompSolver<float>;
