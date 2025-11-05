/**
 * @file deviceArrays.cu
 * @brief Templated classes for managing 1D and 2D arrays on a CUDA device in column-major order.
 */
#include "../headers/singleton.h"
#include <vector>
#include <numeric>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <fstream>


template<typename T>
GpuArray<T>::GpuArray(size_t rows, size_t cols, size_t ld, std::shared_ptr<T> _ptr):_rows(rows), _cols(cols), _ld(ld), _ptr(_ptr) {
}

template <typename T>
GpuArray<T>::~GpuArray() = default;

template<typename T>
size_t GpuArray<T>::bytes() const {
    return sizeof(T) * size();
}

template <typename T>
__global__ void fill2dKernel(DeviceData2d<T> a, const T val){

    if (const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < a.cols * a.rows)
        a[idx] = val;
}

template<typename T>
void GpuArray<T>::fill(T val, cudaStream_t stream) {

    constexpr size_t BLOCK_SIZE = 256;
    size_t num_blocks = (this->size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fill2dKernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(this->toKernel(), val);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename T>
DeviceData2d<T> GpuArray<T>::toKernel() { return DeviceData2d<T>(_rows, _cols, _ld, _ptr.get()); }

template<typename T>
std::shared_ptr<T> GpuArray<T>::ptr() const{
    return _ptr;
}

template<typename T>
DeviceData2d<T> GpuArray<T>::toKernel() const {
    return DeviceData2d<T>(_rows, _cols, _ld, _ptr.get());
}

template<typename T>
void GpuArray<T>::freeMem() {
    _ptr.reset();
}

template <typename T>
void GpuArray<T>::mult(
    const GpuArray<T>& other,
    GpuArray<T>* result,
    Handle* handle,
    const Singleton<T> *alpha,
    const Singleton<T> *beta,
    bool transposeA,
    bool transposeB
) const {
    
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    std::unique_ptr<Singleton<T>> temp_a_ptr;
    const Singleton<T>* a = _get_or_create_target(static_cast<T>(1), *h, alpha, temp_a_ptr);
    std::unique_ptr<Singleton<T>> temp_b_ptr;
    const Singleton<T>* b = _get_or_create_target(static_cast<T>(0), *h, beta, temp_b_ptr);

    cublasOperation_t transA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
            transB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

    if constexpr (std::is_same_v<T, float>)
        CHECK_CUDA_ERROR(cublasSgemm(h->handle,
        transA, transB,
        this->_rows, other._cols, this->_cols,
        a->toKernel(),
        this->toKernel(), this->_ld,
        other.toKernel(), other._ld,
        b->toKernel(),
        result->toKernel(), result->_ld));
    else if constexpr (std::is_same_v<T, double>)
        CHECK_CUDA_ERROR(cublasDgemm(h->handle,
        transA, transB,
        this->_rows, other._cols, this->_cols,
        a->toKernel(),
        this->toKernel(), this->_ld,
        other.toKernel(), other._ld,
        b->toKernel(),
        result->toKernel(), result->_ld));
    else throw std::invalid_argument("Unsupported type.");
}

// --- Helper Functions and Macros Definitions ---


template <typename T>
Mat<T>* GpuArray<T>::_get_or_create_target(size_t rows, size_t cols, Mat<T>* result, std::unique_ptr<Mat<T>>& out_ptr_unique) const {
    if (result) return result;
    else {
        out_ptr_unique = std::make_unique<Mat<T>>(Mat<T>::create(rows, cols));

        return out_ptr_unique.get();
    }
}

template <typename T>
Singleton<T>* GpuArray<T>::_get_or_create_target(Singleton<T>* result, std::unique_ptr<Singleton<T>>& out_ptr_unique, cudaStream_t stream) const {
    if (result) return result;
    else {
        out_ptr_unique = std::make_unique<Singleton<T>>(Singleton<T>::create(stream));
        return out_ptr_unique.get();
    }
}

template <typename T>
const Singleton<T>* GpuArray<T>::_get_or_create_target(T defaultVal, Handle& hand, const Singleton<T>* result, std::unique_ptr<Singleton<T>>& out_ptr_unique) const {
    if (result) return result;
    else {
        out_ptr_unique = std::make_unique<Singleton<T>>(Singleton<T>::create(defaultVal, hand.stream));
        return out_ptr_unique.get();
    }
}

template <typename T>
Vec<T>* GpuArray<T>::_get_or_create_target(size_t length, Vec<T>* result, std::unique_ptr<Vec<T>>& out_ptr_unique, cudaStream_t stream) const {
    if (result) return result;
    else {
        out_ptr_unique = std::make_unique<Vec<T>>(Vec<T>::create(length, stream));
        return out_ptr_unique.get();
    }
}

// --- kronecker_kernel.cu (or part of deviceArrays.cu) ---

#include <cuda_runtime.h>
// Include the deviceArrays.h header here to get GpuArray class definition

/**
 * @brief CUDA kernel for computing the Kronecker product C = A (x) B using a 2D block grid.
 *
 * This kernel assigns one thread to each element of the resulting matrix C by
 * mapping the thread's 2D global index (row, col) directly to the element's
 * position in C.
 *
 * The dimensions are:
 * - A: (heightA x widthA)
 * - B: (heightB x widthB)
 * - C: (heightA * heightB) x (widthA * widthB)
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @param A Pointer to matrix A data on the device.
 * @param B Pointer to matrix B data on the device.
 * @param result Pointer to result matrix C data on the device.
 */
template <typename T>
__global__ void kroneckerKernel(const DeviceData2d<T> a, const DeviceData2d<T> b, DeviceData2d<T> result) {

    if (const GridInd2d indR; indR < result)
        result(indR) += a(indR.row / b.rows, indR.col / b.cols) * b(indR.row % b.rows, indR.col % b.cols);

}

template<typename T>
void GpuArray<T>::multKronecker(const GpuArray<T>& other, GpuArray<T>& result, cudaStream_t stream) const {

    constexpr int threadsPerBlock = 16;

    const int gridX = (result._cols + threadsPerBlock - 1) / threadsPerBlock;
    const int gridY = (result._rows + threadsPerBlock - 1) / threadsPerBlock;

    dim3 blockSize(threadsPerBlock, threadsPerBlock);
    dim3 gridSize(gridX, gridY);

    kroneckerKernel<<<gridSize, blockSize, 0, stream>>>(
        this->toKernel(),
        other.toKernel(),
        result.toKernel()
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
}

template class GpuArray<float>;
template class GpuArray<double>;
template class GpuArray<int32_t>;
template class GpuArray<size_t>;
template class GpuArray<unsigned char>;