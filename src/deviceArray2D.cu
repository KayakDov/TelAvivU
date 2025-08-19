#include "deviceArrays.h"


template <>
CuArray2D<float> CuArray2D<float>::mult(
    const CuArray2D<float>& other,
    CuArray2D<float>* result,
    Handle* handle,
    float alpha,
    float beta,
    bool transposeA,
    bool transposeB
) const {
    if (this->_cols != other._rows) throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    
    CuArray2D<float>* resPtr = result ? result: new CuArray2D<float>(this->_rows, other._cols);
    
    CuArray<float>::mult(other, resPtr, handle, alpha, beta, transposeA, transposeB);

    if (!result) {
        CuArray2D<float> temp = *resPtr;
        delete resPtr;
        return temp;
    }
    
    return *resPtr;
}

template <>
CuArray2D<double> CuArray2D<double>::mult(
    const CuArray2D<double>& other,
    CuArray2D<double>* result,
    Handle* handle,
    double alpha,
    double beta,
    bool transposeA,
    bool transposeB
) const {
    CuArray2D<double>* resPtr = result ? result: new CuArray2D<double>(this->_rows, other._cols);
    
    CuArray<double>::mult(other, resPtr, handle, alpha, beta, transposeA, transposeB);
    
    if (!result) {
        CuArray2D<double> temp = *resPtr;
        delete resPtr;
        return temp;
    }
    
    return *resPtr;
}

template <>
CuArray1D<float> CuArray2D<float>::mult(
    const CuArray1D<float>& other,
    CuArray1D<float>* result,
    Handle* handle,
    float alpha,
    float beta,
    bool transpose
    
) const {

    CuArray1D<float>* resPtr = result ? result: new CuArray1D<float>(other._cols);
    
    Handle* h = handle ? handle : new Handle();

    cublasSgemv(h->handle, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, this->_rows, this->_cols, &alpha, this->data(), this->getLD(), other.data(), other.getLD(), &beta, resPtr->data(), resPtr->getLD());

    if (!result) {
        CuArray1D<float> temp = *resPtr;
        delete resPtr;
        return temp;
    }

    if (!handle){
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
         delete h;
    }
    
    return *resPtr;
}

/**
 * Multiplies a 2D array with a 1D array, returning a new 1D array.
 * @param other The 1D array to multiply with.
 * @param result Optional pointer to an existing 1D array to store the result.
 * @param handle Optional CublasHandle for managing the cuBLAS context.
 * @param alpha Scalar multiplier for the result.   
 *  @param beta Scalar multiplier for the existing values in the result array.
 * @param transpose If true, the 2D array is transposed before multiplication.
 * @return A new 1D array containing the result of the multiplication.
 */
template <>
CuArray1D<double> CuArray2D<double>::mult(
    const CuArray1D<double>& other,
    CuArray1D<double>* result,
    Handle* handle,
    double alpha,
    double beta,
    bool transpose
) const {

    CuArray1D<double>* resPtr = result ? result: new CuArray1D<double>(other._cols);

    Handle* h = handle ? handle : new Handle();
    
    cublasDgemv(
        h->handle,
        transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
        this->_rows, this->_cols,
        &alpha,
        this->data(), this->getLD(),
        other.data(), other.getLD(),
        &beta,
        resPtr->data(), resPtr->getLD()
    );

    if (!result) {
        CuArray1D<double> temp = *resPtr;
        delete resPtr;
        return temp;
    }

    if (!handle){
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
         delete h;
    }
    
    return *resPtr;
}

template <>
CuArray2D<float> CuArray2D<float>::operator*(const CuArray2D<float>& other) const {
    return this->mult(other);
}
template <>
CuArray2D<double> CuArray2D<double>::operator*(const CuArray2D<double>& other) const {
    return this->mult(other);
}
template <>
CuArray1D<float> CuArray2D<float>::operator*(const CuArray1D<float>& other) const {
    return this->mult(other);
}
template <>
CuArray1D<double> CuArray2D<double>::operator*(const CuArray1D<double>& other) const {
    return this->mult(other);
}

template <typename T>
CuArray2D<T>::CuArray2D(size_t rows, size_t cols): CuArray<T>(rows, cols, 0) {
    void* rawPtr = nullptr;
    size_t pitch = 0;
    cudaError_t err = cudaMallocPitch(&rawPtr, &pitch, rows * sizeof(T), cols);
    if (err != cudaSuccess)
        throw std::runtime_error("cudaMallocPitch failed");

    this->_ptr = std::shared_ptr<void>(rawPtr, cudaFreeDeleter);
    this->_ld = pitch / sizeof(T);
}

template <typename T>
CuArray2D<T>::CuArray2D(const CuArray2D<T>& superArray, size_t startRow, size_t startCol, size_t height, size_t width)
    : CuArray<T>(height, width, superArray.getLD()) {
    size_t offset = startCol * superArray.getLD() + startRow;
    this->_ptr = std::shared_ptr<void>(
        superArray._ptr,
        static_cast<char*>(superArray._ptr.get()) + offset * sizeof(T)
    );
}

template <typename T>
size_t CuArray2D<T>::size() const {
    return this->_rows * this->_cols;
}

template <typename T>
size_t CuArray2D<T>::bytes() const {
    return this->_cols * this->_ld * sizeof(T);
}

template <typename T>
void CuArray2D<T>::set(const T* src, cudaStream_t stream) {
    cudaMemcpy2DAsync(
        this->_ptr.get(), this->_ld * sizeof(T),
        src, this->_rows * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyHostToDevice, stream
    );
}

template <typename T>
void CuArray2D<T>::get(T* dst, cudaStream_t stream) const {
    cudaMemcpy2DAsync(
        dst, this->_rows * sizeof(T),
        this->_ptr.get(), this->_ld * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyDeviceToHost, stream
    );
}

template <typename T>
void CuArray2D<T>::set(const CuArray<T>& src, cudaStream_t stream) {
    cudaMemcpy2DAsync(
        this->_ptr.get(), this->_ld * sizeof(T),
        src.data(), src.getLD() * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyDeviceToDevice, stream
    );
}

template <typename T>
void CuArray2D<T>::get(CuArray<T>& dst, cudaStream_t cuStream) const {
    cudaMemcpy2DAsync(
        dst.data(), dst.getLD() * sizeof(T),
        this->_ptr.get(), this->_ld * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyDeviceToDevice, cuStream
    );
}

template <typename T>
void CuArray2D<T>::set(std::istream& input_stream, cudaStream_t cuStream) {

    SetFromFile<T> helper(this->_rows, this->_cols, input_stream);

    while (helper.hasNext()) {
        helper.readNextChunk();
        CuArray2D<T> subArray(
            *this,
            0,
            helper.getColsProcessed(),
            this->_rows,
            helper.getNextChunkColNumber()
        );

        subArray.set(helper.getHostBuffer(), cuStream);

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());//TODO: this might be avoidable with multi threading

        helper.updateProgress();
    }
}

template <typename T>
void CuArray2D<T>::get(std::ostream& output_stream, cudaStream_t stream) const {

    GetToFile<T> helper(this->_rows, this->_cols, output_stream);

    while (helper.hasNext()) {
        CuArray2D<T> subArray(
            *this,
            0,
            helper.getColsProcessed(),
            this->_rows,
            helper.getNextChunkColNumber()
        );

        subArray.get(helper.getHostBuffer(), stream);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());//TODO: this might be avoidable with multi threading

        helper.writeNextChunkToFile();
        helper.updateProgress();
    }
}


template <>
CuArray2D<float> CuArray2D<float>::plus(
    const CuArray2D<float>& x, 
    CuArray2D<float>* result,
    float alpha,
    float beta,
    bool transposeA,
    bool transposeB,
    Handle* handle
) {
    if (this->_rows != x._rows || this->_cols != x._cols) {
        throw std::invalid_argument("Matrix dimensions do not match for add.");
    }
    
    // Determine the result pointer, creating a new one if necessary
    CuArray2D<float>* resPtr = result ? result : new CuArray2D<float>(this->_rows, this->_cols);
    
    Handle* h = handle ? handle : new Handle();

    cublasSgeam(
        h->handle, 
        transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
        transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
        this->_rows, this->_cols, 
        &alpha, x.data(), x.getLD(),
        &beta, this->data(), this->getLD(),
        resPtr->data(), resPtr->getLD()
    );

    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
    
    if (!result) {
        CuArray2D<float> temp = *resPtr;
        delete resPtr;
        return temp;
    }
    
    return *resPtr;
}

template <>
CuArray2D<double> CuArray2D<double>::plus(
    const CuArray2D<double>& x, 
    CuArray2D<double>* result,
    double alpha,
    double beta,
    bool transposeA,
    bool transposeB,
    Handle* handle
) {
    if (this->_rows != x._rows || this->_cols != x._cols) {
        throw std::invalid_argument("Matrix dimensions do not match for add.");
    }

    // Determine the result pointer, creating a new one if necessary
    CuArray2D<double>* resPtr = result ? result : new CuArray2D<double>(this->_rows, this->_cols);
    
    Handle* h = handle ? handle : new Handle();

    cublasDgeam(
        h->handle, 
        transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
        transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
        this->_rows, this->_cols, 
        &alpha, x.data(), x.getLD(),
        &beta, this->data(), this->getLD(),
        resPtr->data(), resPtr->getLD()
    );
    
    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
    
    if (!result) {
        CuArray2D<double> temp = *resPtr;
        delete resPtr;
        return temp;
    }
    
    return *resPtr;
}

template <>
CuArray2D<float> CuArray2D<float>::minus(
    const CuArray2D<float>& x,
    CuArray2D<float>* result,
    float alpha,
    float beta,
    bool transposeA,
    bool transposeB,
    Handle* handle
) {
    return this->plus(x, result, -alpha, beta, transposeA, transposeB, handle);
}

template <>
CuArray2D<double> CuArray2D<double>::minus(
    const CuArray2D<double>& x,
    CuArray2D<double>* result,
    double alpha,
    double beta,
    bool transposeA,
    bool transposeB,
    Handle* handle
) {
    return this->plus(x, result, -alpha, beta, transposeA, transposeB, handle);
}

template <typename T>
__global__ void scaleKernel(T* matrix, size_t rows, size_t cols, size_t ld, T alpha) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) matrix[row + col * ld] *= alpha;
    
}

template <typename T>
void CuArray2D<T>::_scale_impl(T alpha, Handle* handle) {
    if (this->_rows == 0 || this->_cols == 0) {
        return;
    }
    
    Handle* h = handle;
    if (!h) {
        h = new Handle();
    }

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(
        (this->_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (this->_rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    scaleKernel<<<numBlocks, threadsPerBlock, 0, h->stream>>>(
        this->data(),
        this->_rows,
        this->_cols,
        this->getLD(),
        alpha
    );

    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
}

template <>
void CuArray2D<float>::mult(float alpha, Handle* handle) {
    this->_scale_impl(alpha, handle);
}

template <>
void CuArray2D<double>::mult(double alpha, Handle* handle) {
    this->_scale_impl(alpha, handle);
}


template class CuArray2D<int>;
template class CuArray2D<float>;
template class CuArray2D<double>;