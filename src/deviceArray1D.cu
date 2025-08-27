#include "deviceArrays.h"



template <typename T>
Vec<T>::Vec(size_t length)
    : gpuArray<T>(1, length, 1) {
    void* rawPtr = nullptr;
    cudaMalloc(&rawPtr, length * sizeof(T));
    this->_ptr = std::shared_ptr<void>(rawPtr, cudaFreeDeleter);
}

template <typename T>
Vec<T>::Vec(const Vec<T>& superArray, size_t offset, size_t length, size_t stride)
    : gpuArray<T>(1, length, stride * superArray.getLD()) {
    this->_ptr = std::shared_ptr<void>(
        superArray._ptr,
        static_cast<char*>(superArray._ptr.get()) + offset * superArray.getLD() * sizeof(T)
    );
}

template <>
Vec<float> Vec<float>::mult(
    const Mat<float>& other,
    Vec<float>* result,
    Handle* handle,
    float alpha,
    float beta,
    bool transpose
) const {

    Vec<float>* resPtr = result ? result: new Vec<float>(other._cols);
    
    other.mult(*this, resPtr, handle, alpha, beta, !transpose);  

    if (!result) {
        Vec<float> temp = *resPtr;
        delete resPtr;
        return temp;
    }
    
    return *resPtr;
}

template <>
Vec<double> Vec<double>::mult(
    const Mat<double>& other,
    Vec<double>* result,
    Handle* handle,
    double alpha,
    double beta,
    bool transpose
) const {

    Vec<double>* resPtr = result ? result: new Vec<double>(other._cols);
    
    other.mult(*this, resPtr, handle, alpha, beta, !transpose);

    if (!result) {
        Vec<double> temp = *resPtr;
        delete resPtr;
        return temp;
    }
    
    return *resPtr;
}

template <>
float Vec<float>::mult(
    const Vec<float>& other,
    Singleton<float>* result,
    Handle* handle
) const {
    if (this->_cols != other._cols)
        throw std::invalid_argument("Vector lengths do not match for dot product.");
    
    
    Handle* h = handle ? handle : new Handle();
    Singleton<float>* r = result? result : new Singleton<float>();
    
    cublasSdot(h->handle, this->_cols, this->data(), this->getLD(), other.data(), other.getLD(), r->data());
    

    if (!handle){
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
         delete h;
    }

    float scalar = r->get();

    if(!result) delete r;

    return scalar;
}

template <>
double Vec<double>::mult(
    const Vec<double>& other,
    Singleton<double>* result,
    Handle* handle
) const {
    if (this->_cols != other._cols)
        throw std::invalid_argument("Vector lengths do not match for dot product.");
    
    Handle* h = handle ? handle : new Handle();
    Singleton<double>* r = result? result : new Singleton<double>();
    
    cublasDdot(h->handle, this->_cols, this->data(), this->getLD(), other.data(), other.getLD(), r->data());

    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }

    double scalar = r->get();

    if(!result) delete r;
    
    return scalar;
}

template <>
Vec<float> Vec<float>::operator*(const Mat<float>& other) const {
    return this->mult(other);
}
template <>
Vec<double> Vec<double>::operator*(const Mat<double>& other) const {
    return this->mult(other);
}
template <>
float Vec<float>::operator*(const Vec<float>& other) const {
    return this->mult(other);
}
template <>
double Vec<double>::operator*(const Vec<double>& other) const {
    return this->mult(other);
}


template <typename T>
Vec<T>::Vec(const Mat<T>& extractFrom, int index, IndexType indexType):
gpuArray<T>(
    1,
    indexType == IndexType::Row ? extractFrom._cols : extractFrom._rows,
    indexType == IndexType::Row ? extractFrom.getLD() : 1
) {
    if ((indexType == IndexType::Column && static_cast<size_t>(index) >= extractFrom._cols) || (indexType == IndexType::Row && static_cast<size_t>(index) >= extractFrom._rows))
        throw std::out_of_range("Out of range");
    size_t offset = indexType == IndexType::Row ? static_cast<size_t>(index) : static_cast<size_t>(index) * extractFrom.getLD();
    this->_ptr = std::shared_ptr<void>(
        extractFrom.getPtr(),
        const_cast<void*>(reinterpret_cast<const void*>(reinterpret_cast<const char*>(extractFrom.data()) + offset * sizeof(T)))
    );
}

template <typename T>
size_t Vec<T>::size() const {
    return this->_cols;
}

template <typename T>
size_t Vec<T>::bytes() const {
    return this->_cols * this->_ld * sizeof(T);
}

template <typename T>
void Vec<T>::set(const T* hostData, cudaStream_t stream) {
    if (this->_ld == 1) cudaMemcpyAsync(this->_ptr.get(), hostData, bytes(), cudaMemcpyHostToDevice, stream);
    else cudaMemcpy2DAsync(
            this->_ptr.get(), this->_ld * sizeof(T),
            hostData, sizeof(T),
            sizeof(T), this->_cols,
            cudaMemcpyHostToDevice, stream
        );
}

template <typename T>
void Vec<T>::get(T* hostData, cudaStream_t stream) const {
    if (this->_ld == 1)
        cudaMemcpyAsync(hostData, this->_ptr.get(), bytes(), cudaMemcpyDeviceToHost, stream);
    else cudaMemcpy2DAsync(
            hostData, sizeof(T),
            this->_ptr.get(), this->_ld * sizeof(T),
            sizeof(T), this->_cols,
            cudaMemcpyDeviceToHost, stream
        );
}

template <typename T>
void Vec<T>::set(const gpuArray<T>& src, cudaStream_t stream) {
    if (this->_ld == 1 && src.getLD() == 1) {
        cudaMemcpyAsync(this->_ptr.get(), src.data(), bytes(), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemcpy2DAsync(
            this->_ptr.get(), this->_ld * sizeof(T),
            src.data(), src.getLD() * sizeof(T),
            sizeof(T), this->_cols,
            cudaMemcpyDeviceToDevice, stream
        );
    }
}

template <typename T>
void Vec<T>::get(gpuArray<T>& dst, cudaStream_t stream) const {
    if (this->_ld == 1 && dst.getLD() == 1) {
        cudaMemcpyAsync(dst.data(), this->_ptr.get(), bytes(), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemcpy2DAsync(
            dst.data(), dst.getLD() * sizeof(T),
            this->_ptr.get(), this->_ld * sizeof(T),
            sizeof(T), this->_cols,
            cudaMemcpyDeviceToDevice, stream
        );
    }
}

template <typename T>
void Vec<T>::set(std::istream& input_stream, bool isText, bool, cudaStream_t stream) {
    StreamSet<T> helper(this->_rows, this->_cols, input_stream);
    while (helper.hasNext()) {
        helper.readChunk(isText);
        Vec<T> subArray(
            *this,
            helper.getColsProcessed(),
            helper.getChunkWidth()
        );
        subArray.set(helper.getBuffer().data(), stream);
        helper.updateProgress();
    }
}

template <typename T>
void Vec<T>::get(std::ostream& output_stream, bool isText, bool, cudaStream_t stream) const {
    StreamGet<T> helper(this->_rows, this->_cols, output_stream);
    while (helper.hasNext()) {
        Vec<T> subArray(
            *this,
            helper.getColsProcessed(),
            helper.getChunkWidth()
        );
        subArray.get(helper.getBuffer().data(), stream);
        helper.writeChunk(isText);
        helper.updateProgress();
    }
}

template <>
void Vec<float>::add(const Vec<float>& x, float alpha, Handle* handle) {
    if (this->_cols != x._cols) 
        throw std::invalid_argument("Vector lengths do not match for add.");
    
    Handle* h = handle ? handle : new Handle();
    
    cublasSaxpy(h->handle, this->_cols, &alpha, x.data(), x.getLD(), this->data(), this->getLD());
    
    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
}

template <>
void Vec<double>::add(const Vec<double>& x, double alpha, Handle* handle) {
    if (this->_cols != x._cols)
        throw std::invalid_argument("Vector lengths do not match for add.");
        
    Handle* h = handle ? handle : new Handle();
    
    cublasDaxpy(h->handle, this->_cols, &alpha, x.data(), x.getLD(), this->data(), this->getLD());
    
    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
}

template <>
void Vec<float>::sub(const Vec<float>& x, float alpha, Handle* handle) {
    this->add(x, -alpha, handle);
}

template <>
void Vec<double>::sub(const Vec<double>& x, double alpha, Handle* handle) {
    this->add(x, -alpha, handle);
}

template <>
void Vec<float>::mult(float alpha, Handle* handle) {
    if (this->_cols == 0) {
        return;
    }
    
    Handle* h = handle ? handle : new Handle();
    
    cublasSscal(h->handle, this->_cols, &alpha, this->data(), this->getLD());
    
    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
}

template <typename T>
Vec<T>::Vec(size_t cols, size_t rows, size_t ld)
    : gpuArray<T>(cols, rows, ld) {}


template <>
void Vec<double>::mult(double alpha, Handle* handle) {
    if (this->_cols == 0) {
        return;
    }
    
    Handle* h = handle ? handle : new Handle();
    
    cublasDscal(h->handle, this->_cols, &alpha, this->data(), this->getLD());
    
    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
}


extern "C" __global__ void setup_kernel_float(curandState* state, unsigned long long seed, size_t size, size_t stride) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        curand_init(seed, id, 0, &state[id * stride]);
    }
}

extern "C" __global__ void setup_kernel_double(curandState* state, unsigned long long seed, size_t size, size_t stride) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        curand_init(seed, id, 0, &state[id * stride]);
    }
}

__global__ void fillRandomKernel_float(float* array, size_t size, size_t stride, curandState* state) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        array[id * stride] = curand_uniform(&state[id * stride]);
    }
}

__global__ void fillRandomKernel_double(double* array, size_t size, size_t stride, curandState* state) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        array[id * stride] = curand_uniform_double(&state[id * stride]);
    }
}

template <>
void Vec<float>::fillRandom(Handle* handle) {
    
    Handle* h = handle ? handle : new Handle();
    
    dim3 threadsPerBlock(256);
    dim3 numBlocks((this->_cols + threadsPerBlock.x - 1) / threadsPerBlock.x);

    curandState* devStates;
    CHECK_CUDA_ERROR(cudaMalloc(&devStates, this->_cols * sizeof(curandState)));

    setup_kernel_float<<<numBlocks, threadsPerBlock, 0, h->stream>>>(devStates, 0, this->size(), this->getLD());

    fillRandomKernel_float<<<numBlocks, threadsPerBlock, 0, h->stream>>>(this->data(), this->_cols, this->getLD(), devStates);
    
    CHECK_CUDA_ERROR(cudaFree(devStates));
    
    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
}

template <>
void Vec<double>::fillRandom(Handle* handle) {
    
    Handle* h = handle ? handle : new Handle();

    dim3 threadsPerBlock(256);
    dim3 numBlocks((this->_cols + threadsPerBlock.x - 1) / threadsPerBlock.x);

    curandState* devStates;
    CHECK_CUDA_ERROR(cudaMalloc(&devStates, this->_cols * sizeof(curandState)));

    setup_kernel_double<<<numBlocks, threadsPerBlock, 0, h->stream>>>(devStates, 0, this->size(), this->getLD());

    fillRandomKernel_double<<<numBlocks, threadsPerBlock, 0, h->stream>>>(this->data(), this->size(), this->getLD(), devStates);
    
    // Free the device states
    CHECK_CUDA_ERROR(cudaFree(devStates));

    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
}

template <typename T>
__global__ void EBEPowKernel(T* data, size_t size, size_t stride, const T* t, const T n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        size_t offset = idx * stride;
        data[offset] = (*t) * pow(data[offset], n);
    }
}

template <typename T>
void Vec<T>::EBEPow(const Singleton<T>& t, const T n, cudaStream_t stream) {
    if (this->_cols == 0) return;

    int blockSize = 256;
    int gridSize  = (this->_cols + blockSize - 1) / blockSize;

    EBEPowKernel<<<gridSize, blockSize, 0, stream>>>(
        this->data(), this->_cols, this->getLD(),
        t.data(), n
    );

    CHECK_CUDA_ERROR(cudaGetLastError());

    if (stream == nullptr) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
}




template class Vec<int>;
template class Vec<float>;
template class Vec<double>;