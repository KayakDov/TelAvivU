#include <utility>

#include "deviceArrays.h"


template<typename T>
Vec<T>::Vec(size_t cols, std::shared_ptr<T> ptr, size_t stride): GpuArray<T>(static_cast<size_t>(1), cols, stride, ptr) {
}

template <typename T>
Vec<T> Vec<T>::create(size_t length){
    T* rawPtr = nullptr;
    cudaMalloc(&rawPtr, length * sizeof(T));
    return Vec<T>(length, std::shared_ptr<T>(rawPtr, cudaFreeDeleter), 1);
}

template <typename T>
Vec<T> Vec<T>::subVec(size_t offset, size_t length, size_t stride){
    return Vec<T>(length, std::shared_ptr<T>(this->_ptr, (this->_ptr.data() + offset * this->getLD() * stride)));
}

template <typename T>
Vec<T> Vec<T>::mult(
    const Mat<T>& other,
    Vec<T>* result,
    Handle* handle,
    const Singleton<T>* alpha,
    const Singleton<T>* beta,
    bool transpose
) const {

    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    std::unique_ptr<Vec<T>> temp_res_ptr;
    Vec<T>* resPtr = this->_get_or_create_target(other._cols, result, temp_res_ptr);
    std::unique_ptr<Singleton<T>> temp_a_ptr;
    const Singleton<T>* a = this->_get_or_create_target(static_cast<T>(1), *h, alpha, temp_a_ptr);
    std::unique_ptr<Singleton<T>> temp_b_ptr;
    const Singleton<T>* b = this->_get_or_create_target(static_cast<T>(0), *h, beta, temp_b_ptr);
    
    other.mult(*this, resPtr, handle, a, b, !transpose);  
    
    return *resPtr;
}

template <typename T>
T Vec<T>::mult(
    const Vec<T>& other,
    Singleton<T>* result,
    Handle* handle
) const {
    if (this->_cols != other._cols)
        throw std::invalid_argument("Vector lengths do not match for dot product.");

    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    std::unique_ptr<Singleton<T>> temp_res_ptr;
    Singleton<T>* resPtr = this->_get_or_create_target(result, temp_res_ptr);
    
    if constexpr (std::is_same_v<T, float>)
        cublasSdot(h->handle, this->_cols, this->data(), this->getLD(), other.data(), other.getLD(), resPtr->data());
    else if constexpr (std::is_same_v<T, double>)
        cublasDdot(h->handle, this->_cols, this->data(), this->getLD(), other.data(), other.getLD(), resPtr->data());
    else static_assert(!std::is_same_v<T, float> && !std::is_same_v<T, double>, "Vec::add unsupported type.");

    T scalar = resPtr->get();

    return scalar;
}


template <typename T>
Vec<T> Vec<T>::operator*(const Mat<T>& other) const {
    return this->mult(other);
}

template <typename T>
T Vec<T>::operator*(const Vec<T>& other) const {
    return this->mult(other);
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
void Vec<T>::set(const GpuArray<T>& src, cudaStream_t stream) {
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
void Vec<T>::get(GpuArray<T>& dst, cudaStream_t stream) const {
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

template<typename T>
Singleton<T> Vec<T>::get(size_t i) {
    return Singleton<T>(std::shared_ptr<T>(this->_ptr, this->_ptr.data() + i * this->getLD()));
}

template <typename T>
void Vec<T>::add(const Vec<T>& x, const Singleton<T> *alpha, Handle* handle) {
    if (this->_cols != x._cols) 
        throw std::invalid_argument("Vector lengths do not match for add.");
    
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    std::unique_ptr<Singleton<T>> temp_a_ptr;
    const Singleton<T>* a = this->_get_or_create_target(static_cast<T>(1), *h, alpha, temp_a_ptr);
    
    if constexpr (std::is_same_v<T, float>)
        cublasSaxpy(h->handle, this->_cols, a->data(), x.data(), x.getLD(), this->data(), this->getLD());
    else if constexpr(std::is_same_v<T, double>) 
        cublasDaxpy(h->handle, this->_cols, a->data(), x.data(), x.getLD(), this->data(), this->getLD());
    else throw std::invalid_argument("Vec::add unsupported type.");
}

template <typename T>
void Vec<T>::sub(const Vec<T>& x, const Singleton<T>* alpha, Handle* handle) {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    Singleton<T> a;
    a.set(-alpha->get(), h->stream);
    this->add(x, &a, h);
}

template <typename T>
void Vec<T>::mult(const Singleton<T>& alpha, Handle* handle) {
    
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    
    if constexpr (std::is_same_v<T, float>)
        cublasSscal(h->handle, this->_cols, alpha.data(), this->data(), this->getLD());
    else if constexpr(std::is_same_v<T, double>) 
        cublasDscal(h->handle, this->_cols, alpha.data(), this->data(), this->getLD());
    else throw std::invalid_argument("Unsupported type.");
}

extern "C" __global__ void setup_kernel_float(curandState* state, unsigned long long seed, size_t size, size_t stride) {
    if (unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; id < size) curand_init(seed, id, 0, &state[id * stride]);
}

extern "C" __global__ void setup_kernel_double(curandState* state, unsigned long long seed, size_t size, size_t stride) {
    if (unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; id < size) curand_init(seed, id, 0, &state[id * stride]);

}

__global__ void fillRandomKernel_float(float* array, size_t size, size_t stride, curandState* state) {
    if (unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; id < size) array[id * stride] = curand_uniform(&state[id * stride]);
}

__global__ void fillRandomKernel_double(double* array, size_t size, size_t stride, curandState* state) {
    if (unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; id < size) array[id * stride] = curand_uniform_double(&state[id * stride]);
}

template <typename T>
void Vec<T>::fillRandom(Handle* handle) {
    
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    
    dim3 threadsPerBlock(256);
    dim3 numBlocks((this->_cols + threadsPerBlock.x - 1) / threadsPerBlock.x);

    curandState* devStates;
    CHECK_CUDA_ERROR(cudaMalloc(&devStates, this->_cols * sizeof(curandState)));

    if constexpr (std::is_same_v<T, float>){
        setup_kernel_float<<<numBlocks, threadsPerBlock, 0, h->stream>>>(devStates, 0, this->size(), this->getLD());
        h->synch();
        fillRandomKernel_float<<<numBlocks, threadsPerBlock, 0, h->stream>>>(this->data(), this->_cols, this->getLD(), devStates);
    } else if constexpr (std::is_same_v<T, double>) {
        setup_kernel_double<<<numBlocks, threadsPerBlock, 0, h->stream>>>(devStates, 0, this->size(), this->getLD());
        h->synch();
        fillRandomKernel_double<<<numBlocks, threadsPerBlock, 0, h->stream>>>(this->data(), this->_cols, this->getLD(), devStates);
    } else throw std::invalid_argument("Unsupported type.");
    
    CHECK_CUDA_ERROR(cudaFree(devStates));
}


template <typename T>
__global__ void EBEPowKernel(T* data, size_t size, size_t stride, const T* t, const T* n) {

    if (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size) {
        size_t offset = idx * stride;
        data[offset] = (*t) * pow(data[offset], *n);
    }
}

template <typename T>
void Vec<T>::EBEPow(const Singleton<T>& t, const Singleton<T>& n, cudaStream_t stream) {
    if (this->_cols == 0) return;

    int blockSize = 256;
    int gridSize  = (this->_cols + blockSize - 1) / blockSize;

    EBEPowKernel<<<gridSize, blockSize, 0, stream>>>(
        this->data(), this->_cols, this->getLD(),
        t.data(), n.data()
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename T>
void Vec<T>::setSum(const Vec& a, const Vec& b, const Singleton<T>* alpha, const Singleton<T>* beta, Handle* handle){

    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    std::unique_ptr<Singleton<T>> temp_a_ptr;
    const Singleton<T>* alph = this->_get_or_create_target(static_cast<T>(1), *h, alpha, temp_a_ptr);
    std::unique_ptr<Singleton<T>> temp_b_ptr;
    const Singleton<T>* bet = this->_get_or_create_target(0, *h, beta, temp_b_ptr);

    this->set(a, h->stream);
    this->mult(*alph, h);
    this->add(b, bet, h);
}

template<typename T>
Tensor<T>::Tensor(size_t rows, size_t cols, size_t layers, size_t ld, std::shared_ptr<T> _ptr):GpuArray<T>(rows, cols * layers, ld, _ptr) {

}

template<typename T>
Tensor<T> Tensor<T>::create(size_t rows, size_t cols, size_t layers) {
    Mat<T> temp = Mat<T>::create(rows, cols * layers);
    return Tensor<T>(rows, cols, layers, temp._ld, temp._ptr);
}

template<typename T>
Mat<T> Tensor<T>::layer(size_t index) {
    return Mat<T>(this->_rows, this->_cols, this->_ld, std::shared_ptr<T>(this->_ptr, this->_ptr.get() + index * this->_ld * this->_cols));
}

template<typename T>
Vec<T> Tensor<T>::depth(size_t row, size_t col) {
    return Vec<T>(
        this->_layers,
        std::shared_ptr<T>(this->_ptr, this->_ptr.get() + col * this->_ld + row),
        this->_ld * this->_cols
        );
}

template<typename T>
Singleton<T> Tensor<T>::get(size_t row, size_t col, size_t layer) {
    return Singleton<T>(std::shared_ptr<T>(this->_ptr, this->_ptr.get() + layer * this->_ld * this->_cols + col * this->_ld + row));
}


template class Vec<int>;
template class Vec<float>;
template class Vec<double>;