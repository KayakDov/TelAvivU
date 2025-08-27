#include "deviceArrays.h"

template <typename T>
Singleton<T>::Singleton():Vec<T>(1){}

template <typename T>
Singleton<T>::Singleton(const Vec<T>& superVector, int index): Vec<T>(superVector, index, 1){}

template <typename T>
Singleton<T>::Singleton(const Mat<T>& superMatrix, int row, int col):Vec<T>(1,1,1){
    size_t offset = col * superMatrix.getLD() + row;
    this->_ptr = std::shared_ptr<void>(
        superMatrix.getPtr(),
        static_cast<char*>(superMatrix.getPtr().get()) + offset * sizeof(T)
    );
}

template <typename T>
T Singleton<T>::get(cudaStream_t stream) const{
    T cpuPointer[1];
    this->Vec<T>::get(cpuPointer, stream);
    return cpuPointer[0];
}
template <typename T>
void Singleton<T>::set(const T val, cudaStream_t stream){
    T cpuPointer[1];
    cpuPointer[0] = val;
    this->Vec<T>::set(cpuPointer, stream);    
}

template class Singleton<int>;
template class Singleton<float>;
template class Singleton<double>;