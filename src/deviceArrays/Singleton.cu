#include "deviceArrays/deviceArrays.h"


template<typename T>
Singleton<T>::Singleton(std::shared_ptr<T> ptr):Vec<T>(1, ptr, 1) {}

template<typename T>
Singleton<T> Singleton<T>::create(cudaStream_t stream) {
    return Vec<T>::create(1, stream).get(0);
}

template<typename T>
Singleton<T> Singleton<T>::create(T val, cudaStream_t stream) {
    Singleton<T> temp = create(stream);
    temp.set(val, stream);
    return temp;
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

template <typename T>
const Singleton<T> Singleton<T>::ONE(static_cast<T>(1));

template <typename T>
const Singleton<T> Singleton<T>::ZERO(static_cast<T>(0));

template <typename T>
const Singleton<T> Singleton<T>::MINUS_ONE(static_cast<T>(-1));


template class Singleton<int>;
template class Singleton<float>;
template class Singleton<double>;