
#ifndef BICGSTAB_POISSONFDM_CUH
#define BICGSTAB_POISSONFDM_CUH
#include "deviceArrays/deviceArrays.h"

template <typename T>
class PoissonFDM {
private:
    Tensor<T> _grid;
    const Tensor<T> _f;
    const size_t _rows, _cols, _layers;

    [[nodiscard]] size_t layerSize() const {
        return _cols*_layers;
    }


    void solve2d(Vec<T>& x) {

    }
    void solve3d(Vec<T>& x, Handle& handle) {
        size_t n = (_rows - 2) * (_cols - 2) * (_layers - 2);
        size_t numNonZeroDiags = 7;
        Mat<T> A(numNonZeroDiags, n);
        Vec<T> b(n);
        Vec<T> indices(numNonZeroDiags);
        indices.set({0, 1, -1, _cols, -_cols, layerSize(), -layerSize()}, handle.stream);
    }
public:
    PoissonFDM(Tensor<T>& grid, size_t rows, size_t cols, size_t layers, const Tensor<T>& _f): _grid(grid), _rows(rows), _cols(cols), _layers(layers), _f(_f) {}

    void solve(Vec<T>& x) {
        if (_layers == 1) solve2d(x);
        else solve3d(x);

    }

};


#endif //BICGSTAB_POISSONFDM_CUH