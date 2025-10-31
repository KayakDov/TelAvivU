
#include "GridDim.h"


size_t GridDim::size() const {
    return _rows * _cols * _layers;
}
GridDim::GridDim(const size_t rows, const size_t cols, const size_t layers) : _rows(rows), _cols(cols), _layers(layers) {
}

