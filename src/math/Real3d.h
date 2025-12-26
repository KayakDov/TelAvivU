#ifndef CUDABANDED_REAL3D_H
#define CUDABANDED_REAL3D_H

#include <cstddef>

class Real3d {

    public:
    double x, y, z;
    Real3d(double x, double y, double z);

    double& operator[](size_t i);

    static const Real3d ZERO;
};


#endif //CUDABANDED_REAL3D_H