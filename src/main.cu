#include <iostream>
#include "deviceArrays.h"
#include "testMethods.cu"
#include "algorithms.cu"

using namespace std;

int main(int argc, char const *argv[])
{
    // Check if a CUDA device is available.
    checkForDevice();
    
    CuArray2D<float> A(3, 3);
    CuArray1D<float> b(3);
    CuArray1D<float> x(3);

    float dataA[] = {1.0f, 2.0f, 3.0f,
                     4.0f, 5.0f, 6.0f,
                     0.0f, 8.0f, 0.0f};
    A.set(dataA);

    cout << "A:\n" << A << endl;

    float data_b[] = {1.0f, 2.0f, 3.0f};
    b.set(data_b);

    cout << "b:\n" << b << endl;

    const int diags[] = {-1, 0, 1};
    
    unpreconditionedBiCGSTAB(A, diags, b, &x, 20, 1e-6f);

    cout << "x:\n" << x << endl;


    // A.diagMult(diags, x);
    // cout << "\nproduct:\n" << A.diagMult(diags, b) << endl;
    
    return 0;
}