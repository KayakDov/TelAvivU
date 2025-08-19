#include <iostream>
#include "deviceArrays.h"
#include "testMethods.cu"

using namespace std;

int main(int argc, char const *argv[])
{
    // Check if a CUDA device is available.
    checkForDevice();
    
    CuArray1D<float> a(3);
    CuArray1D<float> b(3);

    cout << "Input a." << endl;
    cin >> a;
    cout << "Input b." << endl;
    cin >> b;
    cout << a << " * " << b << " = " << a * b << endl;
    
    return 0;
}