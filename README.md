# **Fortran Interface: Guide to the Poisson Eigen Decomposition Solver**

This file contains instructions for Fortran programmers on how to build the necessary C++/CUDA library call the Fortran bindings.

## **Building the C++/CUDA Solver Library (CMake)**

The Fortran bindings require that the core C++/CUDA solver library is built first using CMake.

From your project root directory, follow these steps:

1. **Navigate to the Build Directory**  
   
2. **Build the EigenDecomp Target**  
   The target containing the Fortran interface logic is EigenDecomp. Compile it using the CMake build system:  
   cmake \--build . \--target EigenDecomp

   This process generates the necessary object files and links them into the required executable/library components (like libBiCGSTAB\_LIB.a and the final EigenDecomp binary). The output will be located within this cmake-build-debug directory.  Ignore the warnings; this is a work in progress.

## **Using the Fortran Module**

The interface is contained within the eigendecomp\_interface module.

To access the solver from your Fortran code, simply add the use statement:

program my\_fortran\_app  
use eigendecomp\_interface  
implicit none  
\! ...  
end program my\_fortran\_app

# **3D Poisson Solver: Eigen Decomposition Summary**

This solver uses an eigen decomposition approach to quickly solve the discrete 3D Poisson equation: L u \= f.

## **Steps**

1. **Initialization and Input**
   * **Inputs:** Boundary coefficients (frontBack, leftRight, topBottom) defining the discretized Laplacian matrix (L), and the Right-Hand Side vector (f).
2. **Spectral Decomposition**
   * The solver implicitly uses the pre-computed eigenvalues (lambda) and eigenmatrices (Phi) corresponding to the 1D discrete Laplacian for each grid dimension.
3. **Forward Transformation (3D FFT-like)**
   * The Right-Hand Side vector (f) is transformed into the eigen-space (frequency domain) by successive application of the transposed eigenmatrices across each dimension (Depth, Width, and Height).
   * **Operation:** The transformed vector (f\_tilde) is calculated as:  
     f\_tilde \= Phi\_Depth \* Phi\_Width \* Phi\_Height \* f

4. **Decoupled Solution**
   * The system is solved algebraically in the eigen-space where the matrix is diagonal, yielding the transformed solution (u\_tilde).
   * **Operation:** The calculation is an element-wise division:  
     u\_tilde\[i,j,k\] \= f\_tilde\[i,j,k\] / lambda\[i,j,k\]

5. **Inverse Transformation**
   * The final solution u is recovered by applying the inverse transformation to the decoupled solution (u\_tilde).
   * **Operation:** This is the reverse application of the eigenmatrices:  
     u \= Inverse(Phi\_Depth \* Phi\_Width \* Phi\_Height) \* u\_tilde

6. **Output**
   * The final solution vector x (which is u), mapped to the H x W x D grid.

### **Solver Subroutines and Data Types**

Two subroutines are available, one for single precision and one for double precision:

* **Single Precision:** eigenDecompSolver\_float (uses real(c\_float))
* **Double Precision:** eigenDecompSolver\_double (uses real(c\_double))

### **Argument Details**

The subroutines accept ten arguments. Note that all dimension and stride arguments are passed **by value** (value) using the C size type (integer(c\_size\_t)).

Note, all data is column major. Two-dimensional matrices have a distance of ld between the first elements of each column, with ld - height padding at the end of each column.  Three-dimensional data (row, column, depth) is stored stored with row changing fastest, then depth, then column.  So when flattened, the first height elements are the first column in the first layer, the next height elements are the first column in the second layer, and after all the first columns in all the layers we have the second columns in each layer in turn, and so on.
The boundary matrices frontBack, leftRight, and topBottom are stored as three-dimensional matrices, each with two layers.  The first layer is the front/left/top and the second layer back/right/bottom.
The first row of the front and back boundary matrices is up against the top.  The first column of the front and back matrices is up against the left boundary.
The first row of the left and right matrices is up against the top.  The first column of the left and right matrices is up against the back.
The first row the top and bottom matrices is up against the back boundary.  The first column up against the left boundary.

| Argument Name | Intent | Data Type (Fortran Kind) | Description                                                                                                                                                                                                                                                                                                                                              |
| :---- | :---- | :---- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| frontBack, leftRight, topBottom | in | real(c\_float) / real(c\_double) | The boundaries of the grid on which the Poisson operator is defined.  frontBack has height x width = 2H * W, leftRight has height x width = 2H * D, and topBottom has height x width 2D * W.  The height of each matrix is doubled because the front boundary is stored above the back boundary, the left above the right, and the top above the bottom. |
| fbLd, lrLd, tbLd | value | integer(c\_size\_t) | The stride or **Leading Dimension (LD)** for the corresponding input boundary matrices (fbLD is the leading dimension for frontBakc, lrLD the leading dimension for leftRight).  This is the distance between columns of the matrix, and should be greater than or equal to the height of the matrix, allowing for padding.                              |
| f | out | real(c\_float) / real(c\_double) | This should be the right hand side of the Poisson operator.  It should have H*W*D elements stored in column major order as described above where the next column is in the next layer, not adjacent within the same layer. Values stored here will be overwritten with scratch work.                                                                     |
| fStride | value | integer(c\_size\_t) | The stride for f. The amount of space for one element of f to the next.  Typically 1.  This is stored as a vector, not a matrix.                                                                                                                                                                                                                         |
| x | out | real(c\_float) / real(c\_double) | Output array.  Should have H*W*D elements.  This is stored as a vector, not a matrix.  Use the column major order described above.                                                                                                                                                                                                                       |
| xStride | value | integer(c\_size\_t) | The stride for the output array x. Typically 1.                                                                                                                                                                                                                                                                                                          |
| height, width, depth | value | integer(c\_size\_t) | The dimensions of the 3D computational grid: **Height (**$H$**)**, **Width (**$W$**)**, and **Depth (**$D$**)**.                                                                                                                                                                                                                                         |

## **Linking Your Fortran Program**

The final step is linking your compiled Fortran code against the C++/CUDA library.

1. **Compile Fortran:**  
   gfortran \-c src/eigendecomp\_interface.f90  
   gfortran \-c your\_main\_program.f90

2. Link Everything Together:  
   You must link the Fortran objects with the C++ library output and required runtimes (stdc++ and cudart).  
   gfortran your\_main\_program.o eigendecomp\_interface.o \\  
   \-L\~/Documents/TelAvivU/cmake-build-debug \-lBiCGSTAB\_LIB \\  
   \-o final\_solver\_app \\  
   \-Wl,-rpath,\~/Documents/TelAvivU/cmake-build-debug \\  
   \-lstdc++ \-lcudart

    * **\-L**: Path to the required library.
    * **\-lBiCGSTAB\_LIB**: Links the dependency library (e.g., libBiCGSTAB\_LIB.a).
    * **\-lstdc++ \-lcudart**: Links C++ and CUDA runtime libraries.

This should result in a runnable executable.