#include "EigenDecompSolver.cu"


/**
 * @brief Wrap a raw GPU pointer in a non-owning shared_ptr.
 *
 * The returned shared_ptr does **not take ownership** of the
 * underlying CUDA device memory. No deallocation occurs when the
 * shared_ptr goes out of scope.
 *
 * @tparam T Element type
 * @param p Raw CUDA device pointer
 * @return std::shared_ptr<T> with no-op deleter
 */
template<typename T>
std::shared_ptr<T> nonOwningGpuPtr(T* p) {
    return std::shared_ptr<T>(p, [](T*){});
}


/**
 * @brief Solve the 3D eigen-decomposition linear system on GPU data.
 *
 * This method wraps raw device pointers into Mat<T>, Vec<T>, and CubeBoundary<T>
 * objects and executes an EigenDecompSolver. No memory is allocated or deallocated.
 *
 * ## Data Format (IMPORTANT)
 * (unchanged — omitted for brevity)
 */
template<typename T>
void eigenDecompSolver(const T* frontBack,  const size_t fbLd,
                       const T* leftRight,  const size_t lrLd,
                       const T* topBottom,  const size_t tbLd,
                       T* f,                const size_t fStride,
                       T* x,                const size_t xStride,
                       const size_t height,
                       const size_t width,
                       const size_t depth)
{
    // Construct faces: rows, cols, leading dimension, data
    Mat<T> fb(2 * height, width, fbLd, nonOwningGpuPtr(const_cast<T*>(frontBack)));
    Mat<T> lr(2 * height, depth, lrLd, nonOwningGpuPtr(const_cast<T*>(leftRight)));
    Mat<T> tb(2 * depth,  width, tbLd, nonOwningGpuPtr(const_cast<T*>(topBottom)));

    CubeBoundary<T> boundary(fb, lr, tb);

    const size_t n = height * width * depth;

    Vec<T> xVec(n, nonOwningGpuPtr(x), xStride);
    Vec<T> fVec(n, nonOwningGpuPtr(const_cast<T*>(f)), fStride);

    Handle hand;

    cudaDeviceSynchronize();
    EigenDecompSolver eds(boundary, xVec, fVec, hand);
    cudaDeviceSynchronize();
}


// ============================================================================
//                    EXPORTED SYMBOLS FOR FORTRAN
// ============================================================================

extern "C" {

    void eigenDecompSolver_float(
        const float* frontBack, const size_t* fbLd,
        const float* leftRight, const size_t* lrLd,
        const float* topBottom, const size_t* tbLd,
        float* f,               const size_t* fStride,
        float* x,               const size_t* xStride,
        const size_t* height,   const size_t* width,
        const size_t* depth)
    {
        eigenDecompSolver<float>(
            frontBack,  *fbLd,
            leftRight,  *lrLd,
            topBottom,  *tbLd,
            f,          *fStride,
            x,          *xStride,
            *height, *width, *depth
        );
    }


    void eigenDecompSolver_double(
        const double* frontBack, const size_t* fbLd,
        const double* leftRight, const size_t* lrLd,
        const double* topBottom, const size_t* tbLd,
        double* f,               const size_t* fStride,
        double* x,               const size_t* xStride,
        const size_t* height,    const size_t* width,
        const size_t* depth)
    {
        eigenDecompSolver<double>(
            frontBack,  *fbLd,
            leftRight,  *lrLd,
            topBottom,  *tbLd,
            f,          *fStride,
            x,          *xStride,
            *height, *width, *depth
        );
    }

} // extern "C"


/**
 * @brief Main entry point to demonstrate the FastDiagonalizationMethod
 *        for a 2x2x2 grid.
 */
int main() {

    Handle hand;

    const size_t height = 2, width = 2, depth = 4;

    const auto boundary = CubeBoundary<double>::ZeroTo1(height, width, depth, hand);

    auto memAloc = Mat<double>::create(boundary.internalSize(), 2);

    auto x = memAloc.col(0);
    auto f = memAloc.col(1);

    f.fill(0, hand);

    EigenDecompSolver<double> fdm(boundary, x, f, hand);

    std::cout << "x = \n" << GpuOut<double>(x.tensor(height, depth), hand) << std::endl;

    return 0;
}