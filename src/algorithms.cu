#include "deviceArrays.h"
#include <cmath>

/**
 * Unpreconditioned BiCGSTAB algorithm to solve the linear system Ax = b.
 * @param A The coefficient matrix (CuArray2D).
 * @param diags The diagonals of the matrix A.
 * @param b The right-hand side vector (CuArray1D).
 * @param preAlocated Optional pre-allocated memory for intermediate computations (CuArray2D).  
 * There should be 7 columns and b.size() rows.
 * @param maxIterations Maximum number of iterations to perform. Default is
 * the size of vector b.
 * @param tolerance Convergence tolerance. Default is 1e-6 for float and 1e-12 for double.
 * @return The solution vector x (CuArray1D).
 * @throws std::runtime_error if T is not float or double.
 */
template<typename T>
CuArray1D<T> unpreconditionedBiCGSTAB(
    const CuArray2D<T>& A, 
    const CuArray1D<int>& diags,
    const CuArray1D<T>& b, 
    CuArray1D<T>* x = nullptr, 
    size_t maxIterations = -1, 
    T tolerance = std::is_same<T,double>::value ? T(1e-12) : T(1e-6),
    CuArray2D<T>* preAlocated = nullptr
){
    if(maxIterations == (size_t)-1) maxIterations = b.size();
    static_assert(std::is_same<T,float>::value || std::is_same<T,double>::value,
              "Algorithms.cu unpreconditionedBiCGSTAB: T must be float or double");
    Handle handle;

    const CuArray2D<T>& pa = preAlocated ? *preAlocated : CuArray2D<T>(b.size(), 7);
    CuArray1D<T> r(pa, 0, IndexType::Column);
    CuArray1D<T> r_tilde(pa, 1, IndexType::Column);
    CuArray1D<T> p(pa, 2, IndexType::Column);
    CuArray1D<T> v(pa, 3, IndexType::Column);
    CuArray1D<T> s(pa, 4, IndexType::Column);
    CuArray1D<T> t(pa, 5, IndexType::Column);
    CuArray1D<T> h(pa, 6, IndexType::Column);


    CuArray1D<T> result = x ? *x : CuArray1D<T>(b.size());    
    
    r.set(b);
    A.diagMult(diags, result, &r, &handle, T(-1), T(1)); // r = b - A * x

    
    r_tilde.fillRandom(&handle); // r_tilde is a random vector

    T rho = r_tilde.mult(r);

    
    p.set(r);
    

    for(int i = 0; i < maxIterations; i++) {
        A.diagMult(diags, p, &v, &handle, T(1), T(0)); // v = A * p
        T alpha = rho / r_tilde.mult(v);
        h.set(result);
        h.add(p, alpha, &handle); // h = x + alpha * p
        s.set(r);
        s.sub(v, alpha, &handle); // s = r - alpha * v
        if(s.mult(s) < tolerance) {
            result.set(h);
            break;
        }
        A.diagMult(diags, s, &t, &handle, T(1), T(0)); // t = A * s
        T omega = t.mult(s) / t.mult(t);
        result.set(h);
        result.add(s, omega, &handle); // x = h + omega * s
        r.set(s);
        r.sub(t, omega, &handle); // r = s - omega * t
        if(r.mult(r) < tolerance) break;
        T rho_new = r_tilde.mult(r);
        T beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;
        
        p.mult(beta);
        p.add(r, T(1), &handle); // p = r + beta * p
        p.sub(v, beta * omega, &handle); // p = p - beta * omega * v        
    }
    
    return result;
}

