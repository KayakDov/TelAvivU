#include "deviceArrays.h"
#include <cmath>

//TODO: test random number generation, test this method here, write multipliation for sparse matrices

template<typename T>
CuArray1D<T> unpreconditionedBiCGSTAB(
    const CuArray2D<T>& A, 
    const int* diags,
    const CuArray1D<T>& b, 
    CuArray1D<T>* x = nullptr, 
    size_t maxIterations = -1, 
    T tolerance = std::is_same<T,double>::value ? T(1e-12) : T(1e-6)
){
    if(maxIterations == (size_t)-1) maxIterations = b.size();
    static_assert(std::is_same<T,float>::value || std::is_same<T,double>::value,
              "Algorithms.cu unpreconditionedBiCGSTAB: T must be float or double");
    Handle handle;

    CuArray1D<T> result = x ? *x : CuArray1D<T>(b.size());

    CuArray1D<T> r(b.size());
    
    r.set(b);
    A.diagMult(diags, *x, &r, &handle, T(-1), T(1)); // r = b - A * x

    CuArray1D<T> r_tilde(r.size());
    r_tilde.fillRandom(&handle); // r_tilde is a random vector

    // std::cout << "initial x: " << result << std::endl;
    // std::cout << "Initial r: " << r << std::endl;
    // std::cout << "Initial r_tilde: " << r_tilde << std::endl;
    // std::cout << "Initial b: " << b << std::endl;
    // std::cout << "Initial residual: " << r.mult(r) << std::endl;
    // std::cout << "Initial r_tilde: " << r_tilde.mult(r_tilde) << std::endl;

    T rho = r_tilde.mult(r);

    CuArray1D<T> p(r.size());
    p.set(r);

    CuArray1D<T> v(r.size());//TODO: allow for prealocated memory to pass as a pararamater.  Allow for all the variables to use the same block, maybe a 2d matrix.
    CuArray1D<T> s(r.size());
    CuArray1D<T> t(r.size());
    CuArray1D<T> h(r.size());

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

