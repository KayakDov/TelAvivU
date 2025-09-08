#include "deviceArrays.h"
#include <cmath>
#include "Event.h"

template <typename T>
class BiCGSTAB{
private:
    T tolerance;
    Handle handle[4]{};
    Event alphaReady, sReady, hReady, omegaReady, rReady, xReady, prodTS;
    const Vec b;
    Mat<T> paM;
    Vec<T> r(paM, 0, IndexType::Column), r_tilde(paM, 1, IndexType::Column), p(paM, 2, IndexType::Column), 
            v(paM, 3, IndexType::Column), s(paM, 4, IndexType::Column), t(paM, 5, IndexType::Column), h(paM, 6, IndexType::Column);
    Vec<T> paV(6); 
    Singleton<T> rho(paV, 0), alpha(paV, 1), omega(paV, 2), rho_new(paV, 3), beta(paV, 4), temp(paV, 5);
    Singleton<T> one(static_cast<T>(1)), zero(static_cast<T>(0)), minusOne(static_cast<T>(-1)); 
    size_t maxIterations;

    void wait(size_t streamIndex, std::initializer_list<std::reference_wrapper<Event>> e) {
        for (Event& e : evs) e.wait(handle[streamIndex]);
    }

    void renew(std::initializer_list<std::reference_wrapper<Event>> e) {
        for (Event& e : evs) e.renew();
    }

    void record(size_t streamIndex, std::initializer_list<std::reference_wrapper<Event>> e) {
        for (Event& e : evs) e.record(handle[streamIndex]);
    }

    void synch(size_t streamInd){
        cudaStreamSynchronize(handle[streamInd].stream);        
    }

    bool isSmall(const Vec& v, size_t streamInd){
        Singleton<T> vSquared;
            v.mult(v, &vSquared, handle + streamInd);
            T vSq = vSquared.get(handle[streamInd].stream);
            synch(streamInd);
            return vSq < tolerance;
    }

    void set(Vec& dst, const Vec& src, size_t streamInd){
        dst.set(src, handle[streamInd].stream);
    }

    void setQuotient(Singleton& dst, const Singleton& numerator, const Singleton& denom, size_t streamInd){
        dst.set(denom);
        denom.EBEPow(numerator, minusOne, handle[streamInd].stream);
    }

public:
    BiCGSTAB(
        const Vec& b, 
        T tolerance = std::is_same<T,double>::value ? T(1e-12) : T(1e-6),
        size_t maxItertions = size_t(-1),
        Mat<T>* preAlocated = nullptr        
    ):b(b),
      paM(preAlocated ? *preAlocated : Mat<T>(b.size(), 7)), 
      tolerance(tolerance){
            if(maxIterations == (size_t)-1) maxIterations = b.size()*5;
            cudaDeviceSynchronize();
            (cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE),...);
        }

    /**
     * Unpreconditioned BiCGSTAB algorithm to solve the linear system Ax = b.
     * @param A The coefficient matrix (Mat).
     * @param diags The diagonals of the matrix A.
     * @param b The right-hand side vector (CuArray1D).
     * @param preAlocated Optional pre-allocated memory for intermediate computations (Mat).  
     * There should be 7 columns and b.size() rows.
     * @param maxIterations Maximum number of iterations to perform. Default is
     * the size of vector b.
     * @param tolerance Convergence tolerance. Default is 1e-6 for float and 1e-12 for double.
     * @return The solution vector x (CuArray1D).
     * @throws std::runtime_error if T is not float or double.
     */    
    Vec<T> unpreconditionedBiCGSTAB(
        const Mat<T>& A, 
        const Vec<int>& diags, 
        Vec<T>* x = nullptr){

        static_assert(std::is_same<T,float>::value || std::is_same<T,double>::value,
                "Algorithms.cu unpreconditionedBiCGSTAB: T must be float or double");
       
        Vec<T> result = x ? *x : Vec<T>(b.size());
        record(0, {xReady});
        
        r_tilde.fillRandom(&handle[0]); // set r_tilde randomly    

        set(r, b, 0);
        A.diagMult(diags, result, &r, handle, &minusOne, &one); // r = b - A * x    
        
        set(p, r, 0);
        
        r_tilde.mult(r, &rho, handle);    
        
        wait(1, xReady);

        int i = 0;
        for(;i < maxIterations; i++) {
        
            A.diagMult(diags, p, &v, handle, &one, &zero); // v = A * p

            r_tilde.mult(v, &alpha, handle);        
            alpha.EBEPow(rho, minusOne, handle[0].stream); //alpha = rho / (r_tilde * v)
            record(0, {alphaReady});

            synch(1);
            omegaReady.renew();
            wait(1, {alphaReady});
            
            set(h, result, 1);
            synch(1);
            renew({alphaReady, xReady});
            h.add(p, alpha, handle + 1); // h = x + alpha * p

            temp.set(alpha, handle[0].stream);
            temp.mult(minusOne, handle);
            s.setSum(r, v, one, temp, handle[0].stream); // s = r - alpha * v
            record(0, {sReady});
            
            set(result, h, 1);

            wait(2, {sReady, hReady});
            if(isSmall(s, 2)) break;
            renew({sReady, hReady});

            A.diagMult(diags, s, &t, handle); // t = A * s

            t.mult(s, &temp, handle + 3);
            record(3, {prodTS});
            t.mult(t, &omega, handle);
            wait(0, {prodTS});
            omega.EBEPow(temp, minusOne, handle[0].stream); //omega = t * s / t * t;

            record(0, {omegaReady});

            wait(1, {omegaReady});
            result.add(s, omega, handle + 1); // x = h + omega * s
            record(1, {xReady});        
            
            synch(0);
            prodTS.renew();
            temp.set(omega, handle[0].stream);
            temp.mult(minusOne, handle);
            r.setSum(s, t, 1, temp, handle); // r = s - omega * t
            record(0, {rReady});

            wait(2, {xReady, rReady});

            if(isSmall(r, 2)) break;
            rReady.renew();

            r_tilde.mult(r, &rho_new, handle);

            setQuotient(temp, rho_new, rho, 0);
            setQuotient(beta, alpha, omega, 0);
            beta.EBEPow(temp, one, handle[0].stream); // beta = (rho_new / rho) * (alpha / omega);
            
            set(rho, rho_new, 0);

            p.mult(beta, handle);
            p.add(r, &one, handle); // p = r + beta * p
            temp.set(beta, handle[0].stream);
            temp.mult(omega, handle);
            p.sub(v, temp, handle); // p = p - beta * omega * v        
        }

        std::cout << "algorithms.cu unpreconditionedBiCGSTAB Number of iterations:" << i << std::endl;
        
        return result;
    }

};