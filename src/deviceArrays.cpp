/**
 * @file deviceArrays.cu
 * @brief Templated classes for managing 1D and 2D arrays on a CUDA device in column-major order.
 */



#include <vector>
#include <numeric>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

/**
 * @enum IndexType
 * @brief Indicates whether to index by row or by column.
 */
enum class IndexType {
    Row,    /**< Row index */
    Column  /**< Column index */
};

/**
 * @brief CUDA device memory deleter function for std::shared_ptr.
 * @param ptr Pointer to CUDA device memory to free.
 */
inline void cudaFreeDeleter(void* ptr) {
    if (ptr) cudaFree(ptr);
}

/**
 * @brief Abstract base template for CUDA array types.
 */
template <typename T>
class CudaArray {

public:
    /** Number of rows (const). */
    const size_t _rows;

    /** Number of columns (const). */
    const size_t _cols;


protected:
    /** Pointer to the device memory (shared pointer). */
    std::shared_ptr<void> _ptr;
    /** Leading dimension (stride) in elements (const). */
    size_t _ld;

    /**
     * @brief Protected constructor for CudaArray.
     * @param rows Number of rows in the array.
     * @param cols Number of columns in the array.
     * @param ld Leading dimension (stride) in elements.
     */
    CudaArray(size_t rows, size_t cols, size_t ld)
        : _rows(rows), _cols(cols), _ld(ld){}

public:
    /**
     * @brief Default destructor for CudaArray.
     * Cleans up the device memory automatically.
     */
    virtual ~CudaArray() = default;

    /**
     * @brief Get the number of elements in the array.
     * @return Total number of elements in the array.
     * @note This is a pure virtual function, must be implemented by derived classes.
     * @return size_t Total number of elements in the array.
     */
    virtual size_t size() const = 0;
    /**
     * @brief Get the size of the array in bytes.
     * @return Total size in bytes of the array.
     * @note This is a pure virtual function, must be implemented by derived classes.
     */
    virtual size_t bytes() const = 0;  

    /**
     * @brief Set the array data from host memory.
     * @param hostData Pointer to the host data to copy from.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     * @note This is a pure virtual function, must be implemented by derived classes.
     */
    virtual void set(const T* hostData, cudaStream_t stream = 0) = 0;

    /**
     * @brief Get the array data to host memory.
     * @param hostData Pointer to the host memory to copy to.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     * @note This is a pure virtual function, must be implemented by derived classes.
     */
    virtual void get(T* hostData, cudaStream_t stream = 0) const = 0;
    
    /**
     * @brief Set the array data from another CudaArray.
     * @param src Source CudaArray to copy from.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     * @note This is a pure virtual function, must be implemented by derived classes.
     */
    virtual void set(const CudaArray<T>& src, cudaStream_t stream = 0) = 0;

    /**
     * @brief Get the array data to another CudaArray.
     * @param dst Destination CudaArray to copy to.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     * @note This is a pure virtual function, must be implemented by derived classes.
     */
    virtual void get(CudaArray<T>& dst, cudaStream_t stream = 0) const = 0;
    
    /**
     * @brief Get the raw pointer to the device memory.
     * @return Pointer to the device memory.
     */
    T* data() { return static_cast<T*>(_ptr.get()); }

    /**
     * @brief Get the raw pointer to the device memory (const version).
     * @return Pointer to the device memory.
     */
    const T* data() const { return static_cast<const T*>(_ptr.get()); }

    /**
     * @brief Get the leading dimension (stride) in elements.
     * @return Leading dimension in elements.
     */
    size_t getLD() const { return _ld; }

    /**
     * @brief Get the shared pointer to the device memory.
     * @return Shared pointer to the device memory.
     */
    std::shared_ptr<void> getPtr() const { return _ptr; }
        
};


/**
 * @brief CUDA 2D array view, column-major storage.
 *
 * Storage layout: columns are contiguous with stride _ld.
 */
template <typename T>
class CudaArray2D : public CudaArray<T> {
public:
    /**
     * @brief Constructor for CudaArray2D.
     * Allocates device memory for a 2D array with given rows and columns.
     * @param rows Number of rows in the array.
     * @param cols Number of columns in the array.
     * @throws std::runtime_error if cudaMallocPitch fails.
     * @note The pitch (leading dimension) is automatically calculated based on the column size.
     */
    CudaArray2D(size_t rows, size_t cols): CudaArray<T>(rows, cols, 0) {
        void* rawPtr = nullptr;
        size_t pitch = 0;
        cudaError_t err = cudaMallocPitch(&rawPtr, &pitch, cols * sizeof(T), rows);
        if (err != cudaSuccess) 
            throw std::runtime_error("cudaMallocPitch failed");
        
        this->_ptr = std::shared_ptr<void>(rawPtr, cudaFreeDeleter);
        this->_ld = pitch / sizeof(T);  // leading dimension in elements
    }

    /**
     * @brief Constructor for CudaArray2D that creates a subarray view.
     * @param superArray The parent CudaArray2D to create a subarray from.
     * @param startRow Starting row index in the parent array.
     * @param startCol Starting column index in the parent array.
     * @param height Height of the subarray.
     * @param width Width of the subarray.
     */
    CudaArray2D(const CudaArray2D<T>& superArray, size_t startRow, size_t startCol, size_t height, size_t width)
        : CudaArray<T>(height, width, superArray.getLD()) {
        size_t offset = startCol * superArray.getLD() + startRow; // column-major: col offset first, then row offset
        this->_ptr = std::shared_ptr<void>(
            superArray._ptr,
            static_cast<void*>(static_cast<char*>(superArray._ptr.get()) + offset * sizeof(T))
        );
    }

    /**
     * @brief The numver of elements in the array.
     * @return Total number of elements in the array (rows * cols).
     */
    size_t size() const override {
        return this->_rows * this->_cols;
    }

    /**
     * @brief Get the size of the array in bytes.
     * @return Total size in bytes of the array (rows * ld * sizeof(T)).
     */
    size_t bytes() const override {
        return this->_cols * this->_ld * sizeof(T);
    }

    /**
     * @brief Set the array data from host memory.
     * @param src Pointer to the host data to copy from.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void set(const T* src, cudaStream_t stream = 0) override {
        cudaMemcpy2DAsync(
            this->_ptr.get(), this->_ld * sizeof(T),
            src, this->_rows * sizeof(T),
            this->_rows * sizeof(T), this->_cols,
            cudaMemcpyHostToDevice, stream
        );
    }

    /**
     * @brief Get the array data to host memory.
     * @param dst Pointer to the host memory to copy to.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void get(T* dst, cudaStream_t stream = 0) const override {
        cudaMemcpy2DAsync(
            dst, this->_rows * sizeof(T),
            this->_ptr.get(), this->_ld * sizeof(T),
            this->_rows * sizeof(T), this->_cols,
            cudaMemcpyDeviceToHost, stream
        );
    }

    /**
     * @brief Set the array data from another CudaArray.
     * @param src Source CudaArray to copy from.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void set(const CudaArray<T>& src, cudaStream_t stream = 0) override {
        cudaMemcpy2DAsync(
            this->_ptr.get(), this->_ld * sizeof(T),
            src.data(), src.getLD() * sizeof(T),
            this->_rows * sizeof(T), this->_cols,
            cudaMemcpyDeviceToDevice, stream
        );
    }

    /**
     * @brief Get the array data to another CudaArray.
     * @param dst Destination CudaArray to copy to.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void get(CudaArray<T>& dst, cudaStream_t stream = 0) const override {
        cudaMemcpy2DAsync(
            dst.data(), dst.getLD() * sizeof(T),
            this->_ptr.get(), this->_ld * sizeof(T),
            this->_rows * sizeof(T), this->_cols,
            cudaMemcpyDeviceToDevice, stream
        );
    }

};

/**
 * @brief CUDA 1D array view, representing either a vector or a single column/row slice.
 *
 * Note: For column-major data,
 *   - _rows = 1
 *   - _cols = length of vector
 *   - _ld = stride between elements (in elements)
 */
template <typename T>
class CudaArray1D : public CudaArray<T> {
public:
    /**
     * @brief Constructor for CudaArray1D.
     * Allocates device memory for a 1D array with given length.
     * @param length Length of the 1D array.
     * @throws std::runtime_error if cudaMalloc fails.
     */
    explicit CudaArray1D(size_t length)
        : CudaArray<T>(1, length, 1) {
        void* rawPtr = nullptr;
        cudaMalloc(&rawPtr, length * sizeof(T));
        this->_ptr = std::shared_ptr<void>(rawPtr, cudaFreeDeleter);
    }

    /**
     * @brief Constructor for CudaArray1D that creates a subarray view.
     * @param superArray The parent CudaArray1D to create a subarray from.
     * @param offset Starting index in the parent array.
     * @param length Length of the subarray.
     * @param ld Leading dimension (stride) in elements for the subarray.
     * @throws std::out_of_range if offset + length exceeds the parent array size.
     * @note The leading dimension is used to calculate the offset correctly.
     */ 
    CudaArray1D(const CudaArray1D<T>& superArray, size_t offset, size_t length, size_t stride = 1)
        : CudaArray<T>(1, length, stride * superArray.getLD()) {
        this->_ptr = std::shared_ptr<void>(
            superArray._ptr,
            static_cast<void*>(static_cast<char*>(superArray._ptr.get()) + offset * this->_ld * sizeof(T))
        );
    }

    /**
     * @brief Constructor for CudaArray1D that extracts a row or column from a CudaArray2D.
     * @param extractFrom The parent CudaArray2D to extract from.
     * @param index The row or column index to extract.
     * @param indexType Specify whether to extract a row or a column (IndexType::Row or IndexType::Column).
     * @throws std::out_of_range if index is out of bounds.
     */
    CudaArray1D(const CudaArray2D<T>& extractFrom, int index, IndexType indexType):  
    CudaArray<T>(
        1, 
        indexType == IndexType::Row ? extractFrom._cols : extractFrom._rows, 
        indexType == IndexType::Row ? extractFrom.getLD() : 1
    ) {
        
        if ((indexType == IndexType::Column && static_cast<size_t>(index) >= extractFrom._cols) || (indexType == IndexType::Row && static_cast<size_t>(index) >= extractFrom._rows))
            throw std::out_of_range("Out of range");
        
        size_t offset = indexType == IndexType::Row ? static_cast<size_t>(index) : static_cast<size_t>(index) * extractFrom.getLD();
       
        this->_ptr = std::shared_ptr<void>(
            extractFrom.getPtr(),
            const_cast<void*>(reinterpret_cast<const void*>(reinterpret_cast<const char*>(extractFrom.data()) + offset * sizeof(T)))
        );
    }

    /**
     * @brief Get the number of elements in the array.
     * @return Total number of elements in the array (cols).
     */
    size_t size() const override {
        return this->_cols;
    }

    /**
     * @brief Get the size of the array in bytes.
     * @return Total size in bytes of the array (cols * ld * sizeof(T)).
     */
    size_t bytes() const override {        
        return this->_cols * this->_ld * sizeof(T);
    }

    /**
     * @brief Set the array data from host memory.
     * @param hostData Pointer to the host data to copy from.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void set(const T* hostData, cudaStream_t stream = 0) override {
        
        if (this->_ld == 1) cudaMemcpyAsync(this->_ptr.get(), hostData, bytes(), cudaMemcpyHostToDevice, stream);
        else cudaMemcpy2DAsync(
                this->_ptr.get(), this->_ld * sizeof(T),
                hostData, sizeof(T),
                sizeof(T), this->_cols,
                cudaMemcpyHostToDevice, stream
            );
    }

    /**
     * @brief Get the array data to host memory.
     * @param hostData Pointer to the host memory to copy to.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void get(T* hostData, cudaStream_t stream = 0) const override {
        if (this->_ld == 1)
            cudaMemcpyAsync(hostData, this->_ptr.get(), bytes(), cudaMemcpyDeviceToHost, stream);
        else cudaMemcpy2DAsync(
                hostData, sizeof(T),
                this->_ptr.get(), this->_ld * sizeof(T),
                sizeof(T), this->_cols,
                cudaMemcpyDeviceToHost, stream
            );
    }

    /**
     * @brief Set the array data from another CudaArray.
     * @param src Source CudaArray to copy from.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void set(const CudaArray<T>& src, cudaStream_t stream = 0) override {
        if (this->_ld == 1 && src.getLD() == 1) {
            cudaMemcpyAsync(this->_ptr.get(), src.data(), bytes(), cudaMemcpyDeviceToDevice, stream);
        } else {
            cudaMemcpy2DAsync(
                this->_ptr.get(), this->_ld * sizeof(T),
                src.data(), src.getLD() * sizeof(T),
                sizeof(T), this->_cols,
                cudaMemcpyDeviceToDevice, stream
            );
        }
    }

    /**
     * @brief Get the array data to another CudaArray.
     * @param dst Destination CudaArray to copy to.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void get(CudaArray<T>& dst, cudaStream_t stream = 0) const override {
        if (this->_ld == 1 && dst.getLD() == 1) {
            cudaMemcpyAsync(dst.data(), this->_ptr.get(), bytes(), cudaMemcpyDeviceToDevice, stream);
        } else {
            cudaMemcpy2DAsync(
                dst.data(), dst.getLD() * sizeof(T),
                this->_ptr.get(), this->_ld * sizeof(T),
                sizeof(T), this->_cols,
                cudaMemcpyDeviceToDevice, stream
            );
        }
    }

};



// --- Helper Functions and Macros for Testing ---

/**
 * @brief Helper function to check for CUDA errors and exit on failure.
 * @param err The cudaError_t value to check.
 * @param file The file name where the error occurred.
 * @param line The line number where the error occurred.
 */
void checkCudaErrors(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Macro to wrap CUDA function calls for easy error checking.
 */
#define CHECK_CUDA_ERROR(err) checkCudaErrors(err, __FILE__, __LINE__)

/**
 * @brief A helper function to verify if two vectors are identical.
 * @tparam T The data type of the vectors.
 * @param expected The vector containing the expected values.
 * @param result The vector containing the test results.
 * @param test_name A string to identify the test.
 */
template <typename T>
void verifyVectors(const std::vector<T>& expected, const std::vector<T>& result, const std::string& test_name) {
    if (expected == result) {
        std::cout << "✅ " << test_name << " successful." << std::endl;
    } else {
        std::cout << "❌ " << test_name << " failed." << std::endl;
        std::cerr << "Expected size: " << expected.size() << ", Result size: " << result.size() << std::endl;
    }
}

/**
 * @brief Runs all tests for a specific data type.
 * @tparam T The data type to test (e.g., int, float).
 */
template <typename T>
void runTests() {
    std::cout << "--- Running tests for type " << typeid(T).name() << " ---" << std::endl;

    // --- 2D Array Tests ---
    std::cout << "\n## Testing CudaArray2D" << std::endl;
    const size_t rows_2d = 4;
    const size_t cols_2d = 3;
    std::vector<T> host_data_2d(rows_2d * cols_2d);
    std::iota(host_data_2d.begin(), host_data_2d.end(), static_cast<T>(1));

    try {
        CudaArray2D<T> device_array_2d(rows_2d, cols_2d);
        std::cout << "CudaArray2D created with dimensions " << device_array_2d._rows << "x" << device_array_2d._cols << ", LD: " << device_array_2d.getLD() << std::endl;
        device_array_2d.set(host_data_2d.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        std::vector<T> host_result_2d(rows_2d * cols_2d);
        device_array_2d.get(host_result_2d.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        verifyVectors(host_data_2d, host_result_2d, "CudaArray2D set/get");
    } catch (const std::runtime_error& e) {
        std::cerr << "CudaArray2D Test Failed: " << e.what() << std::endl;
    }

    // --- 2D Subarray Test ---
    std::cout << "\n## Testing CudaArray2D Subarray View" << std::endl;
    try {
        CudaArray2D<T> parent_array(rows_2d, cols_2d);
        parent_array.set(host_data_2d.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        const size_t startRow = 1;
        const size_t startCol = 1;
        const size_t subHeight = 2;
        const size_t subWidth = 2;
        CudaArray2D<T> subArray(parent_array, startRow, startCol, subHeight, subWidth);
        
        std::vector<T> retrievedSubArray(subHeight * subWidth);
        subArray.get(retrievedSubArray.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        std::vector<T> expected_subarray;
        for (size_t c = startCol; c < startCol + subWidth; ++c) {
            for (size_t r = startRow; r < startRow + subHeight; ++r) {
                expected_subarray.push_back(host_data_2d[c * rows_2d + r]);
            }
        }
        verifyVectors(expected_subarray, retrievedSubArray, "CudaArray2D Subarray");
    } catch (const std::runtime_error& e) {
        std::cerr << "CudaArray2D Subarray Test Failed: " << e.what() << std::endl;
    }

    // --- 1D Array Tests ---
    std::cout << "\n## Testing CudaArray1D" << std::endl;
    const size_t length_1d = 8;
    std::vector<T> host_data_1d(length_1d);
    std::iota(host_data_1d.begin(), host_data_1d.end(), static_cast<T>(100));

    try {
        CudaArray1D<T> device_array_1d(length_1d);
        device_array_1d.set(host_data_1d.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        std::vector<T> host_result_1d(length_1d);
        device_array_1d.get(host_result_1d.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        verifyVectors(host_data_1d, host_result_1d, "CudaArray1D set/get");
    } catch (const std::runtime_error& e) {
        std::cerr << "CudaArray1D Test Failed: " << e.what() << std::endl;
    }

    // --- 1D Extraction from 2D Array Test ---
    std::cout << "\n## Testing CudaArray1D Extraction from CudaArray2D" << std::endl;
    try {
        CudaArray2D<T> parent_2d(rows_2d, cols_2d);
        parent_2d.set(host_data_2d.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Test column extraction
        int col_index = 1;
        CudaArray1D<T> extracted_col(parent_2d, col_index, IndexType::Column);
        std::vector<T> host_col_result(extracted_col.size());
        extracted_col.get(host_col_result.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        std::vector<T> expected_col;
        for (size_t r = 0; r < rows_2d; ++r) {
            expected_col.push_back(host_data_2d[col_index * rows_2d + r]);
        }
        verifyVectors(expected_col, host_col_result, "CudaArray1D extraction of a column");
        
        // Test row extraction
        int row_index = 1;
        CudaArray1D<T> extracted_row(parent_2d, row_index, IndexType::Row);
        std::vector<T> host_row_result(extracted_row.size());
        extracted_row.get(host_row_result.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        std::vector<T> expected_row;
        for (size_t c = 0; c < cols_2d; ++c) {
            expected_row.push_back(host_data_2d[c * rows_2d + row_index]);
        }
        verifyVectors(expected_row, host_row_result, "CudaArray1D extraction of a row");
    } catch (const std::runtime_error& e) {
        std::cerr << "CudaArray1D Extraction Test Failed: " << e.what() << std::endl;
    }
}

int main() {
    // Check for CUDA device
    int deviceCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found. Exiting." << std::endl;
        return 1;
    }
    
    std::cout << "Starting thorough testing of CudaArray classes..." << std::endl;

    // Run tests for different data types
    runTests<int>();
    std::cout << "\n========================================\n" << std::endl;
    runTests<float>();
    
    std::cout << "\nAll tests complete." << std::endl;
    
    CHECK_CUDA_ERROR(cudaDeviceReset());
    return 0;
}




