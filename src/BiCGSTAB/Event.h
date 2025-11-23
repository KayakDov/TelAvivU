/**
 * @file Event.h
 * @brief RAII wrapper for CUDA event management using smart pointers.
 */

#ifndef EVENT_H
#define EVENT_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <memory>
#include "../deviceArrays/headers/GpuArray.h"

/**
 * @class Event
 * @brief A robust RAII wrapper around a CUDA event using std::unique_ptr.
 *
 * This class manages a CUDA event (`cudaEvent_t`) through a smart pointer
 * with a custom deleter. It supports:
 *  - creation
 *  - destruction (automatic via RAII)
 *  - renewal (recreate event)
 *  - recording on a CUDA stream
 *  - making a stream wait on the event
 *
 * The event is always either:
 *  - owned exclusively by this class, or
 *  - null (if moved-from)
 *
 * No destructor is needed, and no `valid` flag is used.
 */
class Event {
public:

    /**
     * @brief Constructs a new CUDA event.
     *
     * The event is created with `cudaEventDisableTiming` for minimal overhead.
     *
     * @throws std::runtime_error if CUDA fails to create the event.
     */
    Event();

    /**
     * @brief Recreates the CUDA event.
     *
     * The previous event is destroyed automatically via the custom deleter.
     *
     * @throws std::runtime_error if CUDA fails to recreate the event.
     */
    void renew();

    /**
     * @brief Records the event on the CUDA stream associated with the given Handle.
     *
     * @param h A Handle object containing a CUDA stream.
     * @throws std::runtime_error if the event is null or if `cudaEventRecord` fails.
     */
    void record(const Handle& h) const;

    /**
     * @brief Makes the stream in the given Handle wait until this event completes.
     *
     * @param h A Handle object containing a CUDA stream.
     * @throws std::runtime_error if the event is null or if `cudaStreamWaitEvent` fails.
     */
    void wait(const Handle& h) const;

    /**
     * @brief Renew the CUDA event and immediately record it on the given stream.
     *
     * Useful for checkpointing or marking phase boundaries in asynchronous pipelines.
     *
     * @param h The Handle whose CUDA stream is used.
     */
    void renewAndRecord(const Handle& h);

    /// @name Deleted copy operations
    /// @{
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    /// @}

    /// @name Move semantics
    /// @{
    Event(Event&&) noexcept = default;
    Event& operator=(Event&&) noexcept = default;
    /// @}

private:

    /**
     * @struct EventDeleter
     * @brief Custom deleter that destroys the CUDA event when the unique_ptr resets.
     */
    struct EventDeleter {
        void operator()(cudaEvent_t e) const noexcept {
            if (e) CHECK_CUDA_ERROR(cudaEventDestroy(e));
        }
    };

    /// @brief Smart pointer owning the CUDA event.
    using EventPtr = std::unique_ptr<std::remove_pointer<cudaEvent_t>::type, EventDeleter>;

    EventPtr event; ///< The owned CUDA event (or null if moved-from).
};

#endif // EVENT_H
