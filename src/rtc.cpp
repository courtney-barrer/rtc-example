#include "span_cast.hpp"
#include "span_format.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/chrono.h>

#include <span>
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include <atomic>
#include <span>
#include <string_view>

namespace nb = nanobind;

/**
 * @brief A template struct representing an updatable value.
 *
 * This struct provides functionality to store and update a value of type T.
 * It maintains two copies of the value, referred to as "current" and "next".
 * The "current" value can be accessed and modified using various member functions and operators.
 * The "next" value can be updated using the `update` function.
 * The `commit` function can be used to make the "next" value the new "current" value.
 *
 * @tparam T The type of the value to be stored and updated.
 */
template<typename T>
struct updatable {
    std::array<T, 2> values; /**< An array to store the two copies of the value. */
    /**
     * @brief An atomic pointer to the current value.
     *
     * This pointer is atomic to allow for thread-safe access and modification of the current value.
     *
     */
    T* current_; /**< A pointer to the current value. */
    T* next_; /**< A pointer to the next value. */
    bool has_changed; /**< A flag indicating whether the value has changed. */

    /**
     * @brief Default constructor.
     *
     * Initializes the values array, sets the current and next pointers to the first element of the array,
     * and sets the has_changed flag to false.
     */
    updatable()
        : values{}
        , current_(&values[0])
        , next_(&values[1])
        , has_changed(false)
    {}

    /**
     * @brief Constructor with initial value.
     *
     * Initializes the values array with the given value, sets the current and next pointers to the first element of the array,
     * and sets the has_changed flag to false.
     *
     * @param value The initial value.
     */
    updatable(T value)
        : values{value, value}
        , current_(&values[0])
        , next_(&values[1])
        , has_changed(false)
    {}

    /// Get a reference to the current value.
    T& current() { return *current_; }

    /// Get a const reference to the current value.
    T const& current() const { return *current_; }

    /// Get a reference to the next value.
    T& next() { return *next_; }

    /// Get a const reference to the next value.
    T const& next() const { return *next_; }

    /// Get a reference to the current value.
    T& operator*() { return *current_; }

    /// Get a const reference to the current value.
    T const& operator*() const { return *current_; }

    /// Get a pointer to the current value.
    T* operator->() { return current_; }

    /// Get a const pointer to the current value.
    T const* operator->() const { return current_; }

    /**
     * @brief Update the next value.
     *
     * This function updates the next value with the given value and keep the information that a new value is available.
     *
     * @param value The new value.
     */
    void update(T value)
    {
        *next_ = value;
        has_changed = true;
    }

    /**
     * @brief Set the has_changed flag to true.
     *
     * This function is useful when the next value has been updated directly without using the `update` function.
     */
    void set_changed() { has_changed = true; }

    /**
     * @brief Commit the changes.
     *
     * This function makes the next value the new current value.
     * If the has_changed flag is true, it also swaps the current and next pointers.
     */
    void commit()
    {
        if (has_changed) {
            std::swap(current_, next_);
            has_changed = false;
        }
    }

    /// Overloaded stream operator.
    friend std::ostream& operator<<(std::ostream& os, const updatable& u) {
        return os << "updatable(current = " << u.current()
           << " @ " << (u.current_ - u.values.data())
           << ", has_new = " << u.has_changed << ")";
    }
};

/**
 * @brief A dummy Real-Time Controller.
 */
struct RTC {

    updatable<std::span<const float>> slope_offsets; /**< container for slope offsets. */
    updatable<float> gain; /**< value for gain. */
    updatable<float> offset; /**< value for offset. */

    std::atomic<bool> commit_asked = false; /**< Flag indicating if a commit is requested. */

    /**
     * @brief Default constructor for RTC.
     */
    RTC() = default;

    /**
     * @brief Performs computation using the current RTC values.
     * @param os The output stream to log some informations.
     */
    void compute(std::ostream& os)
    {
        os << "computing with " << (*this) << '\n';

        // Do some computation here...

        // When computation is done, check if a commit is requested.
        if (commit_asked) {
            commit();
            commit_asked = false;
        }
    }

    /**
     * @brief Sets the slope offsets.
     * @param new_offsets The new slope offsets to set.
     */
    void set_slope_offsets(std::span<const float> new_offsets) {
        slope_offsets.update(new_offsets);
    }

    /**
     * @brief Sets the gain.
     * @param new_gain The new gain to set.
     */
    void set_gain(float new_gain) {
        gain.update(new_gain);
    }

    /**
     * @brief Sets the offset.
     * @param new_offset The new offset to set.
     */
    void set_offset(float new_offset) {
        offset.update(new_offset);
    }

    /**
     * @brief Commits the updated values of slope offsets, gain, and offset.
     *
     * This function should only be called when RTC is not running.
     * Otherwise, call request_commit to ask for a commit to be performed after  the next iteration.
     *
     */
    void commit() {
        slope_offsets.commit();
        gain.commit();
        offset.commit();

        std::cout << "commit done\n";
    }

    /**
     * @brief Requests a commit of the updated values.
     */
    void request_commit() {
        commit_asked = true;
    }

    /// Overloaded stream operator.
    friend std::ostream& operator<<(std::ostream& os, const RTC& rtc) {
        os << "RTC(\n\tgain = " << rtc.gain << ",\n\toffset = " << rtc.offset << ",\n\tslope_offsets = " << rtc.slope_offsets << "\n)";
        return os;
    }
};
/**
 * @brief Runs the RTC asynchronously until a stop signal is received.
 *
 * This function runs the RTC asynchronously until a stop signal is received. It periodically checks the value of the `command` atomic variable and calls the `compute` function of the RTC if the `command` is true. It also prints the iteration count every 10 iterations.
 *
 * @param stop_token The stop token used to check if a stop signal is received.
 * @param command The atomic boolean variable indicating whether the RTC should be executed.
 * @param rtc The RTC object.
 * @param os The output stream to write the log messages.
 * @param period The period to sleep if the compute function finishes early.
 */
void run_async(std::stop_token stop_token, std::atomic<bool>& command, RTC& rtc, std::ostream& os, std::chrono::microseconds period)
{
    std::size_t count = 0;
    os << "Running @ " << (1e6 / period.count()) << "Hz...\n";
    while (!stop_token.stop_requested()) {

        auto start_time = std::chrono::steady_clock::now();

        if (command){
            if (++count % 10 == 0)
                os << "iteration " << count << '\n';
            rtc.compute(os);
        }

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        if (elapsed_time < period)
            std::this_thread::sleep_for(period - elapsed_time);

    }
    os << "Exited\n";
}

/**
 * @brief Represents an asynchronous runner for RTC operations.
 *
 * The AsyncRunner class provides functionality to control the execution of RTC operations
 * in an asynchronous manner. It allows pausing, resuming, starting, and stopping the execution
 * of RTC operations. It also provides a method to flush the output stream.
 */
struct AsyncRunner {

    RTC& rtc; /**< Reference to the RTC object. */
    std::atomic<bool> command; /**< Flag indicating the current command state. */
    std::jthread t; /**< Thread for executing the RTC operations asynchronously. */
    std::stringstream ss; /**< Output stream for storing the results of RTC operations. */
    std::chrono::microseconds period; /**< Time period between RTC operations. */

    /**
     * @brief Constructs an AsyncRunner object.
     *
     * @param rtc The RTC object to be controlled.
     * @param period The time period between RTC operations.
     */
    AsyncRunner(RTC& rtc, std::chrono::microseconds period)
        : rtc(rtc)
        , command(false)
        , t()
        , ss()
        , period(period)
    {}

    ~AsyncRunner() {
        if (t.joinable()) {
            stop();
        }
    }

    /// Gets the current state of the AsyncRunner.
    std::string state()
    {
        std::stringstream ss;
        ss << std::boolalpha
                  << "running: " << t.joinable()
                  << ", command: " << command.load()
                  << ", stop requested: " << t.get_stop_token().stop_requested()
                  << '\n';
        return ss.str();
    }
    /// Pauses the execution of RTC operations.
    void pause() {
        command = false;
    }

    /// Resumes the execution of RTC operations.
    void resume() {
        command = true;
    }

    /// Starts the execution of RTC operations asynchronously.
    void start() {
        if (t.joinable())
            stop();

        command = true;
        t = std::jthread(run_async, std::ref(command), std::ref(rtc), std::ref(ss), period);
    }

    /// Stops the execution of RTC operations.
    void stop() {
        t.request_stop();
        t.join();
    }

    /// Flushes the output stream and returns the flushed content.
    std::string flush() {
        auto res = ss.str();
        ss.str("");
        return res;
    }

};

NB_MODULE(_rtc, m) {
    using namespace nb::literals;

    nb::class_<RTC>(m, "RTC")
        .def(nb::init<>())
        .def("compute", &RTC::compute)
        .def("set_slope_offsets", &RTC::set_slope_offsets)
        .def("set_gain", &RTC::set_gain)
        .def("set_offset", &RTC::set_offset)
        .def("commit", &RTC::commit)
        .def("request_commit", &RTC::request_commit)
        .def("__repr__", [](const RTC& rtc) {
            std::stringstream ss;
            ss << rtc;
            return ss.str();
        });

    nb::class_<AsyncRunner>(m, "AsyncRunner")
        .def(nb::init<RTC&, std::chrono::microseconds>(), nb::arg("rtc"), nb::arg("period") = std::chrono::microseconds(1000), "Constructs an AsyncRunner object.")
        .def("start", &AsyncRunner::start)
        .def("stop", &AsyncRunner::stop)
        .def("pause", &AsyncRunner::pause)
        .def("resume", &AsyncRunner::resume)
        .def("state", &AsyncRunner::state)
        .def("flush", &AsyncRunner::flush);
}
