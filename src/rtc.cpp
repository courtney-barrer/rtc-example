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
#include <vector>
#include  <fstream>

#include <BMCApi.h>
#include "FliSdk.h"

namespace nb = nanobind;
using namespace std;



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



/// @brief A PID controller with  Anti-windup
class PIDController {
public:
    PIDController()
        : kp(0.0), ki(0.0), kd(0.0), integral_limit(1.0) {} // Default constructor

    // later iterations could make these vectors so different gains per mode.
    // or alternatively just intialize various classes
    PIDController(double kp, double ki, double kd, double integral_limit = 1.0)
        : kp(kp), ki(ki), kd(kd), integral_limit(integral_limit) {}

    std::vector<double> compute_pid(const std::vector<double>& setpoint, const std::vector<double>& measured) {
        if (setpoint.size() != measured.size()) {
            throw std::invalid_argument("Input vectors must be of the same size.");
        }

        // Resize integrals and prev_errors to match the size of the input vectors
        if (integrals.size() != setpoint.size()) {
            integrals.resize(setpoint.size(), 0.0);
            prev_errors.resize(setpoint.size(), 0.0);
        }

        std::vector<double> output(setpoint.size());
        for (size_t i = 0; i < setpoint.size(); ++i) {
            // NOTE: this takes convention that error is opposite sign 
            // to the reconstruction
            double error = setpoint[i] - measured[i];
            integrals[i] += error;

            // Anti-windup: clamp the integrative term
            if (integrals[i] > integral_limit.current()) {
                integrals[i] = integral_limit.current();
            } else if (integrals[i] < -integral_limit.current()) {
                integrals[i] = -integral_limit.current();
            }

            double derivative = error - prev_errors[i];
            output[i] = kp.current() * error + ki.current() * integrals[i] + kd.current() * derivative;
            prev_errors[i] = error;
        }

        // Commit any updated PID gains
        kp.commit();
        ki.commit();
        kd.commit();
        integral_limit.commit();

        return output;
    }

    void reset() {
        std::fill(integrals.begin(), integrals.end(), 0.0);
        std::fill(prev_errors.begin(), prev_errors.end(), 0.0);
    }

    updatable<double> kp;
    updatable<double> ki;
    updatable<double> kd;
    updatable<double> integral_limit; // Updatable anti-windup limit

    std::vector<double> integrals;
    std::vector<double> prev_errors;
};


// // gets value at indicies for a input vector (span)
// template<typename DestType, typename T>
// std::vector<DestType> getValuesAtIndices(std::span<T> data_span, std::span<const int> indices_span) {
//     std::vector<DestType> values;
//     values.reserve(indices_span.size());

//     for (size_t index : indices_span) {
//         assert(index < data_span.size() && "Index out of bounds!"); // Ensure index is within bounds
//         values.push_back(static_cast<DestType>(data_span[index]));
//     }

//     return values;
// }

// Gets value at indices for an input vector
template<typename DestType, typename T>
std::vector<DestType> getValuesAtIndices(const std::vector<T>& data, const std::vector<int>& indices) {
    std::vector<DestType> values;
    values.reserve(indices.size());

    for (int index : indices) {
        assert(index >= 0 && index < static_cast<int>(data.size()) && "Index out of bounds!"); // Ensure index is within bounds
        values.push_back(static_cast<DestType>(data[index]));
    }

    return values;
}


/// @brief multiply vector v by matrix r
/// @param v  - vector 
/// @param r  - flattened matrix 
/// @param result - result of multiplication
// void matrix_vector_multiply(const std::vector<auto>& v, updatable<std::span<float>>& r, std::vector<double>& result) {
//     // auto fine if input double or float vector - casting issues possible if int vector input
//     std::size_t M = result.size();
//     std::size_t N = v.size();
//     auto R = r.current(); // get current value of reconstructor
//     // Ensure the result vector has the correct size
//     result.resize(M);

//     for (std::size_t i = 0; i < M; ++i) {
//         result[i] = 0.0;
//         for (std::size_t j = 0; j < N; ++j) {
//             result[i] += R[i * N + j] * v[j];
//         }
//     }
// }
// USE VECTOR INSTEAD OF SPAN
void matrix_vector_multiply(
    const std::vector<double>& v,         // Input vector
    const std::vector<float>& R,          // Matrix as a vector (assuming row-major order)
    std::vector<double>& result           // Output result vector
) {
    std::size_t M = result.size(); // Number of rows in the matrix
    std::size_t N = v.size();      // Number of columns in the matrix

    // Ensure the result vector has the correct size
    result.resize(M, 0.0);

    for (std::size_t i = 0; i < M; ++i) {
        result[i] = 0.0;
        for (std::size_t j = 0; j < N; ++j) {
            result[i] += R[i * N + j] * v[j];
        }
    }
}



/// @brief read in a csv file and store as pointer to array
/// @param filePath 
/// @return array 
double* readCSV(const std::string& filePath) {
    static const int ARRAY_SIZE = 140;
    double values[ARRAY_SIZE]; // non-static array to return pointer to

    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::string line;
    int count = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        int columnCount = 0;

        while (std::getline(ss, item, ',')) {
            columnCount++;
        }

        if (columnCount != 1) {
            throw std::runtime_error("File must contain exactly one column");
        }

        // Parse the single column value
        double value;
        std::stringstream(line) >> value;
        values[count] = value;
        count++;
    }

    if (count != ARRAY_SIZE) {
        throw std::runtime_error("File must contain exactly 140 rows");
    }

    return values;
}




/*
state_structure - loop , camera simulation, dm_simulation,  
camera mode - 

then do get image function and update dm function that checks for simulated states



*/

/* @brief to hold state flags of the RTC for checking things */
struct rtc_state_struct {
    bool close_loop_mode = false; // AO in closed loop
    bool dm_simulation_mode = false; // Do we want to simulate the DM?
    bool camera_simulation_mode = false; // Do we want to simulate the camera with synthetic images?
    bool signal_simulation_mode = false; // Do we want to simulate the input signal to the controller?

    // Default constructor
    rtc_state_struct() noexcept = default; // Using defaulted constructor
};

/*@brief to hold simulated signals that can be injected in the RTC*/
struct simulated_signals_struct{
    //put simulated 
    int val;
};

/*@brief to hold camera settings*/
struct camera_settings_struct {
    double det_dit = 0.0016; // detector integration time (s)
    double det_fps = 600.0; // frames per second (Hz)
    std::string det_gain="medium"; // "low" or "medium" or "high"
    bool det_crop_enabled = false; //true/false
    bool det_tag_enabled = false; //true/false
    std::string det_cropping_rows="0-639"; //"r1-r2" where  r1 multiple of 4, r2 multiple 4-1
    std::string det_cropping_cols="0-511"; //"c1-c2" where c1 multiple of 32, c2 multiple 32-1
    uint16_t image_height = 640; // i.e. rows in image 
    uint16_t image_width = 512;// i.e. cols in image
    uint32_t full_image_length = 327680; //640*512;

    // Default constructor
    camera_settings_struct() noexcept = default; // Using defaulted constructor
};

/*@brief to hold things for ZWFS signal processing and projection onto modes/DM commands
we have an array of reconstructor matricies here to allow flexibility in control implementation
CM, R_TT, R_HO can be used to go directly from processed intensities to DM commands
I2M can be used to go from processed intensities to a modal basis
M2C can be used to go from a pre-defined modal basis to DM commands 

bias, I0, flux_norm are for signal processing 

with nano bind there were issues with these being span's - so changed them all to vectors! 
*/
struct phase_reconstuctor_struct {
    /*
    we have an array of reconstructor matricies here to allow flexibility in control implementation
    CM, R_TT, R_HO can be used to go directly from processed intensities to DM commands
    I2M can be used to go from processed intensities to a modal basis
    M2C can be used to go from a pre-defined modal basis to DM commands 

    bias, I0, flux_norm are for signal processing 

    with nano bind there were issues with these being span's - so changed them all to vectors! 
    */
    updatable<std::vector<float>> IM; /**<unfiltered interaction matrix  */
    updatable<std::vector<float>> CM; /**< control matrix (~ M2C @ I2M.T) signal intensities to dm commands  */
    updatable<std::vector<float>> R_TT; /**< tip/tilt reconstructor */
    updatable<std::vector<float>> R_HO; /**< higher-order reconstructor */
    updatable<std::vector<float>> I2M; /**< intensity (signal) to mode matrix */
    //std::span<float> I2M_a // again there is a bug with updatable I2M as with I0...
    updatable<std::vector<float>> M2C; /**< mode to DM command matrix. */ 

    updatable<std::vector<uint16_t>> bias; /**< bias. */
    updatable<std::vector<float>> I0; /**< reference intensity with FPM in. */
    updatable<float> flux_norm; /**< for normalizing intensity across detector. */
};

/*@brief to hold pixel indicies for different pupil regions*/
struct pupil_regions_struct {
    updatable<std::vector<int>> pupil_pixels; /**< pixels inside the active pupil. */
    updatable<std::vector<int>> secondary_pixels; /**< pixels inside secondary obstruction   */
    updatable<std::vector<int>> outside_pixels; /**< pixels outside the active pupil obstruction (but not in secondary obstruction) */
};



void apply_camera_settings( FliSdk* fli, camera_settings_struct cm){
    // NEED TO TEST IN SYDNEY ON CAMERA 
    // does not check if in simulation mode!  
    
    uint32_t _full_image_length_new; 
    //double fps = 0;

    fli->stop();

    // crop first
    fli->serialCamera()->sendCommand("set cropping off"); //FliCamera_sendCommand("set cropping off");
    if (cm.det_crop_enabled) {
        //set cropping and enable
        fli->serialCamera()->sendCommand("set cropping rows "+ cm.det_cropping_rows);
        fli->serialCamera()->sendCommand("set cropping columns "+ cm.det_cropping_cols);
        fli->serialCamera()->sendCommand("set cropping on");
    }

    if (cm.det_tag_enabled) {
        // makes first pixels correspond to frame number and other info 
        // 
        //TO DO: should make corresponding mask for this to be added to 
        //pixel_filter if this is turned on to ensure frame count etc
        //does not get interpretted as intensities. 
        fli->serialCamera()->sendCommand("set imagetags on");  
    } else{
        fli->serialCamera()->sendCommand("set imagetags off");  
    }

    fli->serialCamera()->sendCommand("set cropping rows "+ cm.det_cropping_rows);
    fli->serialCamera()->sendCommand("set cropping cols "+ cm.det_cropping_cols);

    fli->serialCamera()->sendCommand("set cropping cols "+ cm.det_cropping_cols);

    fli->serialCamera()->sendCommand("set sensitivity "+ cm.det_gain);

    //set fps
    fli->serialCamera()->setFps(cm.det_fps);

    //set int
    fli->serialCamera()->sendCommand("set tint " + std::to_string(cm.det_dit));

    //fli->serialCamera()->getFps(fps);
    //cout << "fps despues = " << fps << endl;

    fli->update();

    //uint16_t width, height;
    fli->getCurrentImageDimension(cm.image_width, cm.image_height);
    cout << "image width  =  " << cm.image_width << endl;
    cout << "image height  =  " << cm.image_height << endl;
    _full_image_length_new = static_cast<uint32_t>(cm.image_width) * static_cast<uint32_t>(cm.image_height);

    // _full_image_length_new can be used to update simulated signals etc before appending
    if (_full_image_length_new != cm.full_image_length){
        cout << "_full_image_length_new != cm.full_image_length " << endl; 
        // update the full image length
        cm.full_image_length = _full_image_length_new;
    }

    fli->start();

}

/**
 * @brief A dummy Real-Time Controller.
 */
struct RTC {
    // -------- HARDWARE 
    // object to interact with DM 
    DM hdm = {};
    const size_t dm_size = 140 ; // # actuators on BMC multi-3.5 DM
    std::string dm_serial_number = "17DW019#053"; // USYD ="17DW019#122",  ANU = "17DW019#053";
    std::vector<uint32_t> map_lut; // (BMC multi-3.5 DM)

    // object to interact with camera
    FliSdk* fli = {};

    // -------- SYSTEM MODES
    rtc_state_struct rtc_state;
    camera_settings_struct camera_settings;

    // -------- CONTROL SYSTEMS 
    phase_reconstuctor_struct reco;
    pupil_regions_struct regions;
    PIDController pid; // object to apply PID control to signals

    //telemetry
    size_t telemetry_cnt = 0;

    std::atomic<bool> commit_asked = false; /**< Flag indicating if a commit is requested. */

    /**
     * @brief Default constructor for RTC.
     */
    RTC() //= default;
    {

        /* open BMC DM */  
        //---------------------
        BMCRC	rv = NO_ERR;
        rv = BMCOpen(&hdm, dm_serial_number.c_str());
        // error check
        if (rv != NO_ERR) {
            std::cerr << "Error " << rv << " opening the driver type " << hdm.Driver_Type << ": ";
            std::cerr << BMCErrorString(rv) << std::endl << std::endl;

            // PUT IN SIMULATION MODE 

        }else{
            map_lut.resize(MAX_DM_SIZE);
            uint32_t k = 0;
            // init map lut to zeros (copying examples from BMC)
            for(k=0; k<(int)hdm.ActCount; k++) { 
                map_lut[k] = 0;
            }
            // then we load the default map
            rv = BMCLoadMap(&hdm, NULL, map_lut.data()); 

        }


        // /* open Camera */ 
        //---------------------
        // note: I tried using fakecamera from 
        // /opt/FirstLightImaging/FliSdk/Examples/API_C++/FliFakeCamera
        // but failed to get this to run properly
        FliSdk* fli = new FliSdk();
        std::cout << "Detection of grabbers..." << std::endl;
        vector<string> listOfGrabbers = fli->detectGrabbers();
        std::cout << "Detection of cameras..." << std::endl;
        vector<string> listOfCameras = fli->detectCameras();
        if(listOfGrabbers.size() == 0)
        {
            cout << "No grabber detected, exit." << endl;
            //return;
            // PUT IN SIMULATION MODE 

        }else if (listOfCameras.size() == 0){

            cout << "No camera detected, exit." << endl;
            //put in simulation mode 

        }else{
            int i = 0;
            std::string cameraName;
            for(auto c : listOfCameras)
            {
                cout << "- " << i << " -> " << c << endl;
                ++i;
            }

            int selectedCamera = 0;
            cout << "Which camera to use? (0, 1, ...) " << endl;
            cin >> selectedCamera;

            if(selectedCamera >= listOfCameras.size())
                selectedCamera = 0;

            cameraName = listOfCameras[selectedCamera];
            cout << "Setting camera " << cameraName << endl;
            //take the first camera in the list
            fli->setCamera(cameraName);

            // --------------- RESTORE DEFAULT SETTINGS ON INIT -----
            //fli->serialCamera()->sendCommand("restorefactory");

            cout << "Setting mode Full." << endl;
            //set full mode
            fli->setMode(FliSdk::Mode::Full);

            cout << "Update the SDK..." << endl;
            //update
            fli->update();
            cout << "Done." << endl;


            fli->start();
            cout << "Done." << endl;
            fli->imageProcessing()->enableAutoClip(true);

        }

    }
    

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
    
    void set_slope_offsets(std::span<const float> new_offsets) {
        slope_offsets.update(new_offsets);
    }
     */
    /**
     * @brief Sets the gain.
     * @param new_gain The new gain to set.
    
    void set_gain(float new_gain) {
        gain.update(new_gain);
    }
     */

    /**
     * @brief Sets the offset.
     * @param new_offset The new offset to set.
    
    void set_offset(float new_offset) {
        offset.update(new_offset);
    }
     */

    /**
     * @brief Commits the updated values of slope offsets, gain, and offset.
     *
     * This function should only be called when RTC is not running.
     * Otherwise, call request_commit to ask for a commit to be performed after  the next iteration.
     *
     */
    void commit() {
        //slope_offsets.commit();
        //gain.commit();
        //offset.commit();

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
        os << "RTC(\n\tPUTSTUFFHERE)"; //"RTC(\n\tgain = " << rtc.gain << ",\n\toffset = " << rtc.offset << ",\n\tslope_offsets = " << rtc.slope_offsets << "\n)";
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
        //.def("set_slope_offsets", &RTC::set_slope_offsets)
        //.def("set_gain", &RTC::set_gain)
        //.def("set_offset", &RTC::set_offset)
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