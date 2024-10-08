
#include <push_record.hpp>
#include "span_cast.hpp"
#include "span_format.hpp"

#include <cstdint>
#include <updatable.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/vector.h>

#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include <atomic>
#include <span>
#include <string_view>
#include <vector>
#include <fstream>
#include <unordered_set>

#include <omp.h>

#include <BMCApi.h>
#include "FliSdk.h"

namespace nb = nanobind;
using namespace std;

// ssh rtc@150.203.88.114

/*
put PID and leaky integrator in reco struct.?
include method - modal, zonal etc
*/



class PIDController {
public:
    PIDController()
    //: kp(0.0), ki(0.0), kd(0.0), lower_limit(0.0), upper_limit(1.0), setpoint(0.0) {}
    : kp(1, 0.0), ki(1, 0.0), kd(1, 0.0), lower_limit(1, 0.0), upper_limit(1, 1.0), setpoint(1, 0.0),
      integrals(1, 0.0), prev_errors(1, 0.0) {}

    PIDController(const std::vector<double>& kp, const std::vector<double>& ki, const std::vector<double>& kd, const std::vector<double>& lower_limit, const std::vector<double>& upper_limit,const std::vector<double>& setpoint)
    : kp(kp), ki(ki), kd(kd), output(kp.size(), 0.0), integrals(kp.size(), 0.0), prev_errors(kp.size(), 0.0), lower_limit(lower_limit), upper_limit(upper_limit), setpoint(setpoint) {}

    std::vector<double> process( const std::vector<double>& measured) {
        // Ensure all vectors have the same size - this could lead to unpredictable behaviour!
        // especially if in nanobinded python someone changes something! 
        size_t size = setpoint.size();
        std::string error_message;

        // faster way. less verbose
        // if (kp.size() != size || ki.size() != size || kd.size() != size ||
        //     lower_limit.size() != size || upper_limit.size() != size || measured.size() != size) {
        //     throw std::invalid_argument("All input vectors must be of the same size.");
        //     }
        // slower - more verbose

        if (measured.size() != size) {
            throw std::invalid_argument("Input vector size must match setpoint.size()");
        }
        if (kp.size() != size) {
            error_message += "kp ";
        }
        if (ki.size() != size) {
            error_message += "ki ";
        }
        if (kd.size() != size) {
            error_message += "kd ";
        }
        if (lower_limit.size() != size) {
            error_message += "lower_limit ";
        }
        if (upper_limit.size() != size) {
            error_message += "upper_limit ";
        }
        if (measured.size() != size) {
            error_message += "measured ";
        }

        if (!error_message.empty()) {
            throw std::invalid_argument("Input vectors of incorrect size: " + error_message);
        }

        if (integrals.size() != size) {
            cout << "integrals.size() != size.. reinitializing integrals, prev_errors and output to zero with correct size" << endl; 
            integrals.resize(size, 0.0);
            prev_errors.resize(size, 0.0);
            output.resize(size, 0.0);
        }

        //std::vector<double> output(setpoint.size());
        for (size_t i = 0; i < setpoint.size(); ++i) {
            double error =  setpoint[i] - measured[i]; // should this be other way round? Needs to be consistent with leaky integrator!
            integrals[i] += error;

            // Anti-windup: clamp the integrative term within upper and lower limits
            integrals[i] = std::clamp(integrals[i], lower_limit[i], upper_limit[i]);

            double derivative = error - prev_errors[i];
            output[i] = kp[i] * error + ki[i] * integrals[i] + kd[i] * derivative;
            prev_errors[i] = error;
        }

        return output;
    }


    void reset() {
        std::fill(integrals.begin(), integrals.end(), 0.0f);
        std::fill(prev_errors.begin(), prev_errors.end(), 0.0f);
        std::fill(output.begin(), output.end(), 0.0f);
    }

    std::vector<double> kp;
    std::vector<double> ki;
    std::vector<double> kd;
    std::vector<double> upper_limit;
    std::vector<double> lower_limit;
    std::vector<double> setpoint;
    std::vector<double> output;
    std::vector<double> integrals;
    std::vector<double> prev_errors;
};


// class LeakyIntegrator {
// public:
//     // Constructor to initialize the leaky integrator with a given rho vector and limits for anti-windup
//     LeakyIntegrator()
//         : rho(0.0), lower_limit(-1.0), upper_limit(1.0) {}

//     LeakyIntegrator(const std::vector<double>& rho, const std::vector<double>& lower_limit, const std::vector<double>& upper_limit)
//         : rho(rho), output(rho.size(), 0.0), lower_limit(lower_limit), upper_limit(upper_limit) {
//         if (rho.empty()) {
//             throw std::invalid_argument("Rho vector cannot be empty.");
//         }
//         if (lower_limit.size() != rho.size() || upper_limit.size() != rho.size()) {
//             throw std::invalid_argument("Lower and upper limit vectors must match rho vector size.");
//         }
//     }

//     // Method to process a new input and update the output
//     std::vector<double> process(const std::vector<double>& input) {

//         //error checks 
//         if (input.size() != rho.size()) {
//             throw std::invalid_argument("Input vector size must match rho vector size.");
//         }

//         size_t size = rho.size();
//         std::string error_message;

//         cout << "size" << size << endl; 
        
//         if (rho.size() != size) {
//             error_message += "rho ";
//         }
//         if (lower_limit.size() != size) {
//             error_message += "lower_limit ";
//         }
//         if (upper_limit.size() != size) {
//             error_message += "upper_limit ";
//         }

//         if (!error_message.empty()) {
//             throw std::invalid_argument("Input vectors of incorrect size: " + error_message);
//         }

//         if (output.size() != size) {
//             cout << "size" << size  << " outputsize" << output.size() << endl; 
//             cout << "output.size() != size.. reinitializing output to zero with correct size" << endl; 
//             output.resize(size, 0.0);
//         }

//         // process 
//         for (size_t i = 0; i < rho.size(); ++i) {
//             output[i] = rho[i] * output[i] + input[i];
//             // Apply anti-windup by clamping the output within the specified limits
//             output[i] = std::clamp(output[i], lower_limit[i], upper_limit[i]);
//         }
//         return output;
//     }

//     // Method to reset the integrator output to zero
//     void reset() {
//         std::fill(output.begin(), output.end(), 0.0);
//     }

//     // Members are public
//     std::vector<double> rho;        // 1 - time constants for the leaky integrator per dimension
//     std::vector<double> output;     // Current output of the integrator per dimension
//     std::vector<double> lower_limit;  // Lower output limits for anti-windup
//     std::vector<double> upper_limit;  // Upper output limits for anti-windup
// };
class LeakyIntegrator {
public:
    // Default constructor: Initializes vectors with size 1 and default values
    LeakyIntegrator()
        : rho(1, 0.0), kp(1, 1.0), lower_limit(1, -1.0), upper_limit(1, 1.0), output(1, 0.0) {}

    // Constructor to initialize with rho, kp, lower, and upper limits
    LeakyIntegrator(const std::vector<double>& rho, const std::vector<double>& kp, 
                    const std::vector<double>& lower_limit, const std::vector<double>& upper_limit)
        : rho(rho), kp(kp), output(rho.size(), 0.0), lower_limit(lower_limit), upper_limit(upper_limit) {
        if (rho.empty() || kp.empty()) {
            throw std::invalid_argument("Rho and kp vectors cannot be empty.");
        }
        if (rho.size() != kp.size() || lower_limit.size() != rho.size() || upper_limit.size() != rho.size()) {
            throw std::invalid_argument("Rho, kp, lower, and upper limit vectors must match in size.");
        }
    }

    // Method to process a new input and update the output with proportional gain kp
    std::vector<double> process(const std::vector<double>& input) {

        // Error checks 
        if (input.size() != rho.size() || input.size() != kp.size()) {
            throw std::invalid_argument("Input vector size must match rho and kp vector size.");
        }

        size_t size = rho.size();
        std::string error_message;

        if (rho.size() != size) {
            error_message += "rho ";
        }
        if (kp.size() != size) {
            error_message += "kp ";
        }
        if (lower_limit.size() != size) {
            error_message += "lower_limit ";
        }
        if (upper_limit.size() != size) {
            error_message += "upper_limit ";
        }

        if (!error_message.empty()) {
            throw std::invalid_argument("Input vectors of incorrect size: " + error_message);
        }

        if (output.size() != size) {
            output.resize(size, 0.0);
        }

        // Process each element with leaky integration and proportional gain
        for (size_t i = 0; i < rho.size(); ++i) {
            output[i] = rho[i] * output[i] + kp[i] * input[i];
            // Apply anti-windup by clamping the output within the specified limits
            output[i] = std::clamp(output[i], lower_limit[i], upper_limit[i]);
        }

        return output;
    }

    // Method to reset the integrator output to zero
    void reset() {
        std::fill(output.begin(), output.end(), 0.0);
    }

    // Members are public
    std::vector<double> rho;         // 1 - time constants for the leaky integrator per dimension
    std::vector<double> kp;          // Proportional gain per dimension
    std::vector<double> output;      // Current output of the integrator per dimension
    std::vector<double> lower_limit; // Lower output limits for anti-windup
    std::vector<double> upper_limit; // Upper output limits for anti-windup
};

// gets value at indicies for a input vector (span)
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


// check for DM 
bool isOutOfBounds(const std::vector<double>& vec) {
    for (const auto& value : vec) {
        if (value > 1.0 || value < 0.0) {
            return true;
        }
    }
    return false;
}

// Gets value at indices for an input vector
template<typename DestType, typename T>
void getValuesAtIndices(std::vector<DestType>& values, const T & data, std::span<int> indices) { // std::span<const int> indices
    values.resize(indices.size());

    size_t i = 0;
    for (int index : indices) {
        assert(index >= 0 && index < static_cast<int>(data.size()) && "Index out of bounds!"); // Ensure index is within bounds
        values[i++] = static_cast<DestType>(data[index]);
    }
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
    const std::vector<float>& v,         // Input vector
    const std::vector<float>& R,          // Matrix as a vector (assuming row-major order)
    std::vector<double>& result           // Output result vector
) {
    
    std::size_t N = v.size();      // Number of columns in the matrix
    std::size_t M = R.size() / N; // Number of rows in the matrix
    // ^^ must use R.size to calculate M since this is updatable

    // SHOULD PUT CHECKS HERE THAT N/R.size() IS INTEGER > 0 

    // Ensure the result vector has the correct size
    result.resize(M, 0.0);

    for (std::size_t i = 0; i < M; ++i) {
        result[i] = 0.0;
        for (std::size_t j = 0; j < N; ++j) {
            result[i] += R[i * N + j] * v[j];
        }
    }
}


// if input vector v is doubles ( also trialing here to parallise outer loop using omp)
void matrix_vector_multiply_double(
    const std::vector<double>& v,       // Input vector of doubles
    const std::vector<float>& R,        // Matrix as a vector of floats (row-major order)
    std::vector<double>& result         // Output result vector of doubles
) {
    std::size_t N = v.size();      // Number of columns in the matrix
    std::size_t M = R.size() / N;  // Number of rows in the matrix

    // Ensure the result vector has the correct size
    result.resize(M, 0.0);

    // Parallelize the loop over rows
    #pragma omp parallel for
    for (std::size_t i = 0; i < M; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < N; ++j) {
            sum += static_cast<double>(R[i * N + j]) * v[j];
        }
        result[i] = sum;
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
    // signals used if things are in simulation mode
    std::vector<uint16_t> simulated_image;//(327680, 0); //used if rtc_state_struct.camera_simulation_mode = true. initialized to 0 over 640*512 pixels
    std::vector<double> simulated_dm_cmd;//(140, 0); // used if rtc_state_struct.dm_simulation_mode = true. initialized to 0 over 140 actuators
    std::vector<float> simulated_image_err_signal; // used if rtc_state_struct.signal_simulation_mode = true. This is the processed signal.

    // Default constructor
    simulated_signals_struct() noexcept = default; // Using defaulted constructor
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
    
    std::vector<uint16_t> dark;
    std::vector<uint16_t> bad_pixels; // hold bad pixels



    // Default constructor
    //camera_settings_struct() noexcept = default; // Using defaulted constructor
    camera_settings_struct() noexcept
        : dark(full_image_length, 0),  // Initialize dark with zeros
          bad_pixels()                 // Initialize bad_pixels as an empty vector
    {}
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

    // works for vector<int> but not vector<float>?
    updatable<std::vector<float>> IM; /**<unfiltered interaction matrix  */
    
    updatable<std::vector<float>> CM; /**< control matrix (~ M2C @ I2M.T) signal intensities to dm commands  */
    updatable<std::vector<float>> R_TT; /**< tip/tilt reconstructor */
    updatable<std::vector<float>> R_HO; /**< higher-order reconstructor */
    updatable<std::vector<float>> I2M; /**< intensity (signal) to mode matrix */

    updatable<std::vector<float>> I2M_TT; 
    updatable<std::vector<float>> I2M_HO;

    //std::span<float> I2M_a // again there is a bug with updatable I2M as with I0...
    updatable<std::vector<float>> M2C; /**< mode to DM command matrix. */
    // if we want serpate explictly mode space of TT and HO 
    updatable<std::vector<float>> M2C_TT; /**< mode to DM command matrix. */
    updatable<std::vector<float>> M2C_HO; /**< mode to DM command matrix. */

    updatable<std::vector<uint16_t>> bias; /**< bias. */
    updatable<std::vector<float>> I0; /**< reference intensity with FPM in. */
    updatable<std::vector<float>> N0; /**< reference intensity with FPM out. */
    updatable<float> flux_norm; /**< for normalizing intensity across detector. */
    
    void commit_all(){
        IM.commit();
        CM.commit();
        R_TT.commit();
        R_HO.commit();
        I2M.commit();
        I2M_TT.commit();
        I2M_HO.commit();
        M2C.commit();
        M2C_TT.commit();
        M2C_HO.commit();
        bias.commit();
        I0.commit();
        N0.commit();
        flux_norm.commit();
    }
    
};

/*@brief to hold pixel indicies for different pupil regions*/
struct pupil_regions_struct {
    updatable<std::vector<int>> pupil_pixels; /**< pixels inside the active pupil. */
    updatable<std::vector<int>> secondary_pixels; /**< pixels inside secondary obstruction   */
    updatable<std::vector<int>> outside_pixels; /**< pixels outside the active pupil obstruction (but not in secondary obstruction) */
    updatable<std::vector<int>> local_region_pixels; /* defining the local region used to build reconstructor */
    void commit_all(){
        pupil_pixels.commit();
        secondary_pixels.commit();
        outside_pixels.commit();
        local_region_pixels.commit(); 
    }

    // Templated method to update a member by name (for simplicity when nanobinding so dont have to spell out each function)
    // template<typename T>
    // bool update_member(const std::string& member_name, const T& new_value) {
    //     static const std::unordered_map<std::string, std::function<void(const T&)>> update_map = {
    //         {"pupil_pixels", [this](const T& value) { this->pupil_pixels.update(value); }},
    //         {"secondary_pixels", [this](const T& value) { this->secondary_pixels.update(value); }},
    //         {"outside_pixels", [this](const T& value) { this->outside_pixels.update(value); }}
    //     };

    //     auto it = update_map.find(member_name);
    //     if (it != update_map.end()) {
    //         it->second(new_value);
    //         return true;
    //     } else {
    //         return false; // Member name not found
    //     }
    // }
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
    std::string dm_serial_number = "17DW019#122"; // USYD ="17DW019#122",  ANU = "17DW019#053";
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
    LeakyIntegrator leakyInt; // object to apply PID control to signals

    // -------- SIMULATED SIGNALS (for if in simulation mode )
    simulated_signals_struct rtc_simulation_signals;


    // variables 
    uint16_t current_frame_number; // the current frame number
    int32_t previous_frame_number; // the previous frame number 
    //(signed 32 int since at uint16 overflow the previous frame number needs to go to -1)

    // main variables in image->dm command pipeline (all things that would be good in telemetry)
    std::vector<int32_t> image_vector;
    std::vector<float> image_in_pupil; // image intensities filtered within active pupil  
    std::vector<float> image_setpoint; // reference image filtered within active pupil  
    std::vector<float> image_err_signal; // processed image error signal (normalized)

    // matrix_vector_multiplication returns double.. so all these should be doubles and PID should accept
    // doubles as input.
    std::vector<double> dm_cmd_err; // holds the DM command offset (error) from the reference flat DM surface
    std::vector<double> TT_cmd_err; // holds the DM Tip-Tilt command offset (error) from the reference flat DM surface
    std::vector<double> HO_cmd_err; // holds the DM Higher Order command offset (error) from the reference flat DM surface
    std::vector<double> mode_err; // holds mode error from matrix multiplication of I2M with processed signal 
    std::vector<double> mode_err_TT; // holds mode error specifically from matrix multiplication of I2M_TT with processed signal 
    std::vector<double> mode_err_HO; // holds mode error specifically from matrix multiplication of I2M_HO with processed signal 
    std::vector<double> dm_cmd; // final DM command 

    std::vector<double> dm_disturb; // a disturbance that we can add to the DM 
    std::vector<double> dm_flat; // calibrated DM flat 
    //std::vector<double> pid_setpoint // set-point of PID controller


    //telemetry
    size_t telemetry_cnt = 0; // save telemtry for how many iterations? (>0 to save)

    std::atomic<bool> commit_asked = false; /**< Flag indicating if a commit is requested. */

    /**
     * @brief Default constructor for RTC.
     */
    RTC() //= default;
    {
        // any DM commands to length 140 
        std::vector<double> dm_cmd_err(140, 0.0); // holds the DM command offset (error) from the reference flat DM surface
        std::vector<double> TT_cmd_err(140, 0.0); // holds the DM Tip-Tilt command offset (error) from the reference flat DM surface
        std::vector<double> HO_cmd_err(140, 0.0); // holds the DM Higher Order command offset (error) from the reference flat DM surface
        std::vector<double> dm_cmd(140, 0.0); // final DM command 
        std::vector<double> dm_disturb(140,0); // disturbance we can put on DM 
        /* open BMC DM */
        //---------------------
        BMCRC	rv = NO_ERR;
        rv = BMCOpen(&hdm, dm_serial_number.c_str());
        // error check
        if (rv != NO_ERR) {
            std::cerr << "Error " << rv << " opening the driver type " << hdm.Driver_Type << ": ";
            std::cerr << BMCErrorString(rv) << std::endl << std::endl;
            cout << "Putting DM in simulation mode.." <<endl;
            // PUT IN SIMULATION MODE
            rtc_state.dm_simulation_mode = true;

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

        //FliSdk* fli = new FliSdk();
        this->fli = new FliSdk();
        
        std::cout << "Detection of grabbers..." << std::endl;
        vector<string> listOfGrabbers = fli->detectGrabbers();
        std::cout << "Detection of cameras..." << std::endl;
        vector<string> listOfCameras = fli->detectCameras();
        if(listOfGrabbers.size() == 0)
        {
            cout << "No grabber detected, exit. Putting camera in simulation mode.." << endl;
            // PUT IN SIMULATION MODE
            rtc_state.camera_simulation_mode = true;

        }else if (listOfCameras.size() == 0){

            cout << "No camera detected, exit. Putting camera in simulation mode.." << endl;
            // PUT IN SIMULATION MODE
            rtc_state.camera_simulation_mode = true;

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

    void apply_camera_settings(){
        // configures camera according to current RTC.camera_settings struct
        
        if (rtc_state.camera_simulation_mode){
            cout << "rtc_state.camera_simulation_mode = true. Therefore no actual camera configuration done." << endl;
        } else{
            
            uint32_t _full_image_length_new;
            //fli->start();

            
            //double fps = 0;
		    //fli->serialCamera()->getFps(fps);
            //cout << fps << endl;
            //double fps = 0;

            fli->stop();
            
            // crop first
            fli->serialCamera()->sendCommand("set cropping off"); //FliCamera_sendCommand("set cropping off");
            if (camera_settings.det_crop_enabled) {
                //set cropping and enable
                fli->serialCamera()->sendCommand("set cropping rows "+ camera_settings.det_cropping_rows);
                fli->serialCamera()->sendCommand("set cropping columns "+ camera_settings.det_cropping_cols);
                fli->serialCamera()->sendCommand("set cropping on");
            }

            if (camera_settings.det_tag_enabled) {
                // makes first pixels correspond to frame number and other info
                //
                //TO DO: should make corresponding mask for this to be added to
                //pixel_filter if this is turned on to ensure frame count etc
                //does not get interpretted as intensities.
                fli->serialCamera()->sendCommand("set imagetags on");
            } else{
                fli->serialCamera()->sendCommand("set imagetags off");
            }

            fli->serialCamera()->sendCommand("set cropping rows "+ camera_settings.det_cropping_rows);
            fli->serialCamera()->sendCommand("set cropping cols "+ camera_settings.det_cropping_cols);

            fli->serialCamera()->sendCommand("set cropping cols "+ camera_settings.det_cropping_cols);

            fli->serialCamera()->sendCommand("set sensitivity "+ camera_settings.det_gain);

            //set fps
            fli->serialCamera()->setFps(camera_settings.det_fps);

            //set int
            fli->serialCamera()->sendCommand("set tint " + std::to_string(camera_settings.det_dit));

            //fli->serialCamera()->getFps(fps);
            //cout << "fps despues = " << fps << endl;

            fli->update();

            //uint16_t width, height;
            fli->getCurrentImageDimension(camera_settings.image_width, camera_settings.image_height);
            cout << "image width  =  " << camera_settings.image_width << endl;
            cout << "image height  =  " << camera_settings.image_height << endl;
            _full_image_length_new = static_cast<uint32_t>(camera_settings.image_width) * static_cast<uint32_t>(camera_settings.image_height);

            // _full_image_length_new can be used to update simulated signals etc before appending
            if (_full_image_length_new != camera_settings.full_image_length){
                cout << "_full_image_length_new != camera_settings.full_image_length " << endl;
                // update the full image length
                camera_settings.full_image_length = _full_image_length_new;
            }

            fli->start();
            
        }
    }

    int close_all( ){

        BMCClose(&hdm);

        fli->stop();

        delete fli;

        return 0;
    }


    /// @brief standard way to send a command to the DM in the context of the RTC. cmds should be vectors and here we convert to a pointer before sending it.
    /// this also checks if the DM is in simulation within the RTC struct.
    /// @return a pointer to the uint16_t image
    void send_dm_cmd(std::vector<double> cmd) {
        if (not rtc_state.dm_simulation_mode) {
                    // get cmd pointer
            double *cmd_ptr = cmd.data();
            BMCSetArray(&hdm, cmd_ptr, map_lut.data());
        }
    }

    /// @brief standard way to get the last frame from the camera within RTC scope - this gets if we are in simulation mode
    /// @return a pointer to the uint16_t image
    uint16_t* poll_last_image() {
        uint16_t* rawImage; // Declare rawImage outside of the conditional blocks

        if (rtc_state.camera_simulation_mode) {
            rawImage = &rtc_simulation_signals.simulated_image[0]; // Point to address of the first element in simulated image vector
        } else {
            rawImage = (uint16_t*)fli->getRawImage(); // Assuming fli is a valid pointer to an object with the getRawImage method
        }

        return rawImage;
    }


    void reduce_image(const uint16_t* raw_image,  std::vector<int32_t>& result) {
        // Ensure that the dark frame has the same size as the image
        assert(camera_settings.dark.size() == camera_settings.full_image_length);
        
        // Ensure the result vector is preallocated with the correct size
        assert(result.size() == camera_settings.full_image_length);

        // Convert bad_pixels vector to an unordered_set for O(1) lookup times (this should be done outside of this function!)
        std::unordered_set<size_t> bad_pixel_set(camera_settings.bad_pixels.begin(), camera_settings.bad_pixels.end());

        // Process each pixel: subtract dark frame, set to zero if it's a bad pixel
        for (size_t i = 0; i < camera_settings.full_image_length; ++i) {
            // Subtract dark frame and store result as int32_t
            int32_t processed_value = static_cast<int32_t>(raw_image[i]) - static_cast<int32_t>(camera_settings.dark[i]);

            // Check if the pixel is in the set of bad pixels
            if (bad_pixel_set.find(i) != bad_pixel_set.end()) {
                processed_value = 0;  // Set bad pixels to zero
            }
            
            result[i] = processed_value;
        }
        
    }

    // changed to pass as reference 
    void process_image(std::vector<float>& im,  std::vector<float>& im_setpoint , std::vector<float>& signal) //std::span<const float> im, std::span<const float> im_setpoint, std::vector<float>& signal)
    {
        if (not rtc_state.signal_simulation_mode){
            //std::vector<float> signal(signal_size);
            for (size_t i = 0; i < im.size(); ++i) {

                signal[i] = static_cast<float>(im[i]) / reco.flux_norm.current() - im_setpoint[i];// image_setpoint[i];

            }

        } else if( rtc_simulation_signals.simulated_image_err_signal.size() == im.size()){

            signal = rtc_simulation_signals.simulated_image_err_signal; // create a copy

        } else{

            cout << "!!!!!!!!!!! simulatied_signal.size() != signal_size !!!!!!!!!!!!!!!" << endl;
            cout << "setting signal = vector of zeros size = im.size()" << endl;
            signal.resize(im.size(), 0);
        };

    }



    // ------------- FUNCTIONS TO TEST THINGS ---------------------
    std::vector<uint32_t>  im2vec_test(){
        // to test things 

        // get image
        uint16_t* raw_image = poll_last_image();

        // convert to vector
        std::vector<uint32_t> image_vector(raw_image, raw_image + camera_settings.full_image_length);
        return(image_vector);

    }

    std::vector<int32_t>  reduceImg_test(){
        // to test things 

        // get image
        uint16_t* raw_image = poll_last_image();

        //std::vector<uint32_t> image_vector(raw_image, raw_image + camera_settings.full_image_length);
        std::vector<int32_t> image_vector(camera_settings.full_image_length);

        reduce_image(raw_image,  image_vector);
        // convert to vector
        
        return(image_vector);
    }

    std::vector<float> im2filtered_im_test(){
        // to test things 
        // size of filtered signal may change while RTC is running
        size_t signal_size = regions.pupil_pixels.current().size();

        // have to define here to keep in scope of telemetry
        static std::vector<float> image_err_signal(signal_size); // <- should this be static

        // get image
        uint16_t* raw_image = poll_last_image();

        // convert to vector
        std::vector<uint32_t> image_vector(raw_image, raw_image + camera_settings.full_image_length);

        //static uint16_t frame_cnt = image_vector[0]; // to check if we are on a new frame

        //std::vector<float> image_in_pupil;//<- init at top struct
        getValuesAtIndices(image_in_pupil, image_vector, regions.pupil_pixels.current()  ) ; // image

        //std::vector<float> image_setpoint;//<- init at top struct
        //getValuesAtIndices(image_setpoint, reco.I0.current(),  regions.pupil_pixels.current()  ); // set point intensity

        return image_in_pupil;
    }

    std::vector<float> im2filteredref_im_test(){
        // to test things 
        // size of filtered signal may change while RTC is running
        size_t signal_size = regions.pupil_pixels.current().size();

        // have to define here to keep in scope of telemetry
        static std::vector<float> image_err_signal(signal_size); // <- should this be static

        // get image
        uint16_t* raw_image = poll_last_image();

        // convert to vector
        std::vector<uint32_t> image_vector(raw_image, raw_image + camera_settings.full_image_length);

        //static uint16_t frame_cnt = image_vector[0]; // to check if we are on a new frame

        //std::vector<float> image_in_pupil;//<- init at top struct
        //getValuesAtIndices(image_in_pupil, image_vector, regions.pupil_pixels.current()  ) ; // image

        //std::vector<float> image_setpoint;//<- init at top struct
        getValuesAtIndices(image_setpoint, reco.I0.current(),  regions.pupil_pixels.current()  ); // set point intensity

        return image_setpoint;
    }


    std::vector<float> process_im_test(){
        // to test things 
        // size of filtered signal may change while RTC is running
        size_t signal_size = regions.pupil_pixels.current().size();

        // have to define here to keep in scope of telemetry
        static std::vector<float> image_err_signal(signal_size); // <- should this be static

        // get image
        uint16_t* raw_image = poll_last_image();

        // convert to vector
        //std::vector<uint32_t> image_vector(raw_image, raw_image + camera_settings.full_image_length);

        std::vector<int32_t> image_vector(camera_settings.full_image_length);

        reduce_image(raw_image,  image_vector);
        //static uint16_t frame_cnt = image_vector[0]; // to check if we are on a new frame

        //std::vector<float> image_in_pupil;//<- init at top struct
        getValuesAtIndices(image_in_pupil, image_vector, regions.pupil_pixels.current()  ) ; // image

        //std::vector<float> image_setpoint;//<- init at top struct
        getValuesAtIndices(image_setpoint, reco.I0.current(),  regions.pupil_pixels.current()  ); // set point intensity

        process_image( image_in_pupil, image_setpoint , image_err_signal);

        return image_err_signal;
    }

    /**
    * @brief Performs a single computation using the current RTC values.
    * this can be used to test compute and is nanobinded to python (unlike compute) 
    * */
    std::vector<double> single_compute(int return_case=4)
    {
        // This should be a default argunment 
        //const std::string to_return="full";

        // size of filtered signal may change while RTC is running
        size_t signal_size = regions.pupil_pixels.current().size();

        // have to define here to keep in scope of telemetry
        static std::vector<float> image_err_signal(signal_size); // <- should this be static

        // get image
        uint16_t* raw_image = poll_last_image();

        // convert to vector
        std::vector<uint32_t> image_vector(raw_image, raw_image + camera_settings.full_image_length);

        //static uint16_t frame_cnt = image_vector[0]; // to check if we are on a new frame

        //std::vector<float> image_in_pupil;//<- init at top struct
        getValuesAtIndices(image_in_pupil, image_vector, regions.pupil_pixels.current()  ) ; // image

        //std::vector<float> image_setpoint;//<- init at top struct
        getValuesAtIndices(image_setpoint, reco.I0.current(),  regions.pupil_pixels.current()  ); // set point intensity

        process_image( image_in_pupil, image_setpoint, image_err_signal);

        //return  image_err_signal;
        //matrix_vector_multiply( image_err_signal, reco.CM.current(), dm_cmd_err ) ;

        //Perform element-wise addition
        // for (size_t i = 0; i < dm_size; ++i) {
        //     //cout << flat_dm_array[i] << delta_cmd[i]<< endl ;

        //     cmd[i] = flat_dm_array[i] + delta_cmd[i]; // just roughly offset to center
        // }

        
        //int return_case=4;

        /*
        how to do this case?

            -PID / leaky int has to be turned to case that we  - should check 

            returns final DM command for each case. Test could be with PID, Kp =1 ,others zero 

         */

        switch(return_case){
            // ones - matrix multiplication to cmd or modal space
            case 1: //TT cmd_err
                cout << "TT cmd_err" << endl;
                matrix_vector_multiply( image_err_signal, reco.R_TT.current(), TT_cmd_err ) ;
                return TT_cmd_err ;
                break;
            case 2: //HO cmd_err
                cout << "HO cmd_err" << endl;
                matrix_vector_multiply( image_err_signal, reco.R_HO.current(), HO_cmd_err ) ;
                return HO_cmd_err ;
                break;

            case 3: //mode amplitudes err:
                cout << "mode amplitudes err" << endl;
                // this provides wrong
                cout <<  "sizeof(reco.I2M.current())" << sizeof(reco.I2M.current()) << endl;
                matrix_vector_multiply( image_err_signal, reco.I2M.current(), mode_err ) ;
                return mode_err ;
                break;

            case 4: //"full" DM cmd error:
                cout << "full DM cmd error:" << endl;
                matrix_vector_multiply( image_err_signal, reco.CM.current(), dm_cmd_err ) ;
                return dm_cmd_err ;
                break;

            // teens - apply PID controller in respective spaces 
            case 11: 
                cout << "TT PID output" << endl;
                matrix_vector_multiply( image_err_signal, reco.R_TT.current(), TT_cmd_err ) ;
                pid.process( TT_cmd_err ) ;
                return pid.output ;
                break;

            case 12: //HO cmd_err
                cout << "HO PID output" << endl;
                matrix_vector_multiply( image_err_signal, reco.R_HO.current(), HO_cmd_err ) ;
                pid.process( HO_cmd_err ) ;
                return pid.output ;
                break;

            case 13: //mode amplitudes err:
                cout << "mode amplitudes PID output" << endl;
                matrix_vector_multiply( image_err_signal, reco.I2M.current(), mode_err ) ;
                pid.process( mode_err ) ;
                return pid.output ;
                break;

            case 14: //"full" DM cmd error:
                cout << "full DM cmd PID output:" << endl;
                matrix_vector_multiply( image_err_signal, reco.CM.current(), dm_cmd_err ) ;
                pid.process( dm_cmd_err ) ;
                return pid.output ;
                break;

            // twenties - apply Leaky controller in respective spaces  
            case 21: 
                cout << "TT leaky integrator output" << endl;
                matrix_vector_multiply( image_err_signal, reco.R_TT.current(), TT_cmd_err ) ;
                leakyInt.process( TT_cmd_err ) ;
                return leakyInt.output ;
                break;

            case 22: //HO cmd_err
                cout << "HO leaky integrator output" << endl;
                matrix_vector_multiply( image_err_signal, reco.R_HO.current(), HO_cmd_err ) ;
                leakyInt.process( HO_cmd_err ) ;
                return leakyInt.output ;
                break;

            case 23: //mode amplitudes err:
                cout << "mode amplitudes leaky integrator output" << endl;
                matrix_vector_multiply( image_err_signal, reco.I2M.current(), mode_err ) ;
                leakyInt.process( mode_err ) ;
                return leakyInt.output ;
                break;

            case 24: //"full" DM cmd error:
                cout << "full DM cmd leaky integrator output:" << endl;
                matrix_vector_multiply( image_err_signal, reco.CM.current(), dm_cmd_err ) ;
                leakyInt.process( dm_cmd_err ) ;
                return leakyInt.output ;
                break;


            // Hundreds

            case 113: //mode u_leaky @ M2C:
                cout << "mode amplitudes PID output" << endl;
                matrix_vector_multiply( image_err_signal, reco.I2M.current(), mode_err ) ;
                pid.process( mode_err ) ;
                // note - using DOUBLE matrix_vector_multiply here that also has parallel component
                matrix_vector_multiply_double( pid.output, reco.M2C.current(), dm_cmd_err ) ;
                return dm_cmd_err ;
                break;

            case 123: //mode u_leaky @ M2C:
                cout << "mode amplitudes leaky integrator output" << endl;
                matrix_vector_multiply( image_err_signal, reco.I2M.current(), mode_err ) ;
                leakyInt.process( mode_err ) ;
                // note - using DOUBLE matrix_vector_multiply here that also has parallel component
                matrix_vector_multiply_double( leakyInt.output, reco.M2C.current(), dm_cmd_err ) ;
                return dm_cmd_err ;
                break;

            case 223: // mode u_leaky @ M2C with telemetry

                cout << "mode amplitudes leaky integrator output" << endl;
                matrix_vector_multiply( image_err_signal, reco.I2M.current(), mode_err ) ;
                leakyInt.process( mode_err ) ;
                // note - using DOUBLE matrix_vector_multiply here that also has parallel component
                matrix_vector_multiply_double( leakyInt.output, reco.M2C.current(), dm_cmd_err ) ;


                if (telemetry_cnt > 0){
                    telem_entry entry;

                    entry.image_in_pupil = std::move(image_in_pupil); // im);
        
                    entry.image_err_signal = std::move(image_err_signal);
                    entry.mode_err = std::move(mode_err ); // reconstructed DM command
                    entry.dm_cmd_err = std::move( dm_cmd_err); // final command sent

                    append_telemetry(std::move(entry));

                    --telemetry_cnt;
                }
                return dm_cmd_err ;
                break;

            default:
                cout << "no to_return cases met. Returning CM @ signal" << endl;
                matrix_vector_multiply( image_err_signal, reco.CM.current(), dm_cmd_err ) ;
                return dm_cmd_err;
                break;
        }
        //return dm_cmd_err ; <<- return in switch cases uniquely
    }



    std::vector<double> test22(){
        // get image
        uint16_t* raw_image = poll_last_image();
        // get current frame number and static init previous 
        int32_t current_frame_number = static_cast<int32_t>(raw_image[0]);
        static int32_t previous_frame_number = current_frame_number;

        std::vector<double> dm_cmd(140, 0); // should init somewhere else  
        if (current_frame_number > previous_frame_number){
            // Do some computation here...
            cout << "current frame" << current_frame_number << endl;
            // frame number for raw images from FLI camera is typically unsigned int16 (0-65536)
            // so need to catch case of overflow (current=0, previous = 65535)
            // previous_frame_number needs to be signed in16 (to go negative) while current_frame_number
            // must match raw_image type unsigned int16.
            // update frame number
            if (current_frame_number == 65535){
                previous_frame_number = -1; // catch overflow case for int16 where current=0, previous = 65535
            }else{
                previous_frame_number = current_frame_number;
            }

            // size of filtered signal may change while RTC is running
            size_t signal_size = regions.pupil_pixels.current().size();

            // have to define here since size may change while rtc running 
            static std::vector<float> image_err_signal(signal_size); // <- should this be static

            std::vector<int32_t> image_vector(camera_settings.full_image_length); //should init outside here
            reduce_image(raw_image,  image_vector);
            //static uint16_t frame_cnt = image_vector[0]; // to check if we are on a new frame

            //std::vector<float> image_in_pupil;//<- init at top struct
            getValuesAtIndices(image_in_pupil, image_vector, regions.pupil_pixels.current()  ) ; // image

            //std::vector<float> image_setpoint;//<- init at top struct
            getValuesAtIndices(image_setpoint, reco.I0.current(),  regions.pupil_pixels.current()  ); // set point intensity

            process_image( image_in_pupil, image_setpoint , image_err_signal);

            // //tip/tilt 
            matrix_vector_multiply( image_err_signal, reco.I2M_TT.current(), mode_err_TT ) ;
            pid.process( mode_err_TT ) ;
            
            // // note - using DOUBLE matrix_vector_multiply here that also has parallel component
            matrix_vector_multiply_double( pid.output, reco.M2C_TT.current(), TT_cmd_err ) ;
            cout << "pid output size" << pid.output.size(); 

            // Higher order 
            matrix_vector_multiply( image_err_signal, reco.I2M_HO.current(), mode_err_HO ) ;
            leakyInt.process( mode_err_HO ) ;
            // note - using DOUBLE matrix_vector_multiply here that also has parallel component
            matrix_vector_multiply_double( leakyInt.output, reco.M2C_HO.current(), HO_cmd_err ) ;
            cout << "leaky output size" << leakyInt.output.size();
            cout << "HO cmd_error size" << HO_cmd_err.size();
            
            //cout << HO_cmd_err[0] <<endl;
            
            for (size_t i = 0; i < TT_cmd_err.size(); ++i) {
                //dm_cmd[i] = TT_cmd_err[i] + HO_cmd_err[i];
                dm_cmd[i] = dm_flat[i] + TT_cmd_err[i] + HO_cmd_err[i] + dm_disturb[i];  // comment out HO_cmd_err if desired
                //cout << dm_cmd[i] << endl;
            }

            if (telemetry_cnt > 0){
                telem_entry entry;
                //dm_span_flag[0] = (double)(k % 2) ;
                
                entry.image_in_pupil = image_err_signal; //std::move( image_err_signal );
                entry.e_TT = mode_err_TT; //std::move(mode_err_TT); // the corresponding image 
                entry.e_HO = mode_err_HO; //std::move(mode_err_HO); 
                entry.u_TT = mode_err_HO; //std::move(mode_err_HO);
                entry.u_HO = leakyInt.output; //std::move(leakyInt.output);
                entry.u_TT = pid.output; //std::move(pid.output);
                entry.cmd_HO = HO_cmd_err; //std::move(HO_cmd_err);
                entry.cmd_TT = TT_cmd_err; //std::move(TT_cmd_err);
                entry.dm_disturb = dm_disturb; 
                //cout << dm_flag[0] << endl;

                append_telemetry(std::move(entry));

                --telemetry_cnt;
                
                cout << "telemetry_cnt=" << telemetry_cnt << endl;
                //if there is a new frame append it to 
            }

        }
        else{
            cout << 'no new frame' << endl;
            std::vector<double> dm_cmd(140,0) ; //dm_cmd(140, 0);
        }

        return dm_cmd ;

    }

    void latency_test(){
        //before starting should set DM array to zero or flat position. 
        
        // just switch between actuator pokes based on modulus of j and switch on
        static int k =0; // for counting if poke in or out.

        //if (k=0){
        //    BMCSetArray(&hdm, zerod_dm, map_lut.data());
        //    //init to flat DM 
        //}

        if (telemetry_cnt > 0){

            //uint16_t* raw_image = poll_last_image(); //(uint16_t*)fli->getRawImage(); // Retrieve raw image data
            //std::span<const uint16_t> image_span(raw_image, full_image_length);//rows * cols); // Create a span from the raw image data
            //std::vector<int32_t>
            image_vector = reduceImg_test(); // THIS MAY NOTE WORK NOW USING FULL FRAME DARKS.. CHECK 

            std::vector<float> result(image_vector.size());

            // Iterate through the input_vector and cast each element to float
            for (size_t i = 0; i <image_vector.size(); ++i) {
                result[i] = static_cast<float>(image_vector[i]);
            }

            static int j = image_vector[0]; // Static count variable, only intialize once
            //static std::vector<double> dm_flag(1,0); // hold DM state flag  
            std::vector<double> dm_flag( 1, 0 );
            //std::span<float> dm_span_flag( dm_flag ) ;

            //uint16_t frame_ct = image_span[0];
            // do we record an image in telemetry? 
            // only record the frame if it is new ( dont repeat old frames )
           
            //cout << image_vector[0] << endl;
            if (j < image_vector[0]) {

                j = image_vector[0];
                telem_entry entry;
                //dm_span_flag[0] = (double)(k % 2) ;
                dm_flag[0] = (double)(k % 2) ;
                entry.image_in_pupil = std::move(result); // the corresponding image 
                entry.dm_cmd_err = std::move(dm_flag); // if DM is push or flat 
                //cout << dm_flag[0] << endl;

                append_telemetry(std::move(entry));

                --telemetry_cnt;
                
                cout << "telemetry_cnt=" << telemetry_cnt << endl;
                //if there is a new frame append it to 
                }
            
            //std::cout << "ok" << std::endl;
            
            
            // do we move the every 10 frames?
            if ((telemetry_cnt % 20)==0) {
                // we use telemetry_cnt here instead of j beacuase of tendancy to skip frames
                // (so modules on j skips the mark sometimes)
                // check to poke in or out
                if ((k % 2)==0){
                    
                    cout << "in"<< endl;
                    BMCSetSingle(&hdm, 65, 0.2);
                    
                        }else{
                    BMCSetSingle(&hdm, 65, 0);
                    cout << "out"<< endl;
                    }
                ++k;
                
            
            }
            //j++;
            cout << "j = " << j << endl;
            
            
        }
        
    }
    
    /**
     * @brief Performs computation using the current RTC values.
     * @param os The output stream to log some informations.
     */
    void compute(std::ostream& os)
    {
        os << "computing with " << (*this) << '\n';

        //latency_test() ;


        // get image
        uint16_t* raw_image = poll_last_image();
        // get current frame number and static init previous 
        int32_t current_frame_number = static_cast<int32_t>(raw_image[0]);
        static int32_t previous_frame_number = current_frame_number;

        std::vector<double> dm_cmd(140, 0); // should init somewhere else  
        if (current_frame_number > previous_frame_number){
            // Do some computation here...
            //cout << "current frame" << current_frame_number << endl;

            // frame number for raw images from FLI camera is typically unsigned int16 (0-65536)
            // so need to catch case of overflow (current=0, previous = 65535)
            // previous_frame_number needs to be signed in16 (to go negative) while current_frame_number
            // must match raw_image type unsigned int16.
            // update frame number

            // Get the current time point using the high-resolution clock
            auto now0 = std::chrono::high_resolution_clock::now();

            // Convert the time point to the duration in seconds (double)
            auto t0 = std::chrono::duration<double>(now0.time_since_epoch()).count();

            if (current_frame_number == 65535){
                previous_frame_number = -1; // catch overflow case for int16 where current=0, previous = 65535
            }else{
                previous_frame_number = current_frame_number;
            }

            // size of filtered signal may change while RTC is running
            size_t signal_size = regions.pupil_pixels.current().size();

            // have to define here since size may change while rtc running 
            static std::vector<float> image_err_signal(signal_size); // <- should this be static

            std::vector<int32_t> image_vector(camera_settings.full_image_length); //should init outside here
            reduce_image(raw_image,  image_vector);
            //static uint16_t frame_cnt = image_vector[0]; // to check if we are on a new frame

            //std::vector<float> image_in_pupil;//<- init at top struct
            getValuesAtIndices(image_in_pupil, image_vector, regions.pupil_pixels.current()  ) ; // image

            //std::vector<float> image_setpoint;//<- init at top struct
            getValuesAtIndices(image_setpoint, reco.I0.current(),  regions.pupil_pixels.current()  ); // set point intensity

            process_image( image_in_pupil, image_setpoint , image_err_signal);

            // //tip/tilt 
            matrix_vector_multiply( image_err_signal, reco.I2M_TT.current(), mode_err_TT ) ;
            pid.process( mode_err_TT ) ;
            
            // // note - using DOUBLE matrix_vector_multiply here that also has parallel component
            matrix_vector_multiply_double( pid.output, reco.M2C_TT.current(), TT_cmd_err ) ;
            cout << "pid output size" << pid.output.size(); 

            // Higher order 
            matrix_vector_multiply( image_err_signal, reco.I2M_HO.current(), mode_err_HO ) ;
            leakyInt.process( mode_err_HO ) ;
            // note - using DOUBLE matrix_vector_multiply here that also has parallel component
            matrix_vector_multiply_double( leakyInt.output, reco.M2C_HO.current(), HO_cmd_err ) ;
            cout << "leaky output size" << leakyInt.output.size();
            cout << "HO cmd_error size" << HO_cmd_err.size();
            
            //cout << HO_cmd_err[0] <<endl;
            
            for (size_t i = 0; i < TT_cmd_err.size(); ++i) {
                //dm_cmd[i] = TT_cmd_err[i] + HO_cmd_err[i];
                dm_cmd[i] = dm_flat[i] + TT_cmd_err[i] + HO_cmd_err[i] + dm_disturb[i];  // comment out HO_cmd_err if desired
                //cout << dm_cmd[i] << endl;
            }

            //isOutOfBounds( dm_cmd )

            // get cmd pointer 
            double *cmd_ptr = dm_cmd.data();
            // update DM 
            BMCSetArray(&hdm, cmd_ptr, map_lut.data());

            // Get the current time point using the high-resolution clock
            auto now1 = std::chrono::high_resolution_clock::now();

            // Convert the time point to the duration in seconds (double)
            auto t1 = std::chrono::duration<double>(now1.time_since_epoch()).count();


            if (telemetry_cnt > 0){
                telem_entry entry;
                //dm_span_flag[0] = (double)(k % 2) ;
                
                entry.image_in_pupil = image_err_signal; //std::move( image_err_signal );
                entry.e_TT = mode_err_TT; //std::move(mode_err_TT); // the corresponding image 
                entry.e_HO = mode_err_HO; //std::move(mode_err_HO); 
                entry.u_TT = mode_err_HO; //std::move(mode_err_HO);
                entry.u_HO = leakyInt.output; //std::move(leakyInt.output);
                entry.u_TT = pid.output; //std::move(pid.output);
                entry.cmd_HO = HO_cmd_err; //std::move(HO_cmd_err);
                entry.cmd_TT = TT_cmd_err; //std::move(TT_cmd_err);
                entry.dm_disturb = dm_disturb; 
                entry.t0 = t0;
                entry.t1 = t1;
                //cout << dm_flag[0] << endl;

                append_telemetry(std::move(entry));

                --telemetry_cnt;
                
                cout << "telemetry_cnt=" << telemetry_cnt << endl;
                //if there is a new frame append it to 
            }

        }


        
            
        //     // convert to vector
        //     std::vector<uint32_t> image_vector(raw_image, raw_image + camera_settings.full_image_length);

        //     // size of filtered signal may change while RTC is running
        //     size_t signal_size = regions.pupil_pixels.current().size();

        //     // have to define here to keep in scope of teletry
        //     static std::vector<float> image_err_signal(signal_size); // <- should this be static

        //     //static uint16_t frame_cnt = image_vector[0]; // to check if we are on a new frame

        //     //std::vector<float> image_in_pupil;//<- init at top struct
        //     getValuesAtIndices(image_in_pupil, image_vector, regions.pupil_pixels.current()  ) ; // image

        //     //std::vector<float> image_setpoint;//<- init at top struct
        //     getValuesAtIndices(image_setpoint, reco.I0.current(),  regions.pupil_pixels.current()  ); // set point intensity

        //     process_image( image_in_pupil, image_setpoint, image_err_signal);

        //     /* TO DO: 
        //     define TT_cmd_err, HO_cmd_err at beginning of struct
        //     test addition onto flat command
            
        //     */ 
        //     //matrix_vector_multiply( image_err_signal, reco.R_TT.current(), TT_cmd_err )
        //     //matrix_vector_multiply( image_err_signal, reco.R_TT.current(), HO_cmd_err )

        //     // PID and leaky to TT_cmd_err and HO_cmd_err (in cmd space)
        //     pid.process(  TT_cmd_err   ) ;// use PID for tip/tilt
        //     leakyInt.process( HO_cmd_err  ) ;//use leaky integrator for HO 

        //     /*
        //     //cmd =  flat_cmd + TT_cmd_err + HO_cmd_err
        //     //Perform element-wise addition
        //     for (size_t i = 0; i < dm_size; ++i) {
        //     //     //cout << flat_dm_array[i] << delta_cmd[i]<< endl ;

        //         dm_cmd[i] = flat_dm_array[i] + LeakyInt.output[i] + pid.output[i]; // just roughly offset to center
        //      }

        //     // send DM cmd 
        //     // get cmd pointer 
        //     double *cmd_ptr = dm_cmd.data();

        //     BMCSetArray(&hdm, cmd_ptr, map_lut.data());
        //     */
        //}
        
        // When computation is done, check if a commit is requested.
        if (commit_asked) {
            commit();
            commit_asked = false;
        }

    }


    void enable_telemetry(size_t iteration_nb) {
        telemetry_cnt = iteration_nb;
    }

    void commit() {
        //slope_offsets.commit();
        //gain.commit();
        //offset.commit();

        reco.IM.commit() ;
        reco.CM.commit();
        reco.R_TT.commit();
        reco.R_HO.commit();
        reco.I2M.commit();
        reco.I2M_TT.commit();
        reco.I2M_HO.commit();
        reco.M2C.commit();
        reco.M2C_TT.commit();
        reco.M2C_HO.commit();
        reco.I0.commit();
        reco.N0.commit();
        reco.flux_norm.commit();

        regions.pupil_pixels.commit();
        regions.secondary_pixels.commit();
        regions.outside_pixels.commit();
        regions.local_region_pixels.commit();

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


void bind_telemetry(nb::module_& m);

NB_MODULE(_rtc, m) {
    using namespace nb::literals;

    register_updatable(m);

    nb::class_<RTC>(m, "RTC")
        .def(nb::init<>())
        .def_rw("rtc_state", &RTC::rtc_state)
        .def_rw("camera_settings", &RTC::camera_settings)
        .def_rw("reco", &RTC::reco)
        .def_rw("regions", &RTC::regions)
        .def_rw("rtc_simulation_signals", &RTC::rtc_simulation_signals)

        .def("apply_camera_settings", &RTC::apply_camera_settings)

        .def("enable_telemetry", &RTC::enable_telemetry)
        // main variables in image->dm command pipeline (all things that would be good in telemetry)
        .def_rw("telemetry_cnt", &RTC::telemetry_cnt)
        .def_rw("image_in_pupil" ,    &RTC::image_in_pupil)
        .def_rw("image_setpoint"  ,    &RTC::image_setpoint)
        .def_rw("image_err_signal",  &RTC::image_err_signal)
        .def_rw("dm_cmd_err" , &RTC::dm_cmd_err)
        .def_rw("TT_cmd_err",  &RTC::TT_cmd_err)
        .def_rw("HO_cmd_err" , &RTC::HO_cmd_err)
        .def_rw("mode_err" , &RTC::mode_err)
        .def_rw("mode_err_TT" , &RTC::mode_err)
        .def_rw("mode_err_HO" , &RTC::mode_err)
        .def_rw("dm_cmd" , &RTC::dm_cmd)
        .def_rw("dm_disturb", &RTC::dm_disturb)
        .def_rw("dm_flat", &RTC::dm_flat)

        // controllers
        //.def("PID", &RTC.pid)
        .def_rw("pid", &RTC::pid, "PID Controller")
        .def_rw("LeakyInt", &RTC::leakyInt, "Leaky Integrator")

        // for testing individually 
        .def("close_all", &RTC::close_all) 
        .def("send_dm_cmd",&RTC::send_dm_cmd)
        .def("im2vec_test", &RTC::im2vec_test)
        .def("reduceImg_test", &RTC::reduceImg_test)
        .def("im2filtered_im_test", &RTC::im2filtered_im_test)
        .def("poll_last_image", &RTC::poll_last_image)
        .def("im2filteredref_im_test", &RTC::im2filteredref_im_test)
        .def("process_im_test", &RTC::process_im_test)
        .def("latency_test",&RTC::latency_test)
        
        .def("test22", &RTC::test22)
        .def("compute", &RTC::compute)
        //.def("set_slope_offsets", &RTC::set_slope_offsets)
        //.def("set_gain", &RTC::set_gain)
        //.def("set_offset", &RTC::set_offset)
        .def("commit", &RTC::commit)
        .def("request_commit", &RTC::request_commit)

        .def("single_compute", &RTC::single_compute)

        // states
        // .def("get_dm_simulation_mode", [](const RTC& r) -> auto { return r.rtc_state.dm_simulation_mode; })
        // .def("set_dm_simulation_mode", [](RTC &self, bool value) { self.rtc_state.dm_simulation_mode = value; })
        // .def("get_camera_simulation_mode", [](const RTC& r) -> auto { return r.rtc_state.camera_simulation_mode; })
        // .def("set_camera_simulation_mode", [](RTC &self, bool value) { self.rtc_state.camera_simulation_mode = value; })

        // // // simulation signals
        // // .def("get_simulated_dm_cmd", [](const RTC& r) -> auto { return r.rtc_simulation_signals.simulated_dm_cmd; })
        // // .def("set_simulated_dm_cmd", [](RTC &self, bool value) { self.rtc_simulation_signals.simulated_dm_cmd = value; })

        // // .def("get_simulated_image", [](const RTC& r) -> auto { return r.rtc_simulation_signals.simulated_image; })
        // // .def("set_simulated_image", [](RTC &self, bool value) { self.rtc_simulation_signals.simulated_image = value; })

        // // .def("get_simulated_signal", [](const RTC& r) -> auto { return r.rtc_simulation_signals.simulated_image_err_signal; })
        // // .def("set_simulated_signal", [](RTC &self, bool value) { self.rtc_simulation_signals.simulated_image_err_signal = value; })

        // // reconstructors
        // .def("set_CM", &RTC::set_CM)
        // .def("set_R_TT", &RTC::set_R_TT)
        // .def("set_R_HO", &RTC::set_R_HO)
        // .def("set_I2M",&RTC::set_I2M)
        // .def("set_M2C",&RTC::set_M2C)

        // .def("get_CM", &RTC::get_CM)
        // .def("get_R_TT", &RTC::get_R_TT)
        // .def("get_R_HO", &RTC::get_R_HO)
        // .def("get_I2M",&RTC::get_I2M)
        // .def("get_M2C",&RTC::get_M2C)
        //.def("get_rtc_state", [](const RTC& self) { return self.rtc_state; })
        //.def("set_rtc_state", [](RTC& self, const rtc_state_struct& state) { self.rtc_state = state; })

        .def("__repr__", [](const RTC& rtc) {
            std::stringstream ss;
            ss << rtc;
            return ss.str();
        });

    // Specialize for rtc_state_struct

    nb::class_<rtc_state_struct>(m, "rtc_state_struct")
        .def(nb::init<>())
        .def_rw("close_loop_mode", &rtc_state_struct::close_loop_mode)
        .def_rw("dm_simulation_mode", &rtc_state_struct::dm_simulation_mode)
        .def_rw("camera_simulation_mode", &rtc_state_struct::camera_simulation_mode)
        .def_rw("signal_simulation_mode", &rtc_state_struct::signal_simulation_mode);

    nb::class_<simulated_signals_struct>(m, "simulated_signals_struct")
        .def(nb::init<>())
        .def_rw("simulated_image", &simulated_signals_struct::simulated_image)
        .def_rw("simulated_dm_cmd", &simulated_signals_struct::simulated_dm_cmd)
        .def_rw("simulated_signal", &simulated_signals_struct::simulated_image_err_signal);

    nb::class_<camera_settings_struct>(m, "camera_settings_struct")
        .def(nb::init<>())
        .def_rw("det_dit", &camera_settings_struct::det_dit)
        .def_rw("det_fps", &camera_settings_struct::det_fps)
        .def_rw("det_gain", &camera_settings_struct::det_gain)
        .def_rw("det_crop_enabled", &camera_settings_struct::det_crop_enabled)
        .def_rw("det_tag_enabled", &camera_settings_struct::det_tag_enabled)
        .def_rw("det_cropping_rows", &camera_settings_struct::det_cropping_rows)
        .def_rw("det_cropping_cols", &camera_settings_struct::det_cropping_cols)
        .def_rw("image_height", &camera_settings_struct::image_height)
        .def_rw("image_width", &camera_settings_struct::image_width)
        .def_rw("full_image_length", &camera_settings_struct::full_image_length)
        .def_rw("dark", &camera_settings_struct::dark)
        .def_rw("bad_pixels", &camera_settings_struct::bad_pixels);

    nb::class_<phase_reconstuctor_struct>(m, "phase_reconstuctor_struct")
        .def(nb::init<>())
        .def("commit_all", &phase_reconstuctor_struct::commit_all)
        .def_rw("IM", &phase_reconstuctor_struct::IM)
        .def_rw("CM", &phase_reconstuctor_struct::CM)
        .def_rw("R_TT", &phase_reconstuctor_struct::R_TT)
        .def_rw("R_HO", &phase_reconstuctor_struct::R_HO)
        .def_rw("I2M", &phase_reconstuctor_struct::I2M)
        .def_rw("I2M_TT", &phase_reconstuctor_struct::I2M_TT)
        .def_rw("I2M_HO", &phase_reconstuctor_struct::I2M_HO)
        .def_rw("M2C", &phase_reconstuctor_struct::M2C)
        .def_rw("M2C_TT", &phase_reconstuctor_struct::M2C_TT)
        .def_rw("M2C_HO", &phase_reconstuctor_struct::M2C_HO)
        .def_rw("bias", &phase_reconstuctor_struct::bias)
        .def_rw("I0", &phase_reconstuctor_struct::I0)
        .def_rw("N0", &phase_reconstuctor_struct::N0)
        .def_rw("flux_norm", &phase_reconstuctor_struct::flux_norm);

    nb::class_<pupil_regions_struct>(m, "pupil_regions_struct")
        .def(nb::init<>())
        .def("commit_all", &pupil_regions_struct::commit_all)

        .def_rw("pupil_pixels", &pupil_regions_struct::pupil_pixels)
        .def_rw("secondary_pixels", &pupil_regions_struct::secondary_pixels)
        .def_rw("outside_pixels", &pupil_regions_struct::outside_pixels)
        .def_rw("local_region_pixels", &pupil_regions_struct::local_region_pixels);

    
    nb::class_<PIDController>(m, "PIDController")
        .def(nb::init<>(), "Default constructor")
        //.def(nb::init<const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&>(), // recentchange
        .def(nb::init<const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&>(),
        "Parameterized constructor",
             nb::arg("kp"), nb::arg("ki"), nb::arg("kd"), nb::arg("lower_limit"), nb::arg("upper_limit"),nb::arg("setpoint"))
        // .def(nb::init<const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&>(),
        //      "Parameterized constructor",
        //      nb::arg("kp"), nb::arg("ki"), nb::arg("kd"), nb::arg("integral_limit"))

        // .def("process", &PIDController::process,
        //      "Compute PID",
        //      nb::arg("setpoint"), nb::arg("measured"))

        .def("process", &PIDController::process,
             "Compute PID",
             nb::arg("measured"))// recentchange
        .def("reset", &PIDController::reset, "Reset the controller")
        .def_rw("kp", &PIDController::kp, "Proportional gain")
        .def_rw("ki", &PIDController::ki, "Integral gain")
        .def_rw("kd", &PIDController::kd, "Derivative gain")
        .def_rw("upper_limit", &PIDController::upper_limit, "upper_limit")
        .def_rw("lower_limit", &PIDController::lower_limit, "lower_limit")
        .def_rw("setpoint", &PIDController::setpoint, "setpoint") // recentchange
        .def_rw("integrals", &PIDController::integrals, "Integral terms")
        .def_rw("prev_errors", &PIDController::prev_errors, "Previous errors")
        .def_rw("output", &PIDController::output);


    nb::class_<LeakyIntegrator>(m, "LeakyIntegrator")
        .def(nb::init<const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&>())
        .def("process", &LeakyIntegrator::process)
        .def("reset", &LeakyIntegrator::reset)
        .def_rw("rho", &LeakyIntegrator::rho)
        .def_rw("kp", &LeakyIntegrator::kp)
        .def_rw("output", &LeakyIntegrator::output)
        .def_rw("lower_limit", &LeakyIntegrator::lower_limit)
        .def_rw("upper_limit", &LeakyIntegrator::upper_limit);


    nb::class_<AsyncRunner>(m, "AsyncRunner")
        .def(nb::init<RTC&, std::chrono::microseconds>(), nb::arg("rtc"), nb::arg("period") = std::chrono::microseconds(1000), "Constructs an AsyncRunner object.")
        .def("start", &AsyncRunner::start)
        .def("stop", &AsyncRunner::stop)
        .def("pause", &AsyncRunner::pause)
        .def("resume", &AsyncRunner::resume)
        .def("state", &AsyncRunner::state)
        .def("flush", &AsyncRunner::flush);

    
    nb::class_<telem_entry>(m, "TelemEntry")
        .def_ro("image_in_pupil", &telem_entry::image_in_pupil)
        .def_ro("image_err_signal", &telem_entry::image_err_signal) 
        .def_ro("mode_err", &telem_entry::mode_err) 
        .def_ro("dm_cmd_err", &telem_entry::dm_cmd_err) 

        .def_ro("e_TT", &telem_entry::e_TT) 
        .def_ro("e_HO", &telem_entry::e_HO)
        .def_ro("u_TT", &telem_entry::u_TT)
        .def_ro("u_HO", &telem_entry::u_HO)
        .def_ro("cmd_TT", &telem_entry::cmd_TT)
        .def_ro("cmd_HO", &telem_entry::cmd_HO)
        .def_ro("dm_disturb", &telem_entry::dm_disturb)
        .def_ro("t0", &telem_entry::t0)
        .def_ro("t1", &telem_entry::t1); 

    bind_telemetry(m);
}
