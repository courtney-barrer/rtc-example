#include "span_cast.hpp"
#include "span_format.hpp"

#include <cstdint>
#include <updatable.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/vector.h>

#include <span>
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include <atomic>
#include <span>
#include <string_view>
#include <vector>
#include <fstream>

#include <BMCApi.h>
#include "FliSdk.h"

namespace nb = nanobind;
using namespace std;

// ssh rtc@150.203.88.114

/*
put PID and leaky integrator in reco struct.?
include method - modal, zonal etc
*/

/// @brief A PID controller with Anti-windup
class PIDController {
public:
    PIDController()
        : kp(0.0), ki(0.0), kd(0.0), lower_limit(0), upper_limit(1.0)  {}

    PIDController(const std::vector<float>& kp, const std::vector<float>& ki, const std::vector<float>& kd, const std::vector<float>& upper_limit, const std::vector<float>& lower_limit)
        : kp(kp), ki(ki), kd(kd), lower_limit(lower_limit), upper_limit(upper_limit) {}

    std::vector<float> process(const std::vector<float>& setpoint, const std::vector<float>& measured) {
        if (setpoint.size() != measured.size()) {
            throw std::invalid_argument("Input vectors must be of the same size.");
        }

        if (integrals.size() != setpoint.size()) {
            integrals.resize(setpoint.size(), 0.0f);
            prev_errors.resize(setpoint.size(), 0.0f);
        }

        std::vector<float> output(setpoint.size());
        for (size_t i = 0; i < setpoint.size(); ++i) {
            float error = setpoint[i] - measured[i];
            integrals[i] += error;

            // Anti-windup: clamp the integrative term within upper and lower limits
            integrals[i] = std::clamp(integrals[i], lower_limit[i], upper_limit[i]);

            float derivative = error - prev_errors[i];
            output[i] = kp[i] * error + ki[i] * integrals[i] + kd[i] * derivative;
            prev_errors[i] = error;
        }

        return output;
    }

    void reset() {
        std::fill(integrals.begin(), integrals.end(), 0.0f);
        std::fill(prev_errors.begin(), prev_errors.end(), 0.0f);
    }

    std::vector<float> kp;
    std::vector<float> ki;
    std::vector<float> kd;
    std::vector<float> upper_limit;
    std::vector<float> lower_limit;

    std::vector<float> integrals;
    std::vector<float> prev_errors;
};


class LeakyIntegrator {
public:
    // Constructor to initialize the leaky integrator with a given rho vector and limits for anti-windup
    LeakyIntegrator()
        : rho(0.0), lower_limit(0.0), upper_limit(0.0) {}

    LeakyIntegrator(const std::vector<double>& rho, const std::vector<double>& lower_limit, const std::vector<double>& upper_limit)
        : rho(rho), output(rho.size(), 0.0), lower_limit(lower_limit), upper_limit(upper_limit) {
        if (rho.empty()) {
            throw std::invalid_argument("Rho vector cannot be empty.");
        }
        if (lower_limit.size() != rho.size() || upper_limit.size() != rho.size()) {
            throw std::invalid_argument("Lower and upper limit vectors must match rho vector size.");
        }
    }

    // Method to process a new input and update the output
    std::vector<double> process(const std::vector<double>& input) {
        if (input.size() != rho.size()) {
            throw std::invalid_argument("Input vector size must match rho vector size.");
        }

        for (size_t i = 0; i < rho.size(); ++i) {
            output[i] = rho[i] * output[i] + input[i];
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
    std::vector<double> rho;        // 1 - time constants for the leaky integrator per dimension
    std::vector<double> output;     // Current output of the integrator per dimension
    std::vector<double> lower_limit;  // Lower output limits for anti-windup
    std::vector<double> upper_limit;  // Upper output limits for anti-windup
};


// gets value at indicies for a input vector (span)
template<typename DestType, typename T>
std::vector<DestType> getValuesAtIndices(std::span<T> data_span, std::span<const int> indices_span) {
    std::vector<DestType> values;
    values.reserve(indices_span.size());

    for (size_t index : indices_span) {
        assert(index < data_span.size() && "Index out of bounds!"); // Ensure index is within bounds
        values.push_back(static_cast<DestType>(data_span[index]));
    }

    return values;
}

// Gets value at indices for an input vector
template<typename DestType, typename T>
void getValuesAtIndices(std::vector<DestType>& values, const T & data, std::span<const int> indices) {
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

    // works for vector<int> but not vector<float>?
    updatable<std::vector<float>> IM; /**<unfiltered interaction matrix  */
    
    updatable<std::vector<float>> CM; /**< control matrix (~ M2C @ I2M.T) signal intensities to dm commands  */
    updatable<std::vector<float>> R_TT; /**< tip/tilt reconstructor */
    updatable<std::vector<float>> R_HO; /**< higher-order reconstructor */
    updatable<std::vector<float>> I2M; /**< intensity (signal) to mode matrix */
    //std::span<float> I2M_a // again there is a bug with updatable I2M as with I0...
    updatable<std::vector<float>> M2C; /**< mode to DM command matrix. */

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
        M2C.commit();
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

    void commit_all(){
        pupil_pixels.commit();
        secondary_pixels.commit();
        outside_pixels.commit();
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
    LeakyIntegrator leakyInt; // object to apply PID control to signals

    // -------- SIMULATED SIGNALS (for if in simulation mode )
    simulated_signals_struct rtc_simulation_signals;

    // variables 

    uint16_t current_frame_number; // the current frame number
    int32_t previous_frame_number; // the previous frame number 
    //(signed 32 int since at uint16 overflow the previous frame number needs to go to -1)

    std::vector<float> image_in_pupil; // image intensities filtered within active pupil  
    std::vector<float> image_setpoint; // reference image filtered within active pupil  
    std::vector<float> image_err_signal; // processed image error signal (normalized)
    std::vector<double> dm_cmd_err; // holds the DM command offset (error) from the reference flat DM surface
    std::vector<double> TT_cmd_err; // holds the DM Tip-Tilt command offset (error) from the reference flat DM surface
    std::vector<double> HO_cmd_err; // holds the DM Higher Order command offset (error) from the reference flat DM surface
    std::vector<double> mode_err; // holds mode error from matrix multiplication of I2M with processed signal 

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
        FliSdk* fli = new FliSdk();
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

    /// @brief standard way to send a command to the DM in the context of the RTC. cmds should be vectors and here we convert to a pointer before sending it.
    /// this also checks if the DM is in simulation within the RTC struct.
    /// @return a pointer to the uint16_t image
    void send_dm_cmd(std::span<double> cmd) {
        if (not rtc_state.dm_simulation_mode) {
                    // get cmd pointer
            double *cmd_ptr = cmd.data();
            BMCSetArray(&hdm, cmd_ptr, map_lut.data());
        }
    }

    void process_image(std::span<const float> im, std::span<const float> im_setpoint, std::vector<float>& signal)
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
    std::vector<uint16_t>  im2vec_test(){
        // to test things 

        // get image
        uint16_t* raw_image = poll_last_image();

        // convert to vector
        std::vector<uint16_t> image_vector(raw_image, raw_image + camera_settings.full_image_length);
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
        std::vector<uint16_t> image_vector(raw_image, raw_image + camera_settings.full_image_length);

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
        std::vector<uint16_t> image_vector(raw_image, raw_image + camera_settings.full_image_length);

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
        std::vector<uint16_t> image_vector(raw_image, raw_image + camera_settings.full_image_length);

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
     * this can be used to test compute and is nanobinded to python
     */

     // Define an enum class for the choices

    //int return_case=4
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
        std::vector<uint16_t> image_vector(raw_image, raw_image + camera_settings.full_image_length);

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

                break;
            case 2: //HO cmd_err
                cout << "HO cmd_err" << endl;
                matrix_vector_multiply( image_err_signal, reco.R_HO.current(), HO_cmd_err ) ;

            case 3: //mode amplitudes err:
                cout << "mode amplitudes err" << endl;
                matrix_vector_multiply( image_err_signal, reco.I2M.current(), mode_err ) ;
                break;
            case 4: //"full" DM cmd error:

                matrix_vector_multiply( image_err_signal, reco.CM.current(), dm_cmd_err ) ;
                break;

            // tens - apply PID controller in respective spaces 
            case 11: 
                cout << 2 << 'do' << endl;
                break;



            // hundres - reconstruct full DM command 



                break;
            default:
                cout << "no to_return cases met. Returning CM @ signal" << endl;
                matrix_vector_multiply( image_err_signal, reco.CM.current(), dm_cmd_err ) ;
                break;
        }
        return dm_cmd_err ;
    }

    /**
     * @brief Performs computation using the current RTC values.
     * @param os The output stream to log some informations.
     */
    void compute(std::ostream& os)
    {
        os << "computing with " << (*this) << '\n';


        // get image
        uint16_t* raw_image = poll_last_image();
        // get current frame number
        // current_frame_number = ads[0] 
        
        if (current_frame_number > previous_frame_number){
            // Do some computation here...

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
            
            // convert to vector
            std::vector<uint16_t> image_vector(raw_image, raw_image + camera_settings.full_image_length);

            // size of filtered signal may change while RTC is running
            size_t signal_size = regions.pupil_pixels.current().size();

            // have to define here to keep in scope of telemetry
            static std::vector<float> image_err_signal(signal_size); // <- should this be static

            //static uint16_t frame_cnt = image_vector[0]; // to check if we are on a new frame

            //std::vector<float> image_in_pupil;//<- init at top struct
            getValuesAtIndices(image_in_pupil, image_vector, regions.pupil_pixels.current()  ) ; // image

            //std::vector<float> image_setpoint;//<- init at top struct
            getValuesAtIndices(image_setpoint, reco.I0.current(),  regions.pupil_pixels.current()  ); // set point intensity

            process_image( image_in_pupil, image_setpoint, image_err_signal);

            /* TO DO: 
            define TT_cmd_err, HO_cmd_err at beginning of struct
            test addition onto flat command
            
            */ 
            //matrix_vector_multiply( image_err_signal, reco.R_TT.current(), TT_cmd_err )
            //matrix_vector_multiply( image_err_signal, reco.R_TT.current(), HO_cmd_err )

            // PID and leaky to TT_cmd_err and HO_cmd_err (in cmd space)
            //u_TT = pid.process(  TT_cmd_err   ) // use PID for tip/tilt
            //u_HO = LeakyInt.process( HO_cmd_err  ) //use leaky integrator for HO 

            //cmd =  flat_cmd + TT_cmd_err + HO_cmd_err
            //Perform element-wise addition
            // for (size_t i = 0; i < dm_size; ++i) {
            //     //cout << flat_dm_array[i] << delta_cmd[i]<< endl ;

            //     cmd[i] = flat_dm_array[i] + u_HO[i] + u_TT[i]; // just roughly offset to center
            // }

            // send DM cmd 


            
            


        }
        
        // When computation is done, check if a commit is requested.
        if (commit_asked) {
            commit();
            commit_asked = false;
        }
    }


    //RECONSTRUCTOR MATRIX
    // void set_CM(std::vector<float> mat) {
    //     reco.CM.update(mat); // control matrix (not filtered for tip/tilt or higher order modes)
    // }

    // void set_R_TT(std::vector<float> mat) {
    //     reco.R_TT.update(mat);
    // }

    // void set_R_HO(std::vector<float> mat) {
    //     reco.R_HO.update(mat);
    // }

    // void set_I2M(std::vector<float> array){
    //     reco.I2M.update(array);
    // }

    // //void set_I2M_a(std::span<float> array){
    // //    //USE I2Ma BECAUSE OF BUG IN UPDATABLE I2M (IT RANDOMLY CHANGES/ ASIGNS VALUES IN NANOBIND)
    // //    I2M_a = array;
    // //}

    // void set_M2C(std::vector<float> array){
    //     reco.M2C.update(array);
    // }



    // void set_I0(std::vector<float> array) {
    //     reco.I0.update(array);
    // }

    // void set_fluxNorm(float value) {
    //     reco.flux_norm.update(value);
    // }

    // // defined region in pupil (where we do phase control)
    // void set_pupil_pixels(std::vector<int> array) {
    //     regions.pupil_pixels.update(array);
    // }


    // // defined region in secondary obstruction
    // void set_secondary_pixels(std::vector<int> array) {
    //     regions.secondary_pixels.update(array);
    // }

    // // defined region in  outside pupil (not including secondary obstruction)
    // void set_outside_pixels(std::vector<int> array) {
    //     regions.outside_pixels.update(array);
    // }



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

        reco.IM.commit() ;
        reco.CM.commit();
        reco.R_TT.commit();
        reco.R_HO.commit();
        reco.I2M.commit();
        reco.M2C.commit();
        reco.I0.commit();
        reco.N0.commit();
        reco.flux_norm.commit();

        regions.pupil_pixels.commit();
        regions.secondary_pixels.commit();
        regions.outside_pixels.commit();

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

    register_updatable(m);

    nb::class_<RTC>(m, "RTC")
        .def(nb::init<>())
        .def_rw("rtc_state", &RTC::rtc_state)
        .def_rw("camera_settings", &RTC::camera_settings)
        .def_rw("reco", &RTC::reco)
        .def_rw("regions", &RTC::regions)
        .def_rw("rtc_simulation_signals", &RTC::rtc_simulation_signals)

        .def("apply_camera_settings", &RTC::apply_camera_settings)

        // controllers
        //.def("PID", &RTC.pid)
        .def_rw("pid", &RTC::pid, "PID Controller")
        .def_rw("LeakyInt", &RTC::leakyInt, "Leaky Integrator")

        // for testing individually 
        .def("im2vec_test", &RTC::im2vec_test)
        .def("im2filtered_im_test", &RTC::im2filtered_im_test)
        .def("poll_last_image", &RTC::poll_last_image)
        .def("im2filteredref_im_test", &RTC::im2filteredref_im_test)
        .def("process_im_test", &RTC::process_im_test)


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
        .def_rw("full_image_length", &camera_settings_struct::full_image_length);

    nb::class_<phase_reconstuctor_struct>(m, "phase_reconstuctor_struct")
        .def(nb::init<>())
        .def("commit_all", &phase_reconstuctor_struct::commit_all)
        .def_rw("IM", &phase_reconstuctor_struct::IM)
        .def_rw("CM", &phase_reconstuctor_struct::CM)
        .def_rw("R_TT", &phase_reconstuctor_struct::R_TT)
        .def_rw("R_HO", &phase_reconstuctor_struct::R_HO)
        .def_rw("I2M", &phase_reconstuctor_struct::I2M)
        .def_rw("M2C", &phase_reconstuctor_struct::M2C)
        .def_rw("bias", &phase_reconstuctor_struct::bias)
        .def_rw("I0", &phase_reconstuctor_struct::I0)
        .def_rw("N0", &phase_reconstuctor_struct::N0)
        .def_rw("flux_norm", &phase_reconstuctor_struct::flux_norm);

    nb::class_<pupil_regions_struct>(m, "pupil_regions_struct")
        .def(nb::init<>())
        .def("commit_all", &pupil_regions_struct::commit_all)

        .def_rw("pupil_pixels", &pupil_regions_struct::pupil_pixels)
        .def_rw("secondary_pixels", &pupil_regions_struct::secondary_pixels)
        .def_rw("outside_pixels", &pupil_regions_struct::outside_pixels);

    
    nb::class_<PIDController>(m, "PIDController")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&>(),
        "Parameterized constructor",
             nb::arg("kp"), nb::arg("ki"), nb::arg("kd"), nb::arg("upper_limit"),nb::arg("lower_limit"))
        // .def(nb::init<const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&>(),
        //      "Parameterized constructor",
        //      nb::arg("kp"), nb::arg("ki"), nb::arg("kd"), nb::arg("integral_limit"))
        .def("process", &PIDController::process,
             "Compute PID",
             nb::arg("setpoint"), nb::arg("measured"))
        .def("reset", &PIDController::reset, "Reset the controlleßßr")
        .def_rw("kp", &PIDController::kp, "Proportional gain")
        .def_rw("ki", &PIDController::ki, "Integral gain")
        .def_rw("kd", &PIDController::kd, "Derivative gain")
        .def_rw("upper_limit", &PIDController::upper_limit, "upper_limit")
        .def_rw("lower_limit", &PIDController::upper_limit, "lower_limit")
        .def_rw("integrals", &PIDController::integrals, "Integral terms")
        .def_rw("prev_errors", &PIDController::prev_errors, "Previous errors");


    nb::class_<LeakyIntegrator>(m, "LeakyIntegrator")
        .def(nb::init<const std::vector<double>&, const std::vector<double>&, const std::vector<double>&>())
        .def("process", &LeakyIntegrator::process)
        .def("reset", &LeakyIntegrator::reset)
        .def_rw("rho", &LeakyIntegrator::rho)
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

}
