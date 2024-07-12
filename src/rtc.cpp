#include <push_record.hpp>

#include "span_cast.hpp"
#include "span_format.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/vector.h>
// can we pass this to BMC DM ? 
#include <nanobind/ndarray.h>

#include <BMCApi.h>
#include "FliSdk.h"


#include <span>
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include <atomic>
#include <span>
#include <string_view>
// things i added 
#include <vector>
#include <numeric> // For std::accumulate

#include <cstdint> // For uint16_t
#include <cassert> // For assert()
#include <fstream>
#include <vector>
#include <cstdlib>  // for calloc
#include <string>

/* TO DO ::

julien 
- best way to hold / cycle dm commands? updatable? ring buffer?
- telemetry (image (<span>), signal (image error <span>), error_DM space (<array>), dm command (<array>) ) 

*/
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



/**
 * @brief CircularBuffer t
 *
 * Option to hold dm cmds in circular buffer
 *
 

class CircularBuffer {
public:
    CircularBuffer(size_t array_size)
        : array_size_(array_size), buffer_(3, std::vector<double>(array_size)), head_(0), count_(0) {}

    void addArray(const std::vector<double>& new_array) {
        assert(new_array.size() == array_size_ && "Array size must match buffer element size");
        buffer_[head_] = new_array;
        head_ = (head_ + 1) % 3;
        if (count_ < 3) {
            ++count_;
        }
    }

    std::vector<double> getArray(size_t index) const {
        assert(index < count_ && "Index out of range");
        return buffer_[(head_ + 3 - count_ + index) % 3];
    }

    size_t size() const {
        return count_;
    }

private:
    size_t array_size_;
    std::array<std::vector<double>, 3> buffer_;
    size_t head_;
    size_t count_;
};

*/

//std::span<const float> v
void matrix_vector_multiply(std::vector<float> v, updatable<std::span<float>>& r, std::vector<double>& result) {
    size_t N = v.size();
    size_t M = r.current().size() / N;

    // Ensure that the input vector and matrix dimensions are compatible
    assert(v.size() == N);
    assert(r.current().size() == N * M);
    
    // Initialize result vector with zeros
    result.resize(M);
    std::fill(result.begin(), result.end(), 0.0f);
    
    // Perform multiplication without explicit reshaping
    auto R = r.current();
    for (size_t j = 0; j < M; ++j) {
        for (size_t i = 0; i < N; ++i) {
            result[j] += R[i * M + j] * v[i];
        }
    }
}

// for checking DM 
bool check_dm_vector(const std::vector<double>& cmd) {
    for (double value : cmd) {
        if (value < 0.0f || value > 1.0f) {
            return true;
        }
    }
    return false;
}


// gets value at indicies 
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
//two overloads or templates with const and non-const versions. 
// template<typename T>
// std::vector<T> getValuesAtIndices(std::span<T> data_span, std::span<const int> indices_span) {
//     std::vector<T> values;
//     values.reserve(indices_span.size());

//     for (size_t index : indices_span) {
//         assert(index < data_span.size() && "Index out of bounds!"); // Ensure index is within bounds
//         values.push_back(data_span[index]);
//     }

//     return values;
// }


double* readCSV(const std::string& filePath) {
    static const int ARRAY_SIZE = 140;
    static double values[ARRAY_SIZE]; // Static array to return pointer to

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




/**
 * @brief A dummy Real-Time Controller.
 */
struct RTC {

    DM hdm = {};
    FliSdk* fli = {};

    // -----------  dimensions of image
    uint16_t image_width;// i.e. cols in image
    uint16_t image_height;// i.e. rows in image 
    uint32_t full_image_length; // = image_width * image_height - must be updated 
    // everytime these get changed! 
    // this should only be in init of RTC and using method update_camera_settings()

    // ----------- DM shapes 
    // serial number to open DM 
    const size_t dm_size = 140 ;
    const char * dm_serial_number = "17DW019#053";
    // init DM map look up table
    
    std::vector<uint32_t> map_lut; // (BMC multi-3.5 DM)

    // paths to DM shape csv files 
    const char * flat_dm_file = "/home/baldr/Documents/baldr/DMShapes/flat_dm.csv";
    const char * checkers_file = "/home/baldr/Documents/baldr/DMShapes/waffle.csv";
    const char * fourTorres_file = "/home/baldr/Documents/baldr/DMShapes/four_torres.csv";
    
    // BMC calibrated DM flat position  
    double* flat_dm_array = readCSV(flat_dm_file);
    // applies a waffle pattern on DM at frequency corresponding to interactuator spacing 
    double* checkers_dm_array = readCSV(checkers_file); 
    // pokes near each corner of the DM 
    double* fourTorres_dm_array = readCSV(fourTorres_file); 

    // updatable DM command 
    updatable<nb::ndarray<double>> dm_cmd ; 
    //or use ring buffer 
    //CircularBuffer dm_cmd_buffer(dm_size);

    // TO DO 
    // updatable basis used in reconstructor (iteract with my python BALDR module)

    // ----------- camera settings 
    //see update_camera method in RTC class to update camera settings 
    // follow proceedure set->commit->update 
    updatable<double> det_dit; // detector integration time (s)
    updatable<double> det_fps; // frames per second (Hz)
    updatable<std::string> det_gain; // "low" or "medium" or "high"
    updatable<bool> det_crop_enabled; //true/false
    updatable<bool> det_tag_enabled; //true/false
    updatable<std::string> det_cropping_rows; //"r1-r2"
    updatable<std::string> det_cropping_cols; //"c1-c2"

    // ----------- updatable variables relating to phase control 
    updatable<std::span<const float>> slope_offsets; /**< container for slope offsets. */
    updatable<float> gain; /**< value for gain. */
    updatable<float> offset; /**< value for offset. */
    updatable<std::span<float>> reconstructor; /**< reconstructor */
    updatable<std::span<uint16_t>> bias; /**< bias. */
    updatable<std::span<float>> I0; /**< reference intensity with FPM in. */
    updatable<float> flux_norm; /**< sum of intensity across detector. */

    //  ----------- updatable variables relating to control regions 
    updatable<std::span<int>> pupil_pixels; /**< pixels inside the active pupil. */
    updatable<std::span<int>> secondary_pixels; /**< pixels inside secondary obstruction   */
    updatable<std::span<int>> outside_pixels; /**< pixels outside the active pupil obstruction (but not in secondary obstruction) */

    std::atomic<bool> commit_asked = false; /**< Flag indicating if a commit is requested. */

     size_t telemetry_cnt = 0;

    /**
     * @brief Default constructor for RTC.
     */



    RTC() {
        /* open BMC DM */  
        BMCRC	rv = NO_ERR;
 
        map_lut.resize(MAX_DM_SIZE);

        rv = BMCOpen(&hdm, dm_serial_number);

        //uint32_t *map_lut; // <- we put this global in scope of struct
        uint32_t	k = 0;
        //default lookup table of DM 
        //map_lut	= (uint32_t *)malloc(sizeof(uint32_t)*MAX_DM_SIZE); // <- we put this global in scope of struct
        //load it 

        // init map lut to zeros (copying examples from BMC)
        for(k=0; k<(int)hdm.ActCount; k++) { 
            map_lut[k] = 0;
	    }
        // then we load the default map
        rv = BMCLoadMap(&hdm, NULL, map_lut.data()); 

        cout << "MAX_DM_SIZE" << MAX_DM_SIZE << endl;

        if (rv != NO_ERR) {
            std::cerr << "Error " << rv << " opening the driver type " << hdm.Driver_Type << ": ";
            std::cerr << BMCErrorString(rv) << std::endl << std::endl;
        }
        cout << "&hdm  "   << &hdm << endl;
        // try poke a single actuator 
        //rv = BMCSetSingle(&hdm, 65, 0.2);

        // // try apply an array 
        // double *test_array1;
	    // double *test_array2;

        // test_array1	= (double *)calloc(hdm.ActCount, sizeof(double));
        // test_array2 = (double *)calloc(hdm.ActCount, sizeof(double));
        
        // for(k=0; k<(int)hdm.ActCount; k++) {
        //     test_array1[k] = 0.5;
        //     test_array2[k] = 0;
        // }
        
        
        //BMCSetArray(&hdm, test_array1, map_lut);
        BMCSetArray(&hdm, flat_dm_array, map_lut.data());
        //cout << "flat_dm_array[140]" << flat_dm_array[0][140] << endl;
        
        // Print the array to verify
        for (int i = 0; i < 140; ++i) {
            std::cout << flat_dm_array[i] << " ";
        }
        std::cout << std::endl;

        /*test_array = (double *)calloc(hdm->ActCount, sizeof(double));
		for(k=0; k<*hdm->ActCount; k++) {
			test_array[k] = 0.5;

			rv = BMCSetArray(&hdm, test_array, map_lut);
			if(rv) {
				printf("\nError %d poking actuator %d.\n",rv,k);
				//err_count++;
			} else {
				printf("\rPoked actuator %d.", k);
				fflush(stdout);
			}

			test_array[k] = 0;

			//if(continuous)
			//	Sleep_us(delay);
			//else {
			//	ch = getch();
			//	if('X' == toupper(ch))
			//		break;
			//}
		}
		printf("\n");
        */
        

        /* open camera */
        this->fli = new FliSdk();
        
        int nbImages = 0;
        /* when i uncomment this ^ line and compile, when importing rtc I get
        ImportError: /home/baldr/miniconda3/lib/python3.12/site-packages/rtc/_rtc.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZN6FliSdkC1Ev
        */

        std::string cameraName;

        //detection of all the grabbers connected to the computer
        std::cout << "Detection of grabbers..." << std::endl;
        vector<string> listOfGrabbers = fli->detectGrabbers();
           
        if(listOfGrabbers.size() == 0)
        {
            cout << "No grabber detected, exit." << endl;
            return;
        }

        cout << "Done." << endl;
        cout << "List of detected grabber(s):" << endl;
        for(auto g : listOfGrabbers)
        {
            cout << "- " << g << endl;
        }
        
        //detection of the cameras connected to the grabbers
        cout << "Detection of cameras..." << endl;
        vector<string> listOfCameras = fli->detectCameras();

        if(listOfCameras.size() == 0)
        {
            cout << "No camera detected, exit." << endl;
            return;
        }

        cout << "Done." << endl;
        cout << "List of detected camera(s):" << endl;
        int i = 0;
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

        //Seting up things
        //double fps = 0;
		//fli->serialCamera()->getFps(fps);
        //cout << "current Fps read: " << fps << endl;
        //cout << "input desired fps value" << endl;
        //cin >> fps;
		//fli->serialCamera()->setFps(fps);
		//fli->serialCamera()->getFps(fps);
		//cout << "new Fps read: " << fps << endl;

        // init image_width and height !! important for shapping raw image!! 
        fli->getCurrentImageDimension(image_width, image_height);
        cout << "width  =  " << image_width << endl;
        cout << "height  =  " << image_height << endl;

        full_image_length = static_cast<uint32_t>(image_width) * static_cast<uint32_t>(image_height);

        cout << "full_image_length  =  " << full_image_length << endl;
        
        //image_width = width;
        //image_height = height; 
        
        //cin >> height;      
        //fli->setImageDimension(width, height);
        //fli->getCurrentImageDimension(width, height);
        //cout << "new width  =  " << width << endl;
        //cout << "new height  =  " << height << endl;   

        // ====== ME FUCKING AROUND 
        //get the raw image 
        //uint16_t* image16b = (uint16_t*)fli->getRawImage();

        /*cout << "asdas" << (uintptr_t)image16b << endl;
        for (size_t i = 0; i < 100; ++i) {
            std::cout << static_cast<int>(image16b[i]) << " ";
        }
        //cout << static_cast<int>(image16b).size() << endl;
        std::cout << std::endl;
        */

        //fli->stop();

        //delete fli;
        
        // Example input size
        //size_t rows = 512;
        //size_t cols = 640;
        
        // Assuming image16b is already populated with data
        //uint16_t* image16b = /* pointer to image data */;
  
        // Ensure image16b has enough data for the specified matrix size
        //assert(rows * cols <= 326529 /* size of image16b */);

        // Cast uint16_t* to a matrix of floats
        //std::vector<std::vector<float>> floatMatrix = castToFloatMatrix(image16b, rows, cols);

        // Print the resulting matrix
        //printMatrix(floatMatrix);
 
    }

    void update_camera_settings( ){

        //print to check stuff 
        double fps = 0;
		fli->serialCamera()->getFps(fps);
        cout << "fps prior = " << fps << endl;

        fli->stop();

        
        // crop first
        fli->serialCamera()->sendCommand("set cropping off"); //FliCamera_sendCommand("set cropping off");
        if (det_crop_enabled.current()) {
            //set cropping and enable
            fli->serialCamera()->sendCommand("set cropping rows "+ det_cropping_rows.current());
            fli->serialCamera()->sendCommand("set cropping columns "+ det_cropping_cols.current());
            fli->serialCamera()->sendCommand("set cropping on");

        }
        

        
        if (det_tag_enabled.current()) {
            // makes first pixels correspond to frame number and other info 
            // 
            //TO DO: should make corresponding mask for this to be added to 
            //pixel_filter if this is turned on to ensure frame count etc
            //does not get interpretted as intensities. 
            //

            fli->serialCamera()->sendCommand("set imagetags on");  
        } else{
            fli->serialCamera()->sendCommand("set imagetags off");  
        }
        

        fli->serialCamera()->sendCommand("set cropping rows "+ det_cropping_rows.current());
        fli->serialCamera()->sendCommand("set cropping cols "+ det_cropping_cols.current());

        //set fps
        fli->serialCamera()->setFps(det_fps.current());

        //set int
        //fli->serialCamera()->sendCommand("set tint " + std::to_string(det_dit))

        fli->serialCamera()->getFps(fps);
        cout << "fps despues = " << fps << endl;

        fli->update();

        //uint16_t width, height;
        fli->getCurrentImageDimension(image_width, image_height);
        cout << "image width  =  " << image_width << endl;
        cout << "image height  =  " << image_height << endl;
        full_image_length = static_cast<uint32_t>(image_width) * static_cast<uint32_t>(image_height);

        cout << "image height  =  " << full_image_length << endl;

        fli->start();

    }

    // void dm_test(){
    //     // double *test_array1;
	//     // double *test_array2;
    //     int k;

    //     // test_array1	= (double *)calloc(hdm.ActCount, sizeof(double));
    //     // test_array2 = (double *)calloc(hdm.ActCount, sizeof(double));
        
    //     //std::vector<double> test_vec(140, 0); // can we pass vectors to BMCSetArray?
    //     // ans is no..
    //     for(k=0; k<(int)hdm.ActCount; k++) { 
    //         map_lut[k] = 0;
	//     }


    //     //double** flat_dm_array = readCSV(flat_dm_file, rows, cols);
    //     BMCSetArray(&hdm, flat_dm_array, map_lut.data());

    //     // test if we can pass vectors? 
    //     //BMCSetArray(&hdm, test_vec, map_lut); // NO! we cant 
    // }

    void poke_dm_actuator(int act_number, double value){
        // act_number between 0-140
        // value between 0-1. 
        BMCSetSingle(&hdm, act_number, value);
    }

    void apply_dm_shape( nb::ndarray<double> array){

        /*
        int k;
        for(k=0; k<(int)hdm.ActCount; k++) { 
            map_lut[k] = 0;
	    }

        BMCLoadMap(&hdm, NULL, map_lut);
        */
        //array_ptr = (double *)calloc(hdm.ActCount, sizeof(double));
        double *array_ptr = array.data();
        BMCSetArray(&hdm, array_ptr, map_lut.data());

    }

    void flatten_dm(){
        cout << flat_dm_array[5] << endl;
        BMCSetArray(&hdm, flat_dm_array, map_lut.data());
    }


    // void normalizeImage(std::span<const uint16_t> img, std::span<const uint16_t> bias, float norm, std::span<float> normalizedImage) {
    //     if (img.size() != bias.size()) {
    //        throw std::invalid_argument("Image and bias must be the same size");
    //     }
    
    //     if (normalizedImage.size() != img.size()) {
    //         throw std::invalid_argument("Normalized image must be the same size as the input image");
    //     }

    //     for (size_t i = 0; i < img.size(); ++i) {
    //       normalizedImage[i] = (static_cast<float>(img[i]) - static_cast<float>(bias[i])) / norm;
    //     }
    // }

    //give a image and a bias to normalize a signal


    //filters an image from pupil pixels 





    // std::span<float> test_normalization( ){

    //     size_t rows = 512; // Number of rows in the image
    //     size_t cols = 640; // Number of columns in the image

    //     //std::vector<uint16_t> img = {100, 150, 200};
    //     uint16_t* raw_image = (uint16_t*)fli->getRawImage(); // Retrieve raw image data
    //     std::span<uint16_t> img(raw_image, rows * cols); // Create a span from the raw image data

    //     std::span<uint16_t> biasSpan = bias.current(); //{50, 50, 50};
    //     //float norm = 50.0f;
    //     float norm = flux_norm.current();
    //     std::vector<float> nI(img.size());

    //     std::span<uint16_t> imgSpan(img);
    //     //std::span<uint16_t> biasSpan(bias);
    //     std::span<float> normalizedImageSpan(nI);

    //     normalizeImage(imgSpan, biasSpan, norm, normalizedImageSpan);
    //     return(normalizedImageSpan);
    // }


    // std::span<float> test_pixel_filter() {

    //     size_t rows = 512; // Number of rows in the image
    //     size_t cols = 640; // Number of columns in the image

    //     // Assuming pupil_pixels.current() and I0.current() return std::span<int> and std::span<float> respectively
    //     std::span<int> indices_span = pupil_pixels.current();

    //     // get image 
    //     uint16_t* raw_image = (uint16_t*)fli->getRawImage(); // Retrieve raw image data
    //     std::span<uint16_t> image_span(raw_image, rows * cols); // Create a span from the raw image data

    //     //get current reference intensity
    //     std::span<float> I0_span = I0.current();
    //     //get current bias
    //     std::span<uint16_t> bias_span = bias.current();
    //     // Get values at specified indices
    //     std::vector<uint16_t> bias_output = getValuesAtIndices<uint16_t>(bias_span, indices_span);
    //     std::vector<float> I0_output = getValuesAtIndices<float>(I0_span, indices_span);
    //     std::vector<uint16_t> im_output = getValuesAtIndices<uint16_t>(image_span, indices_span) ;

    //     std::span<uint16_t> bias_output_span(bias_output);
    //     std::span<float> I0_output_span(I0_output);
    //     std::span<uint16_t> im_output_span(im_output);

    //     //testing to do normalization 
    //     float norm = flux_norm.current();
    //     std::vector<float> nI(im_output_span.size()); //pre-allocate vector for output of normalizeImage
    //     //std::span<uint16_t> imgSpan(img);
    //     std::span<float> normalizedImageSpan(nI); // this is placeholder for output (since function returns span..)

    //     normalizeImage(im_output_span, bias_output_span, norm, normalizedImageSpan);

    //     // dont forget to change function return type if changing output
    //     return normalizedImageSpan; //im_output_span ; //bias_output_span; //I0_output_span;
    // }


    //std::span<const uint16_t>
    //std::vector<double>
    std::vector<double> test(){

        //size_t rows = 512; // Number of rows in the image
        //size_t cols = 640; // Number of columns in the image

        //size_t layers = 140; // Number of layers in the reconstructor matrix
        
        std::vector<double> delta_cmd; // to hold DM command offset from flat reference 
        std::vector<double> cmd; // to hold DM command 

        uint16_t* raw_image = (uint16_t*)fli->getRawImage(); // Retrieve raw image data

        std::span<const uint16_t> image_span(raw_image, full_image_length);//rows * cols); // Create a span from the raw image data
        //remember full_image_length is global in RTC struct and should be udpated every
        // time camera settings gets updated - always check this ! 

        
        //std::span<int> indices_span = pupil_pixels.current() ;

        
        // Get values of at specified indices
        std::vector<float> im = getValuesAtIndices<float>(image_span, pupil_pixels.current() ) ; // image
        //std::vector<float> b = getValuesAtIndices<float>(bias.current(),  pupil_pixels.current()); // bias
        std::vector<float> I_ref = getValuesAtIndices<float>(I0.current(),  pupil_pixels.current() ); // set point intensity
        
        assert(im.size() == I_ref.size());

        //basic checks 
        //assert(im.size() == b.size()); // Ensure sizes match between im and b
        assert(im.size() == I_ref.size()); // Ensure sizes match between im and I_ref
        
        // subtract bias, normalize and then subtract reference intensity 
        // to generate error signal
        size_t size = im.size();
        std::vector<float> signal(size);

        float flux_norm_value = flux_norm.current();
        // debugging
        //cout << "flux_norm = " << flux_norm_value << endl;

        for (size_t i = 0; i < size; ++i) {
          //signal[i] = (static_cast<float>(im[i]) - static_cast<float>(b[i])) / flux_norm.current() - I_ref[i];
          signal[i] = static_cast<float>(im[i]) / flux_norm_value - I_ref[i];
          // debugging
          //cout << "signal i = " << signal[i] << endl;
        }

        // MVM to turn our error vector to a DM command via the reconstructor 
               
        matrix_vector_multiply(signal, reconstructor, delta_cmd); // delta_cmd = R . I (offset from reference flat DM)

        /* THIS DOESNT WORK? WHY?
        cmd = delta_cmd;  // copying is STUPID - just trying to geto work!
        //Perform element-wise addition
        for (size_t i = 0; i < size; ++i) {
            //cout << flat_dm_array[i] << endl ;
            cmd[i] = 0.5 + delta_cmd[i]; // just roughly offset to center
        }
        */


        //if (check_dm_vector(cmd)){
        //    throw std::invalid_argument( "Dangerous DM command less than ZERO or greater than ONE" );
        //}
        
        
        double *cmd_ptr = cmd.data();

        //BMCSetArray(&hdm, cmd_ptr, map_lut.data());

        // update dm_cmd_buffer 
        //cout << cmd[0]  << endl;
        //return cmd;
        if (telemetry_cnt > 0){
            telem_entry entry;

            entry.image_raw = std::move(image_span); // im);
            entry.image_proc = std::move(signal);
            entry.reco_dm_err = std::move(cmd); // reconstructed DM command
            //entry.dm_command = std::move(); // final command sent

            append_telemetry(std::move(entry));

            --telemetry_cnt;
        }
        return delta_cmd;
    }

    void latency_test(int i, int switch_on){
        // just switch between actuator pokes based on modulus of j and switch on
        static int j;
        j = 0;
        
    }


    /**
     * @brief Performs computation using the current RTC values.
     * @param os The output stream to log some informations.
     */
    void compute(std::ostream& os){
        os << "computing with " << (*this) << '\n';

        // Do some computation here...
        test();
        // When computation is done, check if a commit is requested.
        if (commit_asked) {
            commit();
            commit_asked = false;
        }
    }

    void enable_telemetry(size_t iteration_nb) {
        telemetry_cnt = iteration_nb;
    }

    /**
     * @brief Sets the slope offsets.
     * @param new_offsets The new slope offsets to set.
     */


    std::span<uint16_t> get_last_frame() { 
        
        return {(uint16_t*)fli->getRawImage(), full_image_length}; // full dimension for C-RED3 = 512*640 
     
    }


    nb::ndarray<double> get_current_dm_cmd(){
        return dm_cmd.current();
    }

    nb::ndarray<double> get_next_dm_cmd(){
        return dm_cmd.next();
    }
    
    std::span<float> get_reconstructor(){

        std::span<float> reconstructor_span = reconstructor.current();
        return reconstructor_span;
    }


    uint16_t get_img_width(){
        return image_width;
    }

    uint16_t get_img_height(){
        return image_height;
    }

    float get_current_flux_norm(){
        float fn = flux_norm.current();
        return( fn );
    }

    std::span<float> get_current_I0(){
        
        return( I0.current() );
    }

    //can delete 
    void set_slope_offsets(std::span<const float> new_offsets) {
        slope_offsets.update(new_offsets);
    }
    
    //RECONSTRUCTOR MATRIX 
    void set_ctrl_matrix(std::span<float> mat) {
        reconstructor.update(mat);
    }
    // defined region in pupil (where we do phase control)
    void set_pupil_pixels(std::span<int> array) {
        pupil_pixels.update(array);
    }
    // defined region in secondary obstruction 
    void set_secondary_pixels(std::span<int> array) {
        secondary_pixels.update(array);
    }

    // defined region in  outside pupil (not including secondary obstruction)
    void set_outside_pixels(std::span<int> array) {
        outside_pixels.update(array);
    }

    // SIGNAL VARIABLES 
    void set_bias(std::span<uint16_t> array) {
        bias.update(array);
    }

    void set_I0(std::span<float> array) {
        I0.update(array);
    }

    void set_fluxNorm(float value) {
        flux_norm.update(value);
    }

    //set the next DM cmd 
    void set_dm_cmd(nb::ndarray<double> array){
        dm_cmd.update(array);
    }

    // CAMERA SETTINGS 
    void set_det_fps(double value){
        det_fps.update(value);
    }

    void set_det_dit(double value){
        det_dit.update(value);
    }

    void set_det_gain(std::string value){
        det_gain.update(value);
    }
    
    void set_det_tag_enabled(bool value){
        det_tag_enabled.update(value);
    }

    void set_det_crop_enabled(bool value){
        det_crop_enabled.update(value);
    }

    void set_det_cropping_rows(std::string value){
        //should be "r1-r2" where r1, r2 are ints , r1 multiple of 4, r2 multiple 4-1
        det_cropping_rows.update(value);
    }

    void set_det_cropping_cols(std::string value){
        //should be "c1-c2" where c1, c2 are ints , c1 multiple of 32, c2 multiple 32-1
        det_cropping_cols.update(value);
    }
    

    /**
     * @brief Sets the gain.
     * @param new_gain The new gain to set.
     */
    void set_Ki_gain(float new_gain) {
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

    void commit_dm(){
        dm_cmd.commit() ;// put next dm cmd as current
    }

    void commit_camera_settings(){
        // DETECTOR 
        det_dit.commit();
        det_fps.commit();
        det_gain.commit();
        det_cropping_rows.commit();
        det_cropping_cols.commit();
        det_tag_enabled.commit();
        det_crop_enabled.commit();

        //det_crop_enabled.commit(); 
        //det_crop_r1.commit();
        //det_crop_r2.commit();
        //det_crop_c1.commit();
        //det_crop_c2.commit();
    }


    void commit() {
        //for now just commit everything - but we probably want to commit subgroups later 
        
        //slope_offsets.commit();
        //gain.commit();
        //offset.commit();
 
 
        // REGION FILTERS 
        pupil_pixels.commit();

        // RECONSTRUCTOR 
        reconstructor.commit();
        bias.commit();
        I0.commit();
        flux_norm.commit();

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

void bind_telemetry(nb::module_& m);



NB_MODULE(_rtc, m) {
    using namespace nb::literals;

    nb::class_<RTC>(m, "RTC")
        .def(nb::init<>())
        .def("compute", &RTC::compute)
        //.def_rw("value", &RTC::value)

        //reconstuctor
        .def("set_slope_offsets", &RTC::set_slope_offsets) //delete
        .def("set_ctrl_matrix", &RTC::set_ctrl_matrix)
        .def("set_I0", &RTC::set_I0)
        .def("set_bias", &RTC::set_bias)
        .def("set_fluxNorm", &RTC::set_fluxNorm)
        .def("set_Ki_gain", &RTC::set_Ki_gain)
        .def("set_offset", &RTC::set_offset) //delete
        .def("get_last_frame", &RTC::get_last_frame)

        .def("get_current_flux_norm", &RTC::get_current_flux_norm)
        .def("get_current_I0", &RTC::get_current_I0)
        // pupil control regions 
        .def("set_pupil_pixels", &RTC::set_pupil_pixels)
        .def("set_secondary_pixels", &RTC::set_secondary_pixels)
        .def("set_outside_pixels", &RTC::set_outside_pixels)

        // detector 
        .def("get_img_width",&RTC::get_img_width)
        .def("get_img_height",&RTC::get_img_height)
        
        .def("set_det_fps", &RTC::set_det_fps)
        .def("set_det_dit", &RTC::set_det_dit)
        .def("set_det_gain", &RTC::set_det_gain)
        .def("set_det_crop_enabled", &RTC::set_det_crop_enabled)
        .def("set_det_tag_enabled", &RTC::set_det_tag_enabled)
        .def("set_det_cropping_rows", &RTC::set_det_cropping_rows)
        .def("set_det_cropping_cols", &RTC::set_det_cropping_cols)

        .def("update_camera_settings", &RTC::update_camera_settings)
        
        //DM
        .def("set_dm_cmd", &RTC::set_dm_cmd) // sets the next dm cmd in updatable dm_cmd 
        .def("get_current_dm_cmd", &RTC::get_current_dm_cmd) // gets the current dm cmd in updatable dm_cmd 
        .def("get_next_dm_cmd", &RTC::get_next_dm_cmd) // gets the next dm cmd in updatable dm_cmd 

        .def("poke_dm_actuator", &RTC::poke_dm_actuator) // applies a input value to a single input actuator on the DM
        .def("apply_dm_shape", &RTC::apply_dm_shape) // applies a cmd (140 array) to set the DM shape 
        .def("flatten_dm", &RTC::flatten_dm) //flattens DM 
        //telemetry
        .def("enable_telemetry", &RTC::enable_telemetry)

        //updatable
        .def("commit", &RTC::commit)
        .def("commit_dm", &RTC::commit) 
        .def("commit_camera_settings", &RTC::commit_camera_settings)
        .def("request_commit", &RTC::request_commit)

        //
        
        .def("get_reconstructor", &RTC::get_reconstructor)
        //.def("normalizedImage", &RTC::normalizedImage)
        //.def("dm_test", &RTC::dm_test)
        .def("test", &RTC::test)
        //.def("test_normalization",&RTC::test_normalization)
        //.def("test_pixel_filter",&RTC::test_pixel_filter)
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

    nb::class_<telem_entry>(m, "TelemEntry")
        .def_ro("image_raw", &telem_entry::image_raw)
        .def_ro("image_proc", &telem_entry::image_proc) 
        .def_ro("reco_dm_err", &telem_entry::reco_dm_err) 
        .def_ro("dm_command", &telem_entry::dm_command); 

    bind_telemetry(m);

}

