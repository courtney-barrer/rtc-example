#include "span_cast.hpp"
#include "span_format.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/vector.h>

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


In test() up to sending cmd back to DM but for some reason variables I thought were global 
dont work inside test funtion

 function norm std::span<float> (I-B)/sum(I) I, B both raw images type std::span<uint16_t>
- consider using updatable<float> flux_norm so don't have to do sum every time 
 function S= I-I0 where I and I0 are normalized and of type std::span<float>
 deal with cropped regions in input (maybe we can just recrop here.. using this as input)
 - do this for any image taken (get frame) -> crop it here
 function to filter signal from pixels 
 matrix mult R . S , S and R are both std::span<float>
 put in compute
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



// Function for matrix multiplication - we do void so result vector's memory is managed correctly outside the function and then create a std::span from this vector
// OLD DONT USE 
void matrixMultiplication(std::span<float> matrix, std::span<const uint16_t> vector, std::vector<float>& result, size_t layers, size_t rows, size_t cols) {
    std::fill(result.begin(), result.end(), 0.0f);  // Ensure result is zeroed out before computation

    for (size_t l = 0; l < layers; ++l) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[l] += matrix[l * rows * cols + i * cols + j] * vector[i * cols + j];
            }
        }
    }
}

/*
void matrix_vector_multiply(std::span<const uint16_t> v, updatable<std::span<float>>& r, size_t N, size_t M, std::vector<float>& result) {
    // Ensure that the input vector and matrix dimensions are compatible
    assert(v.size() == M);
    assert(r.current().size() == N * M);
    
    // Initialize result vector with zeros
    result.resize(N);
    std::fill(result.begin(), result.end(), 0.0f);
    
    // Perform multiplication without explicit reshaping
    auto R = r.current();
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result[i] += R[i * M + j] * v[j];
        }
    }
}
*/

void matrix_vector_multiply(std::span<const uint16_t> v, updatable<std::span<float>>& r, std::vector<float>& result) {
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

// gets value at indicies 
template<typename T>
std::vector<T> getValuesAtIndices(const std::span<const T>& data_span, std::span<const int> indices_span) {
    std::vector<T> values;
    values.reserve(indices_span.size());

    for (size_t index : indices_span) {
        assert(index < data_span.size() && "Index out of bounds!"); // Ensure index is within bounds
        values.push_back(data_span[index]);
    }

    return values;
}
//two overloads or templates with const and non-const versions. 
template<typename T>
std::vector<T> getValuesAtIndices(const std::span<T>& data_span, std::span<const int> indices_span) {
    std::vector<T> values;
    values.reserve(indices_span.size());

    for (size_t index : indices_span) {
        assert(index < data_span.size() && "Index out of bounds!"); // Ensure index is within bounds
        values.push_back(data_span[index]);
    }

    return values;
}

double** readCSV(const std::string& file_path, size_t& rows, size_t& cols) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return nullptr;
    }

    std::string line;
    std::vector<double*> data;
    cols = 0;

    // Read the file line by line
    while (std::getline(file, line)) {
        std::stringstream line_stream(line);
        std::string cell;
        size_t column_index = 0;

        // Allocate memory for a new row using calloc
        double* row = (double*)calloc(cols, sizeof(double));

        while (std::getline(line_stream, cell, ',')) {
            double value = std::stod(cell);

            // If it's the first row, determine the number of columns
            if (data.size() == 0) {
                cols++;
                row = (double*)realloc(row, cols * sizeof(double));
            }

            row[column_index++] = value;
        }

        data.push_back(row);
        rows++;
    }

    // Allocate memory for the 2D array to return
    double** array = (double**)calloc(rows, sizeof(double*));
    for (size_t i = 0; i < rows; ++i) {
        array[i] = data[i];
    }

    file.close();
    return array;
}



/**
 * @brief A dummy Real-Time Controller.
 */
struct RTC {

    DM hdm = {};
    FliSdk* fli = {};

    float value = 10;
    updatable<std::span<const float>> slope_offsets; /**< container for slope offsets. */
    updatable<float> gain; /**< value for gain. */
    updatable<float> offset; /**< value for offset. */
    updatable<std::span<float>> reconstructor; /**< reconstructor */
    updatable<std::span<uint16_t>> bias; /**< bias. */
    updatable<std::span<float>> I0; /**< reference intensity with FPM in. */
    updatable<float> flux_norm; /**< sum of intensity across detector. */
    updatable<std::span<int>> pupil_pixels; /**< value for offset. */

    std::atomic<bool> commit_asked = false; /**< Flag indicating if a commit is requested. */

    /**
     * @brief Default constructor for RTC.
     */



    RTC() {
        /* open BMC DM */  
        BMCRC	rv = NO_ERR;
        // pre-defined DM shapes
        //const char DMshapes_path = "/home/baldr/Documents/baldr/DMShapes/"
        const char * serial_number = "17DW019#053";
        const char * flat_dm_file = "/home/baldr/Documents/baldr/DMShapes/flat_dm.csv";
        const char * checkers_file = "/home/baldr/Documents/baldr/DMShapes/dm_checker_pattern.csv";
        const char * fourTorres_file = "/home/baldr/Documents/baldr/DMShapes/four_torres.csv";
        size_t rows = 0, cols = 0;
        double** flat_dm_array = readCSV(flat_dm_file, rows, cols);
        
        if (flat_dm_array == nullptr) {
            std::cerr << "Failed to read flat DM CSV file." << std::endl;
            
        }
        cout << "no rows = " << rows << endl;
        cout << "no cols = " << cols << endl;
        /*
        // Print the data to verify
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << flat_dm_array[i][j] << " ";
            }
            std::cout << std::endl;
            //free(flat_dm_array[i]);  // Free the allocated memory for each row
        }
        */

        
        rv = BMCOpen(&hdm, serial_number);

        uint32_t *map_lut;
        uint32_t	k = 0;
	    double	*test_array;
        //default lookup table of DM 
        map_lut	= (uint32_t *)malloc(sizeof(uint32_t)*MAX_DM_SIZE);
        //load it 
        rv = BMCLoadMap(&hdm, NULL, map_lut);

        if (rv != NO_ERR) {
            std::cerr << "Error " << rv << " opening the driver type " << hdm.Driver_Type << ": ";
            std::cerr << BMCErrorString(rv) << std::endl << std::endl;
        }
        cout << "&hdm  "   << &hdm << endl;
        cout << " map_lut  " << map_lut << endl;
        // try poke a single actuator 
        //rv = BMCSetSingle(&hdm, 65, 0.2);

        // try apply an array 
        /*
        double *test_array1;
	    double *test_array2;

        test_array1	= (double *)calloc(hdm.ActCount, sizeof(double));
        test_array2 = (double *)calloc(hdm.ActCount, sizeof(double));
        
        for(k=0; k<(int)hdm.ActCount; k++) {
            map_lut[k] = 0;
            test_array1[k] = 0.5;
            test_array2[k] = 0;
        }
        */
        
        //BMCSetArray(&hdm, test_array1, map_lut);
        //cout << "hdm" << *hdm << endl;
        


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
        double fps = 0;
		fli->serialCamera()->getFps(fps);
        cout << "current Fps read: " << fps << endl;
        cout << "input desired fps value" << endl;
        cin >> fps;
		fli->serialCamera()->setFps(fps);
		fli->serialCamera()->getFps(fps);
		cout << "new Fps read: " << fps << endl;

        uint16_t width, height;
        fli->getCurrentImageDimension(width, height);
        cout << "width  =  " << width << endl;
        cout << "height  =  " << height << endl;
        cout << "input desired height" << endl;
        //cin >> height;      
        //fli->setImageDimension(width, height);
        //fli->getCurrentImageDimension(width, height);
        //cout << "new width  =  " << width << endl;
        //cout << "new height  =  " << height << endl;   

        // ====== ME FUCKING AROUND 
        //get the raw image 
        uint16_t* image16b = (uint16_t*)fli->getRawImage();

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


    //void applyDMshape(double *array1){ 
    //    rv = BMCSetArray(hdm, array1, map_lut);
    //}
    void normalizeImage(const std::span<uint16_t>& img, const std::span<uint16_t>& bias, float norm, std::span<float> normalizedImage) {
        if (img.size() != bias.size()) {
           throw std::invalid_argument("Image and bias must be the same size");
        }
    
        if (normalizedImage.size() != img.size()) {
            throw std::invalid_argument("Normalized image must be the same size as the input image");
        }

        for (size_t i = 0; i < img.size(); ++i) {
          normalizedImage[i] = (static_cast<float>(img[i]) - static_cast<float>(bias[i])) / norm;
        }
    }
    //give a image and a bias to normalize a signal


    //filters an image from pupil pixels 



    std::span<uint16_t> get_last_frame() { 
        
        return {(uint16_t*)fli->getRawImage(), 512*640}; // full dimension for C-RED3 = 512*640 
     
    }

    std::span<float> get_reconstructor(){

        std::span<float> reconstructor_span = reconstructor.current();
        return reconstructor_span;
    }


    std::span<float> test_normalization( ){

        size_t rows = 512; // Number of rows in the image
        size_t cols = 640; // Number of columns in the image

        //std::vector<uint16_t> img = {100, 150, 200};
        uint16_t* raw_image = (uint16_t*)fli->getRawImage(); // Retrieve raw image data
        std::span<uint16_t> img(raw_image, rows * cols); // Create a span from the raw image data

        std::span<uint16_t> biasSpan = bias.current(); //{50, 50, 50};
        //float norm = 50.0f;
        float norm = flux_norm.current();
        std::vector<float> nI(img.size());

        std::span<uint16_t> imgSpan(img);
        //std::span<uint16_t> biasSpan(bias);
        std::span<float> normalizedImageSpan(nI);

        normalizeImage(imgSpan, biasSpan, norm, normalizedImageSpan);
        return(normalizedImageSpan);
    }


    std::span<float> test_pixel_filter() {

        size_t rows = 512; // Number of rows in the image
        size_t cols = 640; // Number of columns in the image

        // Assuming pupil_pixels.current() and I0.current() return std::span<int> and std::span<float> respectively
        std::span<int> indices_span = pupil_pixels.current();

        // get image 
        uint16_t* raw_image = (uint16_t*)fli->getRawImage(); // Retrieve raw image data
        std::span<uint16_t> image_span(raw_image, rows * cols); // Create a span from the raw image data

        //get current reference intensity
        std::span<float> I0_span = I0.current();
        //get current bias
        std::span<uint16_t> bias_span = bias.current();
        // Get values at specified indices
        std::vector<uint16_t> bias_output = getValuesAtIndices(bias_span, indices_span);
        std::vector<float> I0_output = getValuesAtIndices(I0_span, indices_span);
        std::vector<uint16_t> im_output = getValuesAtIndices(image_span, indices_span) ;

        std::span<uint16_t> bias_output_span(bias_output);
        std::span<float> I0_output_span(I0_output);
        std::span<uint16_t> im_output_span(im_output);

        //testing to do normalization 
        float norm = flux_norm.current();
        std::vector<float> nI(im_output_span.size()); //pre-allocate vector for output of normalizeImage
        //std::span<uint16_t> imgSpan(img);
        std::span<float> normalizedImageSpan(nI); // this is placeholder for output (since function returns span..)

        normalizeImage(im_output_span, bias_output_span, norm, normalizedImageSpan);

        // dont forget to change function return type if changing output
        return normalizedImageSpan; //im_output_span ; //bias_output_span; //I0_output_span;
    }

    //std::span<const uint16_t>
    std::vector<float> test(){

        size_t rows = 512; // Number of rows in the image
        size_t cols = 640; // Number of columns in the image

        //size_t layers = 140; // Number of layers in the reconstructor matrix

        uint16_t* raw_image = (uint16_t*)fli->getRawImage(); // Retrieve raw image data

        std::span<const uint16_t> image_span(raw_image, rows * cols); // Create a span from the raw image data
        

        //
        //std::span<int> indices_span = pupil_pixels.current() ;


        // Get values of at specified indices
        std::span<const uint16_t> im = getValuesAtIndices(image_span, pupil_pixels.current() ) ; // image
        std::vector<uint16_t> b = getValuesAtIndices(bias.current(),  pupil_pixels.current()); // bias
        std::vector<float> I_ref = getValuesAtIndices(I0.current(),  pupil_pixels.current() ); // set point intensity

        //basic checks 
        assert(im.size() == b.size()); // Ensure sizes match between im and b
        assert(im.size() == I_ref.size()); // Ensure sizes match between im and I_ref
        
        // subtract bias, normalize and then subtract reference intensity 
        // to generate error signal
        size_t size = im.size();
        std::vector<float> signal(size);
        for (size_t i = 0; i < size; ++i) {
          signal[i] = (static_cast<float>(im[i]) - static_cast<float>(b[i])) / flux_norm.current() - I_ref[i];
        }

        // MVM to turn our error vector to a DM command via the reconstructor 
        std::vector<float> cmd;
        matrix_vector_multiply(im, reconstructor, cmd);
        //cout << hdm << endl;
        // send the cmd to the DM 
        // >>>>>> THIS compiles 
        //cout << "hdm" << &hdm << endl;
        //BMCSetSingle(&hdm, 65, 0.2);

        // >>>>>> THIS DOES NOT compile.. WHY ? 
        //rv = BMCSetArray(&hdm, cmd, map_lut);
        //rv = BMCSetSingle(&hdm, 65, 0.2);
        //map_lut	= (uint32_t *)malloc(sizeof(uint32_t)*MAX_DM_SIZE);
        //BMCSetArray(&hdm, cmd, map_lut);

        return cmd;
    }




    /**
     * @brief Performs computation using the current RTC values.
     * @param os The output stream to log some informations.
     */
    void compute(std::ostream& os)
    {
        os << "computing with " << (*this) << '\n';


        size_t rows = 512; // Number of rows in the image
        size_t cols = 640; // Number of columns in the image

        //get the raw image 
        uint16_t* raw_image = (uint16_t*)fli->getRawImage(); // Retrieve raw image data

        std::span<uint16_t> image_span(raw_image, rows * cols);
        //normalize signal 
        //std::span<float> I = normalize_signal(image_span, bias.next());

        //get pixels within the pupil control region 
        //std::span<float> err = I - I0;

        //compute error signal 


        // reconstructor[i + j * dim_x];

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
    

    void set_ctrl_matrix(std::span<float> mat) {
        reconstructor.update(mat);
    }

    void set_pupil_pixels(std::span<int> array) {
        pupil_pixels.update(array);
    }

    void set_bias(std::span<uint16_t> array) {
        bias.update(array);
    }

    void set_I0(std::span<float> array) {
        I0.update(array);
    }

    void set_fluxNorm(float value) {
        flux_norm.update(value);
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
        pupil_pixels.commit();
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



NB_MODULE(_rtc, m) {
    using namespace nb::literals;

    nb::class_<RTC>(m, "RTC")
        .def(nb::init<>())
        .def("compute", &RTC::compute)
        .def_rw("value", &RTC::value)
        .def("set_slope_offsets", &RTC::set_slope_offsets)
        .def("set_ctrl_matrix", &RTC::set_ctrl_matrix)
        .def("set_I0", &RTC::set_I0)
        .def("set_bias", &RTC::set_bias)
        .def("set_fluxNorm", &RTC::set_fluxNorm)
        .def("set_pupil_pixels", &RTC::set_pupil_pixels)
        .def("set_gain", &RTC::set_gain)
        .def("set_offset", &RTC::set_offset)
        .def("commit", &RTC::commit)
        .def("request_commit", &RTC::request_commit)
        .def("get_last_frame", &RTC::get_last_frame)
        .def("get_reconstructor", &RTC::get_reconstructor)
        //.def("normalizedImage", &RTC::normalizedImage)
        .def("test", &RTC::test)
        .def("test_normalization",&RTC::test_normalization)
        .def("test_pixel_filter",&RTC::test_pixel_filter)
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
