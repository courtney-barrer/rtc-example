#include <baldr/component/rtc/ben.hpp>
#include <baldr/utility/updatable.hpp>
#include <iostream>


namespace baldr::benrtc
{


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
                std::cout << "integrals.size() != size.. reinitializing integrals, prev_errors and output to zero with correct size" << std::endl; 
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

    /*@brief to hold things for ZWFS signal processing and projection onto modes/DM commands
    we have an array of reconstructor matricies here to allow flexibility in control implementation
    CM, R_TT, R_HO can be used to go directly from processed intensities to DM commands
    I2M can be used to go from processed intensities to a modal basis
    M2C can be used to go from a pre-defined modal basis to DM commands

    bias, I0, flux_norm are for signal processing

    /*@brief to hold pixel indicies for different pupil regions*/
    struct pupil_regions {
        // OUTSIDE PIXELS CAN BE SELECTED FOR ESTIMATING STREHL
        updatable<vector<int>> pupil_pixels; /**< pixels inside the active pupil. */
        updatable<vector<int>> secondary_pixels; /**< pixels inside secondary obstruction   */
        updatable<vector<int>> outside_pixels; /**< pixels outside the active pupil obstruction (but not in secondary obstruction) */
    };


    pupil_regions tag_invoke( json::value_to_tag< pupil_regions >, json::value const& jv )
    {
        pupil_regions regions;

        json::object const& obj = jv.as_object();
      
        regions.pupil_pixels     = json::object_to<vector<int>>(obj, "pupil_pixels"    );
        regions.secondary_pixels = json::object_to<vector<int>>(obj, "secondary_pixels");
        regions.outside_pixels   = json::object_to<vector<int>>(obj, "outside_pixels"  );

        return regions;
    };


    struct phase_reconstuctor {
        /*
        we have an array of reconstructor matricies here to allow flexibility in control implementation
        CM, R_TT, R_HO can be used to go directly from processed intensities to DM commands
        I2M can be used to go from processed intensities to a modal basis
        M2C can be used to go from a pre-defined modal basis to DM commands

        bias, I0, flux_norm are for signal processing

        with nano bind there were issues with these being spans - so changed them all to vectors!
        */

        // MANY HERE ARE REDUNDANTE - WE CAN DELETE SOME 
        std::vector<double> dm_flat; 
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

        updatable<vector<uint16_t>> bias; /**< bias. */
        updatable<vector<float>> I0; /**< reference intensity with FPM in. */
        updatable<std::vector<float>> N0; /**< reference intensity with FPM out. */
        updatable<float> flux_norm; /**< for normalizing intensity across detector. */
    };


    phase_reconstuctor tag_invoke( json::value_to_tag< phase_reconstuctor >, json::value const& jv )
    {
        phase_reconstuctor reco;

        json::object const& obj = jv.as_object();
        reco.dm_flat    = json::object_to<vector<double>    >(obj, "dm_flat"  );
        reco.IM         = json::object_to<vector<float>    >(obj, "IM"       );
        reco.CM         = json::object_to<vector<float>    >(obj, "CM"       );
        reco.R_TT       = json::object_to<vector<float>    >(obj, "R_TT"     );
        reco.R_HO       = json::object_to<vector<float>    >(obj, "R_HO"     );
        reco.I2M        = json::object_to<vector<float>    >(obj, "I2M"      );
        reco.I2M_TT     = json::object_to<vector<float>    >(obj, "I2M_TT"   );
        reco.I2M_HO     = json::object_to<vector<float>    >(obj, "I2M_HO"   );
        reco.M2C        = json::object_to<vector<float>    >(obj, "M2C"      );
        reco.M2C_TT     = json::object_to<vector<float>    >(obj, "M2C_TT"   );
        reco.M2C_HO     = json::object_to<vector<float>    >(obj, "M2C_HO"   );
        reco.bias       = json::object_to<vector<uint16_t> >(obj, "bias"     );
        reco.I0         = json::object_to<vector<float>    >(obj, "I0"       );
        reco.flux_norm  = json::object_to<       float     >(obj, "flux_norm");

        return reco;
    }


    PIDController tag_invoke( json::value_to_tag< PIDController >, json::value const& jv )
    {
        json::object const& obj = jv.as_object();
        PIDController pid;
        pid.kp            = json::object_to< vector<double> >(obj, "pid_kp"  );
        pid.ki            = json::object_to< vector<double> >(obj, "pid_ki"  );
        pid.kd            = json::object_to< vector<double> >(obj, "pid_kd"  );
        pid.lower_limit   = json::object_to< vector<double> >(obj, "pid_lower_limit"  );
        pid.upper_limit   = json::object_to< vector<double> >(obj, "pid_upper_limit"  );
        pid.setpoint     = json::object_to< vector<double> >(obj, "pid_setpoint"  );
        return pid;
    }


    LeakyIntegrator tag_invoke( json::value_to_tag< LeakyIntegrator >, json::value const& jv )
    {
        json::object const& obj = jv.as_object();
        LeakyIntegrator leakyInt;
        leakyInt.rho           = json::object_to< vector<double> >(obj, "leaky_rho"  );
        leakyInt.kp            = json::object_to< vector<double> >(obj, "leaky_kp"  );
        leakyInt.lower_limit   = json::object_to< vector<double> >(obj, "leaky_lower_limit"  );
        leakyInt.upper_limit   = json::object_to< vector<double> >(obj, "leaky_upper_limit"  );
        return leakyInt;
    }

    template<typename ReturnType, typename T>
    auto as_lut(T&& tt) {
        return [lut = EMU_FWD(tt)] (size_t index) -> ReturnType {
            return static_cast<ReturnType>(lut[index]);
        };
    }

    struct RTC : interface::RTC
    {
        pupil_regions regions;
        phase_reconstuctor reco;
        PIDController pid; // object to apply PID control to signals
        LeakyIntegrator leakyInt; // object to apply PID control to signals

        std::vector<float> image_err_signal; 
        std::vector<double> TT_cmd_err; // holds the DM Tip-Tilt command offset (error) from the reference flat DM surface
        std::vector<double> HO_cmd_err; // holds the DM Higher Order command offset (error) from the reference flat DM surface
        //std::vector<double> mode_err; // holds mode error from matrix multiplication of I2M with processed signal 
        std::vector<double> mode_err_TT; // holds mode error specifically from matrix multiplication of I2M_TT with processed signal 
        std::vector<double> mode_err_HO; // holds mode error specifically from matrix multiplication of I2M_HO with processed signal 
        std::vector<double> dm_disturb; // a disturbance that we can add to the DM 
        //std::vector<double> dm_flat; // calibrated DM flat 
        //std::vector<double> pid_setpoint // set-point of PID controller


        RTC(json::object config)
            :   regions(json::object_to<pupil_regions>     (config, "regions")),
                reco(json::object_to<phase_reconstuctor>(config, "reco")),
                pid(json::object_to<PIDController>(config, "pid")),
                leakyInt(json::object_to<LeakyIntegrator>(config, "leakyInt")),
                dm_disturb(140, 0),
                TT_cmd_err(140, 0.0),
                HO_cmd_err(140, 0.0),
                mode_err_TT(pid.kp.size(), 0),
                mode_err_HO(leakyInt.kp.size(), 0)

        {}
        //any DM commands to length 140 
        

        void compute(std::span<const uint16_t> frame, std::span<double> dm_cmd) {
            auto& current_io = reco.I0.current();
            auto& I2M_TT = reco.I2M_TT.current();
            auto& M2C_TT = reco.M2C_TT.current();

            // variables to define ? ?
            /*
            mode_err_TT
            TT_cmd_err
            mode_err_HO
            HO_cmd_err
            dm_flat
            dm_disturb
            */
            using std::ranges::views::transform;

            auto frame_lut = as_lut<double>(frame);
            auto frame_sorted = regions.pupil_pixels.current() | transform(frame_lut);

            auto current_io_lut = as_lut<double>(current_io);
            auto I_ref = regions.pupil_pixels.current() | transform(current_io_lut);

            //process signal 
            for (size_t i = 0; i < frame_sorted.size(); ++i) {
                image_err_signal[i] = frame_sorted[i] / reco.flux_norm.current() - I_ref[i];
            }
            

            ///////////////////////
            // TIP / TILT 
            matrix_vector_multiply( image_err_signal, I2M_TT, mode_err_TT ) ;
            pid.process( mode_err_TT ) ;

            // // note - using DOUBLE matrix_vector_multiply here that also has parallel component
            matrix_vector_multiply_double( pid.output, reco.M2C_TT.current(), TT_cmd_err ) ;

            ///////////////////////
            // HIGHER ORDER 
            matrix_vector_multiply( image_err_signal, reco.I2M_HO.current(), mode_err_HO ) ;
            leakyInt.process( mode_err_HO ) ;
            // note - using DOUBLE matrix_vector_multiply here that also has parallel component
            matrix_vector_multiply_double( leakyInt.output, reco.M2C_HO.current(), HO_cmd_err ) ;
        
            for (size_t i = 0; i < reco.dm_flat.size(); ++i) {
                //dm_cmd[i] = TT_cmd_err[i] + HO_cmd_err[i];
                dm_cmd[i] = reco.dm_flat[i] + TT_cmd_err[i] + HO_cmd_err[i] + dm_disturb[i];  // comment out HO_cmd_err if desired
                std::cout << dm_cmd[i] << std::endl;
            }
        
        }

    };

    std::unique_ptr<interface::RTC> make_rtc(json::object config) {
        return std::make_unique<RTC>(config);
    }

} // namespace baldr::benrtc
