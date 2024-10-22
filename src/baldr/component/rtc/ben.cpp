#include <baldr/component/rtc/ben.hpp>
#include <baldr/utility/updatable.hpp>

namespace baldr::benrtc
{
    /*@brief to hold pixel indicies for different pupil regions*/
    struct pupil_regions {
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
    }

    /*@brief to hold things for ZWFS signal processing and projection onto modes/DM commands
    we have an array of reconstructor matricies here to allow flexibility in control implementation
    CM, R_TT, R_HO can be used to go directly from processed intensities to DM commands
    I2M can be used to go from processed intensities to a modal basis
    M2C can be used to go from a pre-defined modal basis to DM commands

    bias, I0, flux_norm are for signal processing

    with nano bind there were issues with these being span's - so changed them all to vectors!
    */
    struct phase_reconstuctor {
        /*
        we have an array of reconstructor matricies here to allow flexibility in control implementation
        CM, R_TT, R_HO can be used to go directly from processed intensities to DM commands
        I2M can be used to go from processed intensities to a modal basis
        M2C can be used to go from a pre-defined modal basis to DM commands

        bias, I0, flux_norm are for signal processing

        with nano bind there were issues with these being span's - so changed them all to vectors!
        */
        updatable<vector<float>> IM; /**<unfiltered interaction matrix  */
        updatable<vector<float>> CM; /**< control matrix (~ M2C @ I2M.T) signal intensities to dm commands  */
        updatable<vector<float>> R_TT; /**< tip/tilt reconstructor */
        updatable<vector<float>> R_HO; /**< higher-order reconstructor */
        updatable<vector<float>> I2M; /**< intensity (signal) to mode matrix */
        //std::span<float> I2M_a // again there is a bug with updatable I2M as with I0...
        updatable<vector<float>> M2C; /**< mode to DM command matrix. */

        updatable<vector<uint16_t>> bias; /**< bias. */
        updatable<vector<float>> I0; /**< reference intensity with FPM in. */
        updatable<float> flux_norm; /**< for normalizing intensity across detector. */
    };

    phase_reconstuctor tag_invoke( json::value_to_tag< phase_reconstuctor >, json::value const& jv )
    {
        phase_reconstuctor reco;

        json::object const& obj = jv.as_object();

        reco.IM         = json::object_to<vector<float>    >(obj, "IM"       );
        reco.CM         = json::object_to<vector<float>    >(obj, "CM"       );
        reco.R_TT       = json::object_to<vector<float>    >(obj, "R_TT"     );
        reco.R_HO       = json::object_to<vector<float>    >(obj, "R_HO"     );
        reco.I2M        = json::object_to<vector<float>    >(obj, "I2M"      );
        reco.M2C        = json::object_to<vector<float>    >(obj, "M2C"      );
        reco.bias       = json::object_to<vector<uint16_t> >(obj, "bias"     );
        reco.I0         = json::object_to<vector<float>    >(obj, "I0"       );
        reco.flux_norm  = json::object_to<       float     >(obj, "flux_norm");

        return reco;
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

        RTC(json::object config)
            : regions(json::object_to<pupil_regions>     (config, "regions"))
            , reco   (json::object_to<phase_reconstuctor>(config, "reco"))
        {}

        void compute(std::span<const uint16_t> frame, std::span<double> signal) {
            auto& current_io = reco.I0.current();

            using std::ranges::views::transform;

            auto frame_lut = as_lut<double>(frame);
            auto frame_sorted = regions.pupil_pixels.current() | transform(frame_lut);

            auto current_io_lut = as_lut<double>(current_io);
            auto I_ref = regions.pupil_pixels.current() | transform(current_io_lut);

            for (size_t i = 0; i < signal.size(); ++i) {
                signal[i] = frame_sorted[i] / reco.flux_norm.current() - I_ref[i];
            }
        }

    };

    std::unique_ptr<interface::RTC> make_rtc(json::object config) {
        return std::make_unique<RTC>(config);
    }

} // namespace baldr::benrtc
