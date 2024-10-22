
#include <baldr/camera.hpp>
#include <baldr/component/camera/flicam.hpp>
#include <baldr/type.hpp>

#include <memory>
#include <sardine/type/json.hpp>
#include <ranges>
#include <stdexcept>

#ifdef BALDR_FLI

#include <FliSdk.h>

namespace baldr::flicam
{

    namespace json = myboost::json;

    /*@brief to hold camera settings*/
    struct CameraSettings {
        int camera_index;
        double det_dit = 0.0016; // detector integration time (s)
        double det_fps = 600.0; // frames per second (Hz)
        string det_gain="medium"; // "low" or "medium" or "high"
        bool det_crop_enabled = false; //true/false
        bool det_tag_enabled = false; //true/false
        string det_cropping_rows="0-639"; //"r1-r2" where  r1 multiple of 4, r2 multiple 4-1
        string det_cropping_cols="0-511"; //"c1-c2" where c1 multiple of 32, c2 multiple 32-1
        uint16_t image_height = 640; // i.e. rows in image
        uint16_t image_width = 512;// i.e. cols in image
        uint32_t full_image_length = 327680; //640*512;
    };

    CameraSettings tag_invoke(
        const myboost::json::value_to_tag< CameraSettings >&,
        myboost::json::value const& jv
    )
    {
        CameraSettings cm;

        json::object const& obj = jv.as_object();

        auto extract = [&]( auto& t, string_view key, auto default_ ) {
            if (obj.contains(key))
                t = value_to<decltype(default_)>( obj.at( key ) );
            else
                t = default_;
        };

        using namespace std::string_literals;

        extract( cm.camera_index,      "camera_index",      0 );
        extract( cm.det_dit,           "det_dit",           0.0016 );
        extract( cm.det_fps,           "det_fps",           600.0);
        extract( cm.det_gain,          "det_gain",          "medium"s);
        extract( cm.det_crop_enabled,  "det_crop_enabled",  false);
        extract( cm.det_tag_enabled,   "det_tag_enabled",   false);
        extract( cm.det_cropping_rows, "det_cropping_rows", "0-639"s);
        extract( cm.det_cropping_cols, "det_cropping_cols", "0-511"s);
        extract( cm.image_height,      "image_height",      640);
        extract( cm.image_width,       "image_width",       512);
        extract( cm.full_image_length, "full_image_length", 327680);

        return cm;
    }


    void apply_camera_settings( FliSdk& fli_sdk, CameraSettings& cm){
        // NEED TO TEST IN SYDNEY ON CAMERA
        // does not check if in simulation mode!

        //double fps = 0;

        fli_sdk.stop();

        auto& camera = *fli_sdk.serialCamera();

        // crop first
        camera.sendCommand("set cropping off"); //FliCamera_sendCommand("set cropping off");
        if (cm.det_crop_enabled) {
            //set cropping and enable
            camera.sendCommand("set cropping rows "+ cm.det_cropping_rows);
            camera.sendCommand("set cropping columns "+ cm.det_cropping_cols);
            camera.sendCommand("set cropping on");
        }

        if (cm.det_tag_enabled) {
            // makes first pixels correspond to frame number and other info
            //
            //TO DO: should make corresponding mask for this to be added to
            //pixel_filter if this is turned on to ensure frame count etc
            //does not get interpretted as intensities.
            camera.sendCommand("set imagetags on");
        } else{
            camera.sendCommand("set imagetags off");
        }

        camera.sendCommand("set cropping rows "+ cm.det_cropping_rows);
        camera.sendCommand("set cropping cols "+ cm.det_cropping_cols);

        camera.sendCommand("set cropping cols "+ cm.det_cropping_cols);

        camera.sendCommand("set sensitivity "+ cm.det_gain);

        //set fps
        camera.setFps(cm.det_fps);

        //set int
        camera.sendCommand("set tint " + std::to_string(cm.det_dit));

        //camera.getFps(fps);
        //cout << "fps despues = " << fps << endl;

        fli_sdk.update();

        //uint16_t width, height;
        fli_sdk.getCurrentImageDimension(cm.image_width, cm.image_height);
        fmt::print("image width  = {}\n", cm.image_width);
        fmt::print("image height  = {}\n", cm.image_height);
        uint32_t _full_image_length_new = static_cast<uint32_t>(cm.image_width) * static_cast<uint32_t>(cm.image_height);

        // _full_image_length_new can be used to update simulated signals etc before appending
        if (_full_image_length_new != cm.full_image_length){
            fmt::print("_full_image_length_new != cm.full_image_length\n");
            // update the full image length
            cm.full_image_length = _full_image_length_new;
        }

        fli_sdk.start();

    }

    struct FliCam : interface::Camera
    {
        CameraSettings camera_settings;
        std::unique_ptr<FliSdk> fli_sdk;

        FliCam(json::object config)
            : //camera_settings(myboost::json::value_to<CameraSettings>(json::value(config)))
             fli_sdk(std::make_unique<FliSdk>())
        {
            fmt::print("Detection of grabbers...\n");
            auto listOfGrabbers = fli_sdk->detectGrabbers();
            fmt::print("Detection of cameras...\n");
            auto listOfCameras = fli_sdk->detectCameras();

            if(listOfGrabbers.size() == 0)
                throw std::runtime_error("No grabber detected, exit. Putting camera in simulation mode..");
            else if (listOfCameras.size() == 0)
                throw std::runtime_error("No camera detected, exit. Putting camera in simulation mode..");
            else{
                int i = 0;
                for(auto& c : listOfCameras)
                    fmt::print("- {} -> {}\n", i++, c);

                if(camera_settings.camera_index >= listOfCameras.size()) {
                    fmt::print("trying to access out of index camera, use 0 instead\n");
                    camera_settings.camera_index = 0;
                }

                auto cameraName = listOfCameras[camera_settings.camera_index];
                fmt::print("Using camera {}\n", cameraName);
                //take the first camera in the list
                fli_sdk->setCamera(cameraName);

                // --------------- RESTORE DEFAULT SETTINGS ON INIT -----
                //fli_sdk->serialCamera()->sendCommand("restorefactory");

                fmt::print("Setting mode Full.\n");
                //set full mode
                fli_sdk->setMode(FliSdk::Mode::Full);

                fmt::print("Update the SDK... ");
                //update
                fli_sdk->update();
                fmt::print("Done.\n");


                fli_sdk->start();
                fli_sdk->imageProcessing()->enableAutoClip(true);

            }

            apply_camera_settings(*fli_sdk, camera_settings);
        }

        void get_frame(std::span<uint16_t> frame) {
            std::span<const uint16_t> current_frame(reinterpret_cast<const uint16_t*>(fli_sdk->getRawImage()), camera_settings.full_image_length);

            std::ranges::copy(current_frame, frame.data());
        }

    };


    std::unique_ptr<interface::Camera> make_camera(json::object config) {
        return std::make_unique<FliCam>(config);
    }

} // namespace baldr::flicam

#else

namespace baldr::flicam
{

    std::unique_ptr<interface::Camera> make_camera(json::object config) {
        throw std::runtime_error("Error fli camera support is not compiled");
    }

} // namespace baldr::flicam

#endif
