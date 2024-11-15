
#include <baldr/component/camera.hpp>
#include <baldr/component/camera/flicam.hpp>
#include <baldr/type.hpp>

#include <memory>
#include <sardine/type/json.hpp>
#include <ranges>
#include <stdexcept>

#ifdef BALDR_FLI

#include <FliSdk.h>

#include <map>
#include <string>


namespace baldr::flicam
{

    namespace json = boost::json;



    /*@brief to hold camera settings*/
    // struct CameraSettings {
    //     int camera_index = 0;
    //     double det_dit = 0.0016; // detector integration time (s)
    //     double det_fps = 600.0; // frames per second (Hz)
    //     string det_gain="medium"; // "low" or "medium" or "high"
    //     bool det_crop_enabled = false; //true/false
    //     bool det_tag_enabled = false; //true/false
    //     string det_cropping_rows="0-320"; //"r1-r2" where  r1 multiple of 4, r2 multiple 4-1
    //     string det_cropping_cols="0-256"; //"c1-c2" where c1 multiple of 32, c2 multiple 32-1
    //     uint16_t image_height = 320; // i.e. rows in image
    //     uint16_t image_width = 256;// i.e. cols in image
    //     uint32_t full_image_length = 256*320; //640*512;
    // };

    struct CameraSettings {
        int camera_index = 0;
        std::string fps = "1000";
        std::string gain = "1";
        std::string testpattern = "off";
        std::string bias = "off";
        std::string flat = "off";
        std::string imagetags = "off";
        std::string led = "off";
        std::string events = "off";
        std::string extsynchro = "off";
        std::string rawimages = "off";
        std::string cooling = "off";
        std::string standby = "off";
        std::string mode = "globalresetsingle";
        std::string resetwidth = "15";
        std::string nbreadworeset = "1";
        std::string cropping = "off";
        std::string aduoffset = "0";
        std::string cropping_columns = "1-320";
        std::string cropping_rows = "1-256";

        uint16_t image_height = 320;
        uint16_t image_width = 256;
        uint32_t full_image_length = 256 * 320;
    };


    std::map<std::string, std::string> camera_settings_to_map(const CameraSettings& cm) {
        return {
            {"fps", cm.fps},
            {"gain", cm.gain},
            {"testpattern", cm.testpattern},
            {"bias", cm.bias},
            {"flat", cm.flat},
            {"imagetags", cm.imagetags},
            {"led", cm.led},
            {"events", cm.events},
            {"extsynchro", cm.extsynchro},
            {"rawimages", cm.rawimages},
            {"cooling", cm.cooling},
            {"standby", cm.standby},
            {"mode", cm.mode},
            {"resetwidth", cm.resetwidth},
            {"nbreadworeset", cm.nbreadworeset},
            {"cropping", cm.cropping},
            {"aduoffset", cm.aduoffset},
            {"cropping columns", cm.cropping_columns},
            {"cropping rows", cm.cropping_rows}
        };
    }



    // CameraSettings tag_invoke(
    //     const boost::json::value_to_tag< CameraSettings >&,
    //     boost::json::value const& jv
    // )
    // {
    //     CameraSettings cm;

    //     json::object const& obj = jv.as_object();
    //     //// this extract caused errors (30/10/24)
    //     // auto extract = [&]( auto& t, string_view key) {
    //     //     if (obj.contains(key))
    //     //         t = value_to<decltype(t)>( obj.at( key ) );

    //     // };

    //     auto extract = [&](auto& t, string_view key) {
    //         using T = std::decay_t<decltype(t)>;  // Remove references for compatibility
    //         if (obj.contains(key)) {
    //             t = value_to<T>(obj.at(key));  // Use a non-reference type for extraction
    //         }
    //     };

    //     using namespace std::string_literals;

    //     extract( cm.camera_index,      "camera_index");
    //     extract( cm.det_dit,           "det_dit");
    //     extract( cm.det_fps,           "det_fps");
    //     extract( cm.det_gain,          "det_gain");
    //     extract( cm.det_crop_enabled,  "det_crop_enabled");
    //     extract( cm.det_tag_enabled,   "det_tag_enabled");
    //     extract( cm.det_cropping_rows, "det_cropping_rows");
    //     extract( cm.det_cropping_cols, "det_cropping_cols");
    //     extract( cm.image_height,      "image_height");
    //     extract( cm.image_width,       "image_width");
    //     extract( cm.full_image_length, "full_image_length");

    //     return cm;
    // }

    // CameraSettings tag_invoke(
    //     const boost::json::value_to_tag<CameraSettings>&,
    //     const boost::json::value& jv
    // ) {
    //     CameraSettings cm;
    //     const auto& obj = jv.as_object();

    //     auto extract = [&](auto& member, std::string_view key) {
    //         using T = std::decay_t<decltype(member)>;
    //         if (obj.contains(key)) {
    //             member = boost::json::value_to<T>(obj.at(key));
    //         }
    //     };


    /// SECOND ATTEMPT
    //     CameraSettings tag_invoke(
    //         const boost::json::value_to_tag<CameraSettings>&,
    //         const boost::json::value& jv
    //     ) {
    //         // CameraSettings cm;
    //         // const auto& obj = jv.as_object();

    //         // auto extract = [&](auto& member, std::string_view key) {
    //         //     if (obj.contains(key)) {
    //         //         member = obj.at(key).as_string();  // Directly extract as a string
    //         //     }
    //         // };
    //         CameraSettings cm;

    //         // json::object const& obj = jv.as_object();
    //         // //// this extract caused errors (30/10/24)
    //         // auto extract = [&]( auto& t, string_view key) {
    //         //     if (obj.contains(key))
    //         //         t = value_to<decltype(t)>( obj.at( key ) );
    //         // };

    //         const auto& obj = jv.as_object();
    //         auto extract = [&](auto& member, std::string_view key) {
    //             using T = std::decay_t<decltype(member)>;
    //             if (obj.contains(key)) {
    //                 member = boost::json::value_to<T>(obj.at(key));
    //             }
    //         };


    //         // Extract fields from JSON and assign to CameraSettings members
    //         extract(cm.fps, "fps");
    //         extract(cm.gain, "gain");
    //         extract(cm.testpattern, "testpattern");
    //         extract(cm.bias, "bias");
    //         extract(cm.flat, "flat");
    //         extract(cm.imagetags, "imagetags");
    //         extract(cm.led, "led");
    //         extract(cm.events, "events");
    //         extract(cm.extsynchro, "extsynchro");
    //         extract(cm.rawimages, "rawimages");
    //         extract(cm.cooling, "cooling");
    //         extract(cm.standby, "standby");
    //         extract(cm.mode, "mode");
    //         extract(cm.resetwidth, "resetwidth");
    //         extract(cm.nbreadworeset, "nbreadworeset");
    //         extract(cm.cropping, "cropping");
    //         extract(cm.aduoffset, "aduoffset");
    //         extract(cm.cropping_columns, "cropping columns");  // Replace space with underscore in struct member
    //         extract(cm.cropping_rows, "cropping rows");        // Replace space with underscore in struct member

    //         // Extract numerical fields
    //         if (obj.contains("camera_index")) {
    //             cm.camera_index = boost::json::value_to<int>(obj.at("camera_index"));
    //         }
    //         // just use default , this gets automatically updated after applyimg settings and getting image!
    //         cm.image_height = 320;
    //         cm.image_width = 256;
    //         cm.full_image_length = 320*256;

    //         return cm;
    //     }

    CameraSettings tag_invoke(
        const boost::json::value_to_tag<CameraSettings>&,
        const boost::json::value& jv
    ) {
        CameraSettings cm;
        const auto& obj = jv.as_object();

        auto extract = [&](auto& member, std::string_view key) {
            using T = std::decay_t<decltype(member)>;
            if (obj.contains(key)) {
                member = boost::json::value_to<T>(obj.at(key));
            }
        };

        extract(cm.fps, "fps");
        extract(cm.gain, "gain");
        extract(cm.testpattern, "testpattern");
        extract(cm.bias, "bias");
        extract(cm.flat, "flat");
        extract(cm.imagetags, "imagetags");
        extract(cm.led, "led");
        extract(cm.events, "events");
        extract(cm.extsynchro, "extsynchro");
        extract(cm.rawimages, "rawimages");
        extract(cm.cooling, "cooling");
        extract(cm.standby, "standby");
        extract(cm.mode, "mode");
        extract(cm.resetwidth, "resetwidth");
        extract(cm.nbreadworeset, "nbreadworeset");
        extract(cm.cropping, "cropping");
        extract(cm.aduoffset, "aduoffset");
        extract(cm.cropping_columns, "cropping columns");
        extract(cm.cropping_rows, "cropping rows");

        if (obj.contains("camera_index")) {
            cm.camera_index = boost::json::value_to<int>(obj.at("camera_index"));
        }

        // Use defaults
        cm.image_height = 320;
        cm.image_width = 256;
        cm.full_image_length = 320 * 256;

        return cm;
    }

    void apply_camera_settings(FliSdk& fli_sdk, CameraSettings& cm) {
        // Stop the camera to apply settings
        fli_sdk.stop();

        auto& camera = *fli_sdk.serialCamera();

        // Get all settings as a map of key-value pairs
        auto settings = camera_settings_to_map(cm);

        // Iterate over each setting and apply it to the camera
        for (const auto& [key, value] : settings) {
            std::string command = "set " + key + " " + value;
            camera.sendCommand(command);


            // Log success or failure
            // if (!ok) {
            //     fmt::print("Failed to send command: {}\n", command);
            // } else {
            //     fmt::print("Successfully sent command: {}\n", command);
            // }

            // //Update cm based on key
            // if (key == "fps") cm.fps = value.as_string();
            // else if (key == "gain") cm.gain = value.as_string();
            // else if (key == "testpattern") cm.testpattern = value.as_string();
            // else if (key == "bias") cm.bias = value.as_string();
            // else if (key == "flat") cm.flat = value.as_string();
            // else if (key == "imagetags") cm.imagetags = value.as_string();
            // else if (key == "led") cm.led = value.as_string();
            // else if (key == "events") cm.events = value.as_string();
            // else if (key == "extsynchro") cm.extsynchro = value.as_string();
            // else if (key == "rawimages") cm.rawimages = value.as_string();
            // else if (key == "cooling") cm.cooling = value.as_string();
            // else if (key == "standby") cm.standby = value.as_string();
            // else if (key == "mode") cm.mode = value.as_string();
            // else if (key == "resetwidth") cm.resetwidth = value.as_string();
            // else if (key == "nbreadworeset") cm.nbreadworeset = value.as_string();
            // else if (key == "cropping") cm.cropping = value.as_string();
            // else if (key == "aduoffset") cm.aduoffset = value.as_string();
            // else if (key == "cropping columns") cm.cropping_columns = value.as_string();
            // else if (key == "cropping rows") cm.cropping_rows = value.as_string();
        }

        // Update SDK after all settings are applied
        fli_sdk.update();

        // Update image dimensions and log them
        uint16_t image_width, image_height;
        fli_sdk.getCurrentImageDimension(image_width, image_height);
        cm.image_width = image_width;
        cm.image_height = image_height;
        cm.full_image_length = static_cast<uint32_t>(image_width) * image_height;

        fli_sdk.start();
    }



    // void apply_camera_settings( FliSdk& fli_sdk, CameraSettings& cm){
    //     // NEED TO TEST IN SYDNEY ON CAMERA
    //     // does not check if in simulation mode!

    //     //double fps = 0;

    //     fli_sdk.stop();

    //     auto& camera = *fli_sdk.serialCamera();



    //     // crop first
    //     camera.sendCommand("set cropping off"); //FliCamera_sendCommand("set cropping off");
    //     if (cm.det_crop_enabled) {
    //         //set cropping and enable
    //         camera.sendCommand("set cropping rows " + cm.det_cropping_rows);
    //         camera.sendCommand("set cropping columns " + cm.det_cropping_cols);
    //         camera.sendCommand("set cropping on");
    //     }

    //     if (cm.det_tag_enabled) {
    //         // makes first pixels correspond to frame number and other info
    //         //
    //         //TO DO: should make corresponding mask for this to be added to
    //         //pixel_filter if this is turned on to ensure frame count etc
    //         //does not get interpretted as intensities.
    //         camera.sendCommand("set imagetags on");
    //     } else{
    //         camera.sendCommand("set imagetags off");
    //     }

    //     camera.sendCommand("set cropping rows "+ cm.det_cropping_rows);
    //     camera.sendCommand("set cropping cols "+ cm.det_cropping_cols);

    //     camera.sendCommand("set cropping cols "+ cm.det_cropping_cols);
    //     //set sensitivity for cred3 / 2  set gain for cred1
    //     camera.sendCommand("set sensitivity " + cm.det_gain);

    //     //set fps
    //     //camera.setFps(cm.det_fps);
    //     //camera.sendCommand("set fps " + std::to_string(cm.det_fps));
    //     camera.setFps(cm.det_fps);
    //     //cred 1 doesn't seem to have tint setting (infered from fps? ) set int
    //     //camera.sendCommand("set tint " + std::to_string(cm.det_dit));

    //     //camera.getFps(fps);
    //     //cout << "fps despues = " << fps << endl;

    //     fli_sdk.update();

    //     //uint16_t width, height;
    //     fli_sdk.getCurrentImageDimension(cm.image_width, cm.image_height);
    //     fmt::print("image width  = {}\n", cm.image_width);
    //     fmt::print("image height  = {}\n", cm.image_height);
    //     uint32_t _full_image_length_new = static_cast<uint32_t>(cm.image_width) * static_cast<uint32_t>(cm.image_height);

    //     // _full_image_length_new can be used to update simulated signals etc before appending
    //     if (_full_image_length_new != cm.full_image_length){
    //         fmt::print("_full_image_length_new != cm.full_image_length\n");
    //         // update the full image length
    //         cm.full_image_length = _full_image_length_new;
    //     }

    //     fli_sdk.start();

    // }



    struct FliCam final : interface::Camera, IRawImageReceivedObserver
    {
        CameraSettings camera_settings;
        std::unique_ptr<FliSdk> fli_sdk;

        std::span<const uint16_t> last_frame;

        bool running = false;


        FliCam(CameraLogic cam_logic, json::object config)
            : interface::Camera(std::move(cam_logic))
            , camera_settings(boost::json::value_to<CameraSettings>(json::value(config)))
            ,  fli_sdk(std::make_unique<FliSdk>())
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


            }

            apply_camera_settings(*fli_sdk, camera_settings);



            // We don't care about fli internal ring buffer. We disable it.
            fli_sdk->enableRingBuffer(true);

            // We register the current object as an observer of the camera.
            // We directly work on the grabber buffer hence the "beforeCopy".
            fli_sdk->addRawImageReceivedObserver(this, /* beforeCopy = */ false ) ;//true);

            fli_sdk->imageProcessing()->enableAutoClip(true);
        }

        ~FliCam() {
            if (fli_sdk)
                fli_sdk->stop();
        }

        void set_command(Command new_command) override  // implement interface::Camera
        {
            switch (new_command) {
                case Command::pause:
                    running = false;
                    stop();
                    break;
                case Command::run:
                case Command::step:
                    running = true;
                    start();
                    break;
                default: break;
            };
        }

        void start() {
            if (not fli_sdk->isStarted()){
                fmt::print("starting the camera\n");
                fli_sdk->start();
            }
        }
        void stop() {
            if (fli_sdk->isStarted()) {
                fmt::print("stopping the camera\n");

                fli_sdk->stop();
            }
        }


        std::span<const uint16_t> get_last_frame() const override // implement interface::Camera
        {
            return last_frame;
        }

        void imageReceived(const uint8_t* image) override // implement IRawImageReceivedObserver
        {
            last_frame = std::span{reinterpret_cast<const uint16_t*>(image), camera_settings.full_image_length};

            if (running) {
                static int cnt = 0;
                //fmt::print("cnt = {}\n", cnt++);
                send_frame(last_frame);
            }
        }

        uint16_t fpsTrigger() override // implement IRawImageReceivedObserver
        {
            // Determine the fps. 0 means "as soon as their is a new frame".
            return 0;
        }

    };


    std::future<void> make_camera(ComponentInfo& ci, CameraLogic cam_logic, json::object config, bool async) {
        return std::make_unique<FliCam>(std::move(cam_logic), config);
    }

} // namespace baldr::flicam

#else

namespace baldr::flicam
{

    std::future<void> make_camera(ComponentInfo& ci, CameraLogic cam_logic, json::object config, bool async) {
        throw std::runtime_error("Error fli camera support is not compiled");
    }

} // namespace baldr::flicam

#endif
