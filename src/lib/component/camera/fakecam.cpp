#include <baldr/component/camera.hpp>
#include <baldr/component/camera/fakecam.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <random>
#include <chrono>
#include <thread>

namespace baldr::fakecam
{

    void fill_with_random_values(std::span<uint16_t> data) {
        // Create a random device and seed the random engine
        std::random_device rd;
        std::mt19937 rd_generator(rd());

        // Define a distribution in the range of uint16_t
        std::uniform_int_distribution<uint16_t> distribution(0, std::numeric_limits<uint16_t>::max());

        auto generator = [&]() { return distribution(rd_generator); };

        // Fill the vector with random values
        std::ranges::generate(data, generator);
    }

    using now_type = std::chrono::time_point<std::chrono::system_clock>;

    struct FakeCamera final : interface::Camera
    {

        ComponentInfo* ci;
        std::vector<uint16_t> data;
        size_t frame_size;
        size_t frame_number;
        size_t index;
        std::chrono::microseconds latency;
        now_type now;

        bool running = false;

        std::jthread thread;

        FakeCamera(ComponentInfo& ci, CameraLogic cam_logic, size_t new_frame_size, size_t new_frame_number, std::chrono::microseconds new_latency)
            : interface::Camera(std::move(cam_logic))
            , ci(&ci)
            , data(new_frame_size * new_frame_number)
            , frame_size(new_frame_size)
            , frame_number(new_frame_number)
            , index(0)
            , latency(new_latency)
            , now( std::chrono::system_clock::now() )
        {
            fill_with_random_values(data);

            thread = std::jthread([this](std::stop_token stoken) mutable {
                while(true) {
                    if (stoken.stop_requested()) {
                        fmt::print("fakecam worker is requested to stop\n");
                        return;
                    }

                    compute();
                }
            });
        }

        FakeCamera(FakeCamera&&) = default;

        ~FakeCamera() {
            if (thread.joinable())
                thread.request_stop();
        }

        void compute() {

            if (running) {

                send_frame(get_last_frame());

                index = (index + 1 ) % frame_number;
                ci->loop_count.add(1);
            }

            const auto previous = now;
            now = std::chrono::system_clock::now();
            const auto elapsed = now - previous;

            if (elapsed < latency) {
                auto remaining = latency - elapsed;
                // fmt::print("sleeping for {}ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(remaining).count());
                std::this_thread::sleep_for(latency - elapsed);
            }
        }

        void set_command(Command new_command) override  // implement interface::Camera
        {
            switch (new_command) {
                case Command::pause:
                    running = false;
                    break;
                case Command::run:
                case Command::step:
                    running = true;
                    break;
                default: break;
            };
        }

        std::span<const uint16_t> get_last_frame() const override {
            return std::span{ data }.subspan(index * frame_size, frame_size);
        }

    };

    std::future<void> make_camera(ComponentInfo& ci, CameraLogic cam_logic, json::object config, bool async) {
        size_t frame_size = json::opt_to<size_t>(config, "size").or_else([]{
            throw std::runtime_error("missing frame size");
        }).value();
        size_t frame_number = json::opt_to<size_t>(config, "number").or_else([]{
            throw std::runtime_error("missing frame number");
        }).value();

        std::chrono::microseconds latency( json::opt_to<size_t>(config, "latency").or_else([]{
            throw std::runtime_error("missing frame latency");
        }).value() );

        auto camera = std::make_unique<FakeCamera>(ci, std::move(cam_logic), frame_size, frame_number, latency);

        if (async) {
            return spawn_async_camera_runner(std::move(camera), ci);
        } else {
            return spawn_runner([camera = std::move(camera)]() mutable {
                camera->compute();

            }, ci);
        }
    }

} // namespace baldr::fakecam
