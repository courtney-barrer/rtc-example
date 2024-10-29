#include <baldr/camera.hpp>
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

    struct FakeCamera : interface::Camera
    {

        std::vector<uint16_t> data;
        size_t frame_size;
        size_t frame_number;
        size_t index;
        std::chrono::microseconds latency;
        now_type now;

        FakeCamera(size_t frame_size, size_t frame_number, std::chrono::microseconds latency)
            : data(frame_size * frame_number)
            , frame_size(frame_size)
            , frame_number(frame_number)
            , index(0)
            , latency(latency)
            , now( std::chrono::system_clock::now() )
        {
            fill_with_random_values(data);
        }


        bool get_frame(std::span<uint16_t> frame) override {
            const auto start = std::chrono::high_resolution_clock::now();

            {
                auto current = std::span{ data }.subspan(index * frame_size, frame_size);

                std::ranges::copy(current, frame.data());

                index = (index + 1 ) % frame_number;
            }

            const auto previous = now;
            now = std::chrono::system_clock::now();
            const auto elapsed = now - previous;

            if (elapsed < latency)
                std::this_thread::sleep_for(latency - elapsed);

            return true;
        }
    };

    std::unique_ptr<interface::Camera> make_camera(json::object config) {
        size_t frame_size = json::opt_to<size_t>(config, "size").or_else([]{
            throw std::runtime_error("missing frame size");
        }).value();
        size_t frame_number = json::opt_to<size_t>(config, "number").or_else([]{
            throw std::runtime_error("missing frame number");
        }).value();

        std::chrono::microseconds latency( json::opt_to<size_t>(config, "latency").or_else([]{
            throw std::runtime_error("missing frame latency");
        }).value() );

        return std::make_unique<FakeCamera>(frame_size, frame_number, latency);
    }

} // namespace baldr::fakecam
