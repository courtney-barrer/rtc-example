#include <baldr/baldr.hpp>

#include <boost/json/object.hpp>
#include <boost/program_options.hpp>
#include <boost/json.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <future>

using namespace baldr;

namespace po = boost::program_options;

json::value parse_json(const std::string& input) {
    // Try to parse as inline JSON
    try {
        return json::parse(input);
    } catch (const boost::system::system_error&) {
        // If parsing fails, assume it's a file path
        std::ifstream file(input);
        if (!file) {
            throw std::runtime_error("Invalid JSON string or file path.");
        }
        std::string content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
        return json::parse(content);
    }
}

int main(int argc, char* argv[]) {
    try {
        std::string component;
        std::string json_config;

        // Define and parse the command line options
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Produce help message")
            ("config,c", po::value<std::string>(&json_config)->required(), "JSON configuration (inline or path to JSON file)");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        // Print help message
        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        // Notify if any required option is missing
        po::notify(vm);

        // Parse the JSON configuration
        json::array config = parse_json(json_config).as_array();

        std::vector<std::future<void>> locks;

        for (const auto& component_config : config) {
            auto& comp_config_obj = component_config.as_object();

            locks.push_back( baldr::init_component_thread(comp_config_obj) );
        }

        fmt::print("All component initialized\n");

        // wait until all component have exited
        for (auto& lock : locks)
            lock.wait();

    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }

}
