#include <vector>
#include <span>
#include <fstream>
#include <iostream>
#include <cstdint>

#include <push_record.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>

namespace nb = nanobind;

std::vector<telem_entry> global_telemetry;

void append_telemetry(telem_entry telem) {
    global_telemetry.push_back(std::move(telem));
}

std::vector<telem_entry>& get_telemetry() {
    return global_telemetry;
}

void clear_telemetry() {
    global_telemetry.clear();
}

void bind_telemetry(nb::module_& m) {
    nb::bind_vector<std::vector<telem_entry>>(m, "VecEntry");

    m.def("get_telemetry", &get_telemetry);
    m.def("clear_telemetry", &clear_telemetry);
}



// Global table to store the data
std::vector<std::vector<float>> global_table;

// Function to append data from std::span<uint8_t> to the global table
void appendToTable(std::span<const uint8_t> data) {
    std::vector<float> row;
    row.reserve(data.size());

    for (uint8_t value : data) {
        row.push_back(static_cast<float>(value));
    }

    global_table.push_back(std::move(row));
}

// Function to save the global table to a CSV file
void saveTableToCSV(const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (const auto& row : global_table) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
    if (file.fail()) {
        std::cerr << "Error writing to file: " << filename << std::endl;
    }
}

int main() {
    // Example usage
    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    std::span<uint8_t> data_span(data);

    // Append data to the global table
    appendToTable(data_span);

    // Save the global table to a CSV file
    saveTableToCSV("output.csv");

    return 0;
}