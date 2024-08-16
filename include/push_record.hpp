#pragma once

#include <span>
#include <vector>
#include <cstdint>
#include <string>
#include <iostream>

struct telem_entry {
    std::vector<float> image_in_pupil; //std::span<const uint16_t> image_raw;
    std::vector<float> image_err_signal; //std::vector<float> image_proc; // processed signal 
    std::vector<double> mode_err; //std::vector<double> reco_dm_err; // product CM * signal
    std::vector<double> dm_cmd_err; //std::vector<double> dm_command; // final cmd sent to DM
};

void append_telemetry(telem_entry telem);
std::vector<telem_entry>& get_telemetry();

void clear_telemetry();

void appendToTable(std::span<const uint8_t> data);

void saveTableToCSV(const std::string& filename);