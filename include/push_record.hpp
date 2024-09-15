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
    //new telemetry (keep old ones above for some functions that still use them)
    std::vector<double> e_TT; // Tip Tilt mode error
    std::vector<double> e_HO; // higher order mode error
    std::vector<double> u_TT; // output of TT controller
    std::vector<double> u_HO; // output of HO controller
    std::vector<double> cmd_TT; // TT reconstruction command
    std::vector<double> cmd_HO; // HO reconstruction command
    std::vector<double> dm_disturb; // HO reconstruction command
};

void append_telemetry(telem_entry telem);
std::vector<telem_entry>& get_telemetry();

void clear_telemetry();

void appendToTable(std::span<const uint8_t> data);

void saveTableToCSV(const std::string& filename);