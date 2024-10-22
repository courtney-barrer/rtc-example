#include <baldr/component/dm/bmc.hpp>

#include <fmt/core.h>
#include <stdexcept>

#ifdef BALDR_BMC

#include <BMCApi.h>

namespace baldr::bmc
{
    using dm_handle = ::DM;

    struct DM : interface::DM
    {
        dm_handle hdm;
        vector<uint32_t> map_lut;

        DM() {
            if (auto rv = BMCOpen(&hdm, dm_serial_number.c_str()); rv != NO_ERR)
                throw std::runtime_error(fmt::format("Error {} opening the driver type {}: {}\n", rv, hdm.Driver_Type, BMCErrorString(rv)));

            // init map lut to zeros (copying examples from BMC)
            map_lut.resize(MAX_DM_SIZE, 0);
            // then we load the default map
            rv = BMCLoadMap(&hdm, NULL, map_lut.data());
        }

        void send_command(span<const double> commands) override {
            BMCSetArray(&hdm, const_cast<double*>(commands.data()), map_lut.data());
        }
    };

    std::unique_ptr<interface::DM> make_dm(json::object config) {
        return std::make_unique<DM>();
    }

} // namespace baldr::bmc

#else

namespace baldr::bmc
{

    std::unique_ptr<interface::DM> make_dm(json::object config) {
        throw std::runtime_error("Error bmc dm support is not compiled");
    }

} // namespace baldr::bmc

#endif
