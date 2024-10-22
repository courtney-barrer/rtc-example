#include <baldr/component/dm/fakedm.hpp>

#include <fmt/core.h>
#include <stdexcept>

namespace baldr::fakedm
{

    struct FakeDM : interface::DM
    {
        size_t index = 0;

        FakeDM() {

        }

        void send_command(span<const double> commands) override {
            fmt::print("received commands: {}\n", index++);
        }
    };

    std::unique_ptr<interface::DM> make_dm(json::object config) {
        return std::make_unique<FakeDM>();
    }

} // namespace baldr::fakedm
