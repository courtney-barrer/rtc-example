#include <sardine/memory_converter.hpp>

#include <list>

namespace sardine
{
    using converter_list = std::list<converter_t>;
    using revert_converter_list = std::list<revert_converter_t>;

    converter_list& instance() {
        static converter_list list;
        return list;
    }

    revert_converter_list& revert_instance() {
        static revert_converter_list list;
        return list;
    }

    void register_converter( converter_t converter, revert_converter_t revert_converter ) {
        instance().push_back(std::move(converter));
        revert_instance().push_back(std::move(revert_converter));
    }

    result<span_b> convert_bytes( bytes_and_device input, emu::dlpack::device_type_t requested_device_type )
    {
        for (auto& converter : instance())
            // converter return optional. nullopt expresses it did nothing so we try with the next one.
            EMU_UNWRAP_RETURN_IF_TRUE(converter(input, requested_device_type));

        return make_unexpected(error::location_conversion_not_handle);
    }

    optional<span_b> revert_convert_bytes( span_cb input )
    {
        for (auto& r_converter : revert_instance())
            EMU_UNWRAP_RETURN_IF_TRUE(r_converter(input));

        return nullopt;
    }


} // namespace sardine
