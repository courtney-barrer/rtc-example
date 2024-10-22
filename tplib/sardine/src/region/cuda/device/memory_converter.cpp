#include "emu/error.hpp"
#include <sardine/type.hpp>
#include <sardine/utility.hpp>
#include <sardine/memory_converter.hpp>

#include <emu/cuda.hpp>
#include <emu/assert.hpp>
#include <emu/cuda/error.hpp>

#include <unordered_map>

namespace cuda::memory
{
    auto format_as(cuda::memory::type_t type) -> std::string_view
    {
        switch (type)
        {
        case cuda::memory::type_t::host_:
            return "host";
        case cuda::memory::type_t::device_:
            return "device";
        case cuda::memory::type_t::array:
            return "array";
        case cuda::memory::type_t::unified_:
            return "unified";
        case cuda::memory::type_t::non_cuda:
            return "non-cuda";
        default:
            return "unknown";
        }
    }
}


namespace sardine::region::cuda::device
{

    struct span_equal{
        constexpr bool operator()(span_b lhs, span_b rhs) const noexcept {
            return lhs.data() == rhs.data() and lhs.size() == rhs.size();
        }
    };

    // key: source region, value: device region
    using revert_converter_reference = std::unordered_map<span_b, span_b, std::hash<span_b>, span_equal>;

    revert_converter_reference& get_revert_converter_reference() {
        static revert_converter_reference instance;
        return instance;
    }

    inline span_b equivalent_subregion(span_cb input_region, span_cb input_subregion, span_b output_region) {

        EMU_ASSERT_MSG(input_region.size() == output_region.size(), "input_region and output_region must have the same size");

        // Calculate the offset of input_subregion within input_region
        std::ptrdiff_t offset = input_subregion.data() - input_region.data();

        EMU_ASSERT_MSG(offset >= 0 && offset + input_subregion.size() <= output_region.size(), "offset and size must fit within output_region");

        // Return the corresponding span in output_region with the same offset and size
        return output_region.subspan(offset, input_subregion.size());
    }

    inline bool is_contained(span_cb region, span_cb data) {
        // Check if data is entirely within the region
        return (data.data() >= region.data()) &&
            (data.data() + data.size() <= region.data() + region.size());
    }

    optional<result<span_b>> converter(bytes_and_device bad, emu::dlpack::device_type_t dt) {
        auto& rcr = get_revert_converter_reference();

        // Check if we already stored the result.
        if (auto it = rcr.find(bad.region); it != rcr.end())
            return it->second;

        auto* v_ptr = v_ptr_of(bad.region);

        // otherwise, check what kind of memory it is.
        auto type = ::cuda::memory::type_of(v_ptr);

        // If non cuda, register it.
        if (type = ::cuda::memory::type_t::non_cuda)
            EMU_CUDA_CHECK_RETURN_UN_EC(cudaHostRegister(v_ptr, bad.region.size(), /* flags = */ 0));

        auto pointer_desc = ::cuda::memory::pointer::wrap(v_ptr);

        span_b device_region(static_cast<byte*>(pointer_desc.get_for_device()), bad.region.size());

        EMU_TRUE_OR_RETURN_UN_EC(device_region.data() != nullptr, error::location_could_get_device_pointer);

        rcr.emplace(bad.region, device_region);

        // Returns the equivalent device subspan of the input data region.
        return equivalent_subregion(bad.region, bad.data, device_region);
    }



    optional<span_b> revert_converter(span_cb maybe_device_data) {
        auto& rcr = get_revert_converter_reference();

        for (auto e : rcr) {
            auto [source_region, device_region] = e;
            if (is_contained(device_region, maybe_device_data))
                return equivalent_subregion(device_region, maybe_device_data, source_region);
        }

        return nullopt;

    }

} // namespace sardine::region::cuda::device

SARDINE_REGISTER_DEVICE_FINDER(sardine_cuda_converter,
    sardine::region::cuda::device::converter,
    sardine::region::cuda::device::revert_converter
)
