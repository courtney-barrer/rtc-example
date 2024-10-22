#include "emu/container.hpp"
#include "emu/detail/mdspan_types.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <config.hpp>

#include <sardine/mapper.hpp>

using namespace sardine;

/**
 * Mapper blablabla
 *
 * mapper support:
 * - scalar/pod type: a single value passed by reference
 * - contiguous [owning] range: span, string_view, vector, string, emu::container.
 *   - Should be possible to specify sub range.
 *   - When constructing vector or string, create new buffer
 * - md span/container
 * - should be possible to pass from one to another as long the destination mapping can model the source
 *   mapping (i.e. scalar -> range: Ok, contiguous mdspan 1d to span: Ok, strided mdspan to span: Ko)
 *
 */

namespace
{

    TEST(Mapping, ScalarTest)
    {
        using type = int;

        int value = 42;

        // Create the mapper fron value
        auto mapper = sardine::mapper_from(value);

        // The mapper should reflect value properties
        ASSERT_EQ(mapper.offset(), 0);
        ASSERT_EQ(mapper.size(), 1);

        // Get bytes that points to value data.
        auto bytes = sardine::as_bytes(value);

        ASSERT_EQ(bytes.data(), reinterpret_cast<std::byte*>(&value));
        ASSERT_EQ(bytes.size(), sizeof(int));

        // Get the value back from bytes
        int& view = mapper.convert(bytes);

        ASSERT_EQ(&view, &value);

        // Generate mapping from mapper.
        auto mapping = mapper.mapping_descriptor();

        // The mapping should reflect value properties
        ASSERT_FALSE(mapping.is_strided());
        ASSERT_FALSE(mapping.is_const());

        ASSERT_EQ(mapping.data_type(), emu::dlpack::data_type_ext<int>);

        // Create a new mapper from the mapping.
        auto new_mapper = sardine::mapper_from_mapping_descriptor<type>(mapping);

        ASSERT_TRUE(new_mapper) << "Could not generate mapper from descriptor because:" << new_mapper.error().message();

        // Get the value back again from bytes
        int& new_view = new_mapper->convert(bytes);

        ASSERT_EQ(&new_view, &value);
    }

    TEST(Mapping, Span)
    {
        std::vector<int> vec{1,2,3,4,5};
        std::span span{vec};

        auto mapper = sardine::mapper_from(span);

        ASSERT_EQ(mapper.offset(), 0);
        ASSERT_EQ(mapper.size(), span.size());
        ASSERT_EQ(mapper.lead_stride(), 1) << "distance between 2 elements should be 1 with contiguous range";

        // Get bytes that points to value data.
        auto bytes = sardine::as_bytes(span);

    }

    TEST(Mapping, SubSpan)
    {
        std::vector<int> vec{1,2,3,4,5};
        std::span span{vec};

        auto mapper = sardine::mapper_from(span);

        // Get bytes that points to value data.
        auto bytes = sardine::as_bytes(span);

        auto sub_mapper = mapper.subspan(1, 3); // start at 1 and take 3 elements

        std::span sub_span = sub_mapper.convert(bytes);

        ASSERT_EQ(sub_span.size(), 3);
        ASSERT_EQ(sub_span.data(), span.data() + 1);

        auto mapping = sub_mapper.mapping_descriptor();

        ASSERT_FALSE(mapping.is_strided());
        ASSERT_FALSE(mapping.is_const());

        ASSERT_EQ(mapping.data_type(), emu::dlpack::data_type_ext<int>);

        // Create a new mapper from the mapping.
        auto new_mapper = sardine::mapper_from_mapping_descriptor<std::span<int>>(mapping);

        ASSERT_TRUE(new_mapper) << "Could not generate mapper from descriptor because:" << new_mapper.error().message();

        std::span new_sub_span = new_mapper->convert(bytes);

        ASSERT_EQ(new_sub_span.size(), 3);
        ASSERT_EQ(new_sub_span.data(), span.data() + 1);

    }

    TEST(Mapping, Vector)
    {
        std::vector<int> vec{1,2,3,4,5};

        auto mapper = sardine::mapper_from(vec);

        ASSERT_EQ(mapper.offset(), 0);
        ASSERT_EQ(mapper.size(), vec.size());
        ASSERT_EQ(mapper.lead_stride(), 1);

        auto bytes = sardine::as_bytes(vec);

        ASSERT_EQ(bytes.data(), reinterpret_cast<std::byte*>(vec.data()));

        std::vector new_vec = mapper.convert(bytes);

        ASSERT_EQ(vec[0], new_vec[0]);
    }

    TEST(Mapping, VectorToSpan)
    {
        std::vector<int> vec{1,2,3,4,5};

        auto mapper = sardine::mapper_from(vec);

        ASSERT_EQ(mapper.offset(), 0);
        ASSERT_EQ(mapper.size(), vec.size());
        ASSERT_EQ(mapper.lead_stride(), 1);

        auto bytes = sardine::as_bytes(vec);

        auto mapping = mapper.mapping_descriptor();

        // Destination only require read only buffer here.
        auto span_mapper = sardine::mapper_from_mapping_descriptor<std::span<const int>>(mapping);

        ASSERT_TRUE(span_mapper) << "Could not generate mapper from descriptor because:" << span_mapper.error().message();

        std::span new_span = span_mapper->convert(bytes);

        ASSERT_EQ(vec.data(), new_span.data());
    }

    TEST(Mapping, StringView)
    {
        std::string_view sv = "Hi mom!";

        auto mapper = sardine::mapper_from(sv);

        ASSERT_EQ(mapper.offset(), 0);
        ASSERT_EQ(mapper.size(), sv.size());
        ASSERT_EQ(mapper.lead_stride(), 1);
    }

    TEST(Mapping, String)
    {
        std::string s = "Hi mom!";

        auto mapper = sardine::mapper_from(s);

        ASSERT_EQ(mapper.offset(), 0);
        ASSERT_EQ(mapper.size(), s.size());
        ASSERT_EQ(mapper.lead_stride(), 1);
    }

    TEST(Mapping, Container)
    {
        auto con = emu::make_container<int>(42);

        auto mapper = sardine::mapper_from(con);

        ASSERT_EQ(mapper.offset(), 0);
        ASSERT_EQ(mapper.size(), con.size());
        ASSERT_EQ(mapper.lead_stride(), 1);
    }

    TEST(Mapping, MdSpan)
    {
        std::array arr{1,2,3,4,5,6,7, 8};
        emu::mdspan md{arr.data(), 2, 4};

        auto mapper = sardine::mapper_from(md);

        ASSERT_EQ(mapper.offset(), 0);
        ASSERT_EQ(mapper.size(), arr.size());
        ASSERT_EQ(mapper.lead_stride(), 4);
    }


    TEST(Mapping, MdSpanStrided)
    {
        std::array arr{1,2,3,4,5,6,7, 8};
        emu::mdspan md{arr.data(), 4, 2};

        auto mapper = sardine::mapper_from(md);

        ASSERT_EQ(mapper.offset(), 0);
        ASSERT_EQ(mapper.size(), arr.size());
        ASSERT_EQ(mapper.lead_stride(), 2);

        auto bytes = sardine::as_bytes(md);

        auto strided_mapper = mapper.submdspan(std::pair{1, 3}, 1);

        emu::mdspan strided_md = strided_mapper.convert(bytes);

        ASSERT_EQ(strided_md(0), md(1,1));
        ASSERT_EQ(strided_md(1), md(2,1));

    }


    // TEST(Mapping, MdViewClosingDimension)
    // {
    //     std::vector<int> vec{1,2,3,4,5};
    //     emu::mdspan_1d<int> spa{vec.data(), 5};

    //     using view_1d_t = sardine::view_t<emu::mdspan_1d<int>>;
    //     using view_0d_t = sardine::view_t<emu::mdspan_0d<int>>;

    //     view_1d_t view(spa);

    //     ASSERT_EQ(view.view().data_handle(), spa.data_handle());

    //     {
    //         view_0d_t sub_view = view.close_lead_dim();

    //         ASSERT_EQ(view.view()(0), sub_view.view()());
    //     }

    //     {
    //         // takes elemets from id 1 to id 4.
    //         view_t sub_view = view.submdspan(std::pair{1, 4});

    //         ASSERT_EQ(sub_view.view().rank(), 1);
    //         ASSERT_EQ(sub_view.view().size(), 3);
    //         ASSERT_EQ(sub_view.view()(0), view.view()(1));
    //     }
    // }

} // namespace
