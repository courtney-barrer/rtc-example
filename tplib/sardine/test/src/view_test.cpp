#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <config.hpp>

#include <sardine/sardine.hpp>

using namespace sardine;

namespace
{

    TEST(View, ClosingDimension)
    {
        std::vector<int> vec{1,2,3,4,5};
        std::span spa{vec};

        using view_t = sardine::view_t<std::span<int>>;

        view_t view(spa);

        ASSERT_EQ(view.view().data(), spa.data());

        {
            sardine::view_t<int> sub_view = view.close_lead_dim();

            ASSERT_EQ(view.view()[0], sub_view.view());
        }

        {
            // takes 3 elements from 1.
            view_t sub_view = view.subspan(1, 3);

            ASSERT_EQ(sub_view.view().size(), 3);
            ASSERT_EQ(sub_view.view()[0], view.view()[1]);
        }
    }

    TEST(View, MdViewClosingDimension)
    {
        std::vector<int> vec{1,2,3,4,5};
        emu::mdspan_1d<int> spa{vec.data(), 5};

        using view_1d_t = sardine::view_t<emu::mdspan_1d<int>>;
        using view_0d_t = sardine::view_t<emu::mdspan_0d<int>>;

        view_1d_t view(spa);

        ASSERT_EQ(view.view().data_handle(), spa.data_handle());

        {
            view_0d_t sub_view = view.close_lead_dim();

            ASSERT_EQ(view.view()(0), sub_view.view()());
        }

        {
            // takes elemets from id 1 to id 4.
            view_t sub_view = view.submdspan(std::pair{1, 4});

            ASSERT_EQ(sub_view.view().rank(), 1);
            ASSERT_EQ(sub_view.view().size(), 3);
            ASSERT_EQ(sub_view.view()(0), view.view()(1));
        }
    }

} // namespace
