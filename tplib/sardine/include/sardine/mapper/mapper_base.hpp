#pragma once

#include <sardine/concepts.hpp>
#include <sardine/mapper/base.hpp>

namespace sardine::buffer
{

    /**
     * mapper_base is a proxy to interact easily with mapper. It provides a common interface to select a sub region of a buffer.
     *
     * It requires some CRTP boilerplate to work properly.
     *
     * For instance, Using a class that inherits from mapper_base<span<int>, Derived> will provide the following methods:
     * - close_lead_dim() // The type returned will be another specialization of Derived but will now inherit from mapper_base<int, Derived>.
     * - subspan(std::size_t new_offset, std::size_t new_size = std::dynamic_extent)
     *
     */

    /**
     * @brief
     *
     * @tparam T
     * @tparam Derived
     */
    template <typename T, typename Derived>
    struct mapper_base : sardine::mapper<T>
    {
        using mapper_t = sardine::mapper<T>;

        mapper_base(const mapper_t & a) : mapper_t(a) {}
        mapper_base(const mapper_base &) = default;

        sardine::mapper<T>       & mapper()       { return *this; }
        sardine::mapper<T> const & mapper() const { return *this; }

    private:
        const Derived& self() const { return *static_cast<const Derived *>(this); }
    };

    template <mapper_cpts::contiguous V, typename Derived>
    struct mapper_base< V, Derived > : sardine::mapper< V >
    {
        using mapper_t = sardine::mapper< V >;

        using element_type = typename mapper_t::element_type;

        mapper_base(const mapper_t & a) : mapper_t(a) {}
        mapper_base(const mapper_base &) = default;

        mapper_t       & mapper()       { return *this; }
        mapper_t const & mapper() const { return *this; }

        auto close_lead_dim() const
        {
            using new_mapper_t = sardine::mapper<element_type>;

            // using new_type = mapper_base<element_type, Derived>;
            return self().clone_with_new_mapper(new_mapper_t(this->offset()));
        }

        auto subspan(size_t new_offset, size_t new_size = dynamic_extent) const {
            // mapper is unaware of the element type, so we need to multiply by the size of the element
            return self().clone_with_new_mapper(mapper_t::subspan(
                new_offset,
                new_size
            ));
        }

    private:
        const Derived &self() const { return *static_cast<const Derived *>(this); }
    };

    template <mapper_cpts::mapped V, typename Derived>
    struct mapper_base< V, Derived > : sardine::mapper< V >
    {
        using mapper_t = sardine::mapper< V >;

        using element_type = typename V::element_type;

        mapper_base(const mapper_t & a) : mapper_t(a) {}
        mapper_base(const mapper_base &) = default;

        mapper_t       & mapper()       { return *this; }
        mapper_t const & mapper() const { return *this; }

        template<typename = void> // lazy compilation to avoid recursive type instantiation. TODO: check if this is necessary
        auto close_lead_dim() const requires (V::rank() > 0) {
            return submdspan(0);
        }

        template<class... SliceSpecifiers>
        auto submdspan(SliceSpecifiers... specs) const {
            return self().clone_with_new_mapper(mapper_t::submdspan(specs...));
        }

    private:
        const Derived &self() const { return *static_cast<const Derived *>(this); }
    };

} // namespace sardine::buffer
