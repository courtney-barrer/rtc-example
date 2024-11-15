#pragma once

// forward declare the boost namespace and avoid to have to include anything.
namespace EMU_BOOST_NAMESPACE {}

namespace sardine
{
    // local alias of boost namespace.
    // Allow to use boost with custom namespace name to avoid conflict with other libraries
    namespace boost = ::EMU_BOOST_NAMESPACE;

} // namespace sardine
