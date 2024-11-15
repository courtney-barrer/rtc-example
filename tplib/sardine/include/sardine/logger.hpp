// If the user wants to change the logger,
// they can define `EMU_LOGGER` here before the first include.

#include <fmt/color.h>

#define EMU_LOGGER(...)                                                            \
    ::fmt::print(stderr, fmt::fg(fmt::color::red), "{}:{}: ", __FILE__, __LINE__); \
    ::fmt::print(stderr, fmt::fg(fmt::color::red), __VA_ARGS__);                   \
    ::fmt::print(stderr, "\n");

#include <emu/macro.hpp>
