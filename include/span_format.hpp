#pragma once

#include <ostream>
#include <span>
#include <typeinfo>

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::span<T>& s)
{
    os << "span<" << typeid(T).name() << ">(" << (uintptr_t)s.data() << ", " << s.size() << ")";
    return os;
}
