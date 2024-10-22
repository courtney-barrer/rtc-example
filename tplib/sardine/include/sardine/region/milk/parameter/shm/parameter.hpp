#ifndef FSP_SHM_PARAMETER_H
#define FSP_SHM_PARAMETER_H

#include <fps/detail/type.h>

namespace fps
{

namespace shm
{

    /// Provide a proxy to the different handled types.
    template<typename T> struct FpsProxy;

    template<> struct FpsProxy<i32> {

        static i32 value(handle::parameter_t& p) {
            return static_cast<i32>(p.val.i64[3] = p.val.i64[0]);
        }

        static i32& ref(handle::parameter_t& p) {
            return *reinterpret_cast<i32*>(&p.val.i64[0]);
        }
    };

    template<> struct FpsProxy<i64> {

        static i64 value(handle::parameter_t& p) {
            return p.val.i64[3] = p.val.i64[0];
        }

        static i64& ref(handle::parameter_t& p) {
            return p.val.i64[0];
        }
    };

    template<> struct FpsProxy<f32> {

        static f32 value(handle::parameter_t& p) {
            return p.val.f32[3] = p.val.f32[0];
        }

        static f32& ref(handle::parameter_t& p) {
            return p.val.f32[0];
        }
    };

    template<> struct FpsProxy<f64> {

        static f64 value(handle::parameter_t& p) {
            return p.val.f64[3] = p.val.f64[0];
        }

        static f64& ref(handle::parameter_t& p) {
            return p.val.f64[0];
        }
    };

    template<> struct FpsProxy<char*> {

        static const char* value(handle::parameter_t& p) {
            return p.val.string[0];
        }

        static char* ref(handle::parameter_t& p) {
            return p.val.string[0];
        }
    };

    template<typename T>
    auto value(handle::parameter_t& p) {
        return FpsProxy<T>::value(p);
    }

    template<>
    inline auto value<std::string>(handle::parameter_t& p) {
        return std::string{FpsProxy<char*>::value(p)};
    }

    template<typename T>
    decltype(auto) ref(handle::parameter_t& p) {
        return FpsProxy<T>::ref(p);
    }

    template<typename T>
    auto ptr(handle::parameter_t& p) {
        return &FpsProxy<T>::ref(p);
    }

    /// Special implementation for string parameters.
    template<>
    inline auto ptr<char *>(handle::parameter_t& p) {
        return FpsProxy<char *>::ref(p);
    }

    template<typename T>
    void set(handle::parameter_t& p, T value) {
        FpsProxy<T>::ref(p) = value;
        p.cnt0++;
    }

    /// Special implementation for string parameters.
    inline void set(handle::parameter_t& p, const char * value) {
        std::strncpy(FpsProxy<char *>::ref(p), value, FUNCTION_PARAMETER_STRMAXLEN);
        p.cnt0++;
    }

    /// Special implementation for string parameters.
    inline void set(handle::parameter_t& p, std::string value) {
        std::strncpy(FpsProxy<char *>::ref(p), value.c_str(), FUNCTION_PARAMETER_STRMAXLEN);
        p.cnt0++;
    }

    template<typename T>
    bool check(handle::parameter_t& p) {
        return p.type == type_of<T>;
    }

} // namespace shm

} // namespace fps

#endif //FSP_SHM_PARAMETER_H