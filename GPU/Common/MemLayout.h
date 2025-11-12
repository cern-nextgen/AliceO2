#ifndef MEMLAYOUT_H
#define MEMLAYOUT_H

#include "GPUCommonDefAPI.h"

#define EXPAND(...) __VA_ARGS__
#define MEMLAYOUT_MEMBERFUNCTIONS(STRUCT_NAME, ...)                                                             \
    template <template <class> class F_out>                                                                     \
    constexpr operator STRUCT_NAME<F_out>() { return { EXPAND(__VA_ARGS__) }; }                                 \
    template <template <class> class F_out>                                                                     \
    constexpr operator STRUCT_NAME<F_out>() const { return { EXPAND(__VA_ARGS__) }; }                           \
    template <template <class> class F_out, class FunctionObject>                                               \
    constexpr STRUCT_NAME<F_out> invoke_on_members(FunctionObject f) { return {f(EXPAND(__VA_ARGS__))}; }       \
    template <template <class> class F_out, class FunctionObject>                                               \
    constexpr STRUCT_NAME<F_out> invoke_on_members(FunctionObject f) const { return {f(EXPAND(__VA_ARGS__))}; } \

namespace MemLayout {

using size_t = decltype(sizeof 0);

template <class T> using value = T;

template <class T> using reference = T&;
template <class T> using reference_restrict = T& GPUrestrict();

template <class T> using const_reference = const T&;
template <class T> using const_reference_restrict = const T& GPUrestrict();

template <class T> using pointer = T*;
template <class T> using pointer_restrtict = T* GPUrestrict();

template <class T> using const_pointer = const T*;
template <class T> using const_pointer_restrict = const T* GPUrestrict();

enum Flag { soa, aos };

template <template <template <class> class> class S, template <class> class F, Flag L>
struct wrapper;

template <template <template <class> class> class S, template <class> class F>
struct wrapper<S, F, Flag::aos> : public F<S<value>> {
    using Base = F<S<value>>;

    template <template <class> class F_out>
    constexpr operator wrapper<S, F_out, Flag::aos>() { return {*static_cast<Base*>(this)}; };

    template <template <class> class F_out>
    constexpr operator wrapper<S, F_out, Flag::aos>() const { return {*static_cast<const Base*>(this)}; };

    constexpr S<reference> operator[](size_t i) {
        return static_cast<Base*>(this)->operator[](i);
    }

    constexpr S<const_reference> operator[](size_t i) const {
        return static_cast<const Base*>(this)->operator[](i);
    }
};

template <template <template <class> class> class S, template <class> class F>
using AoS = wrapper<S, F, Flag::aos>;

// The types S<value>, S<reference>, and S<const_reference> need to be aggregate constructible
template <template <template <class> class> class S, template <class> class F>
struct wrapper<S, F, Flag::soa> : public S<F> {
    template <template <class> class F_out>
    constexpr operator wrapper<S, F_out, Flag::soa>() { return {*this}; };

    template <template <class> class F_out>
    constexpr operator wrapper<S, F_out, Flag::soa>() const { return {*this}; };

    constexpr S<reference> operator[](size_t i) {
        return this->template invoke_on_members<reference>(memberwise<reference, evaluate_at>{{i}});
    }

    constexpr S<const_reference> operator[](size_t i) const {
        return this->template invoke_on_members<reference>(memberwise<const_reference, evaluate_at>{{i}});
    }

  private:

    struct evaluate_at {
        size_t i;
        template <template <class> class F_in, class T>
        constexpr reference<T> operator()(F_in<T> & t) const { return t[i]; }
        template <template <class> class F_in, class T>
        constexpr const_reference<T> operator()(const F_in<T> & t) const { return t[i]; }
    };

    template <template <class> class F_out, class FunctionObject>
    struct memberwise {
        FunctionObject f;
        template <class... Args>  // HACK: NVCC cannot deduce template parameters of f.operator() like so: { f(args)... }
        constexpr S<F_out> operator()(Args&... args) const { return {f.template operator()<F>(args)...}; }
    };
};

template <template <template <class> class> class S, template <class> class F>
using SoA = wrapper<S, F, Flag::soa>;

namespace type_traits {

template<bool B, class T = void>
struct enable_if {};
 
template<class T>
struct enable_if<true, T> { typedef T type; };

template< bool B, class T = void >
using enable_if_t = typename enable_if<B, T>::type;

struct false_type {
    static constexpr bool value = false;
    constexpr operator bool() const noexcept { return value; }
};

struct true_type {
    static constexpr bool value = true;
    constexpr operator bool() const noexcept { return value; }
};

template<class T, class U>
struct is_same : false_type {};

template<class T>
struct is_same<T, T> : true_type {};

}  // namespace type_traits

template<class T_left, class T_right>
using enable_if_equal = type_traits::enable_if_t<type_traits::is_same<T_left, T_right>::value>;

template<class T_left, class T_right>
using disable_if_equal = type_traits::enable_if_t<!type_traits::is_same<T_left, T_right>::value>;

#if __cplusplus >= 202002L
template<template <class> class F_left, template <class> class F_right>
concept is_same = type_traits::is_same<F_left<int>, F_right<int>>::value;
template<template <class> class F>
concept is_value = is_same<F, value>;
template<template <class> class F>
concept is_reference = is_same<F, reference>;
template<template <class> class F>
concept is_const_reference = is_same<F, const_reference>;
#endif

}  // namespace MemLayout

#endif // MEMLAYOUT_H
