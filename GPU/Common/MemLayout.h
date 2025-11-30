#ifndef MEMLAYOUT_H
#define MEMLAYOUT_H

#include "GPUCommonDefAPI.h"

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

// The types S<value>, S<reference>, and S<const_reference> need to be aggregate constructible
template <template <template <class> class> class S, template <class> class F, Flag L>
struct wrapper;

template <template <template <class> class> class S, template <class> class F>
struct wrapper<S, F, Flag::aos> { using type = F<S<value>>; };

template <template <template <class> class> class S, template <class> class F>
struct wrapper<S, F, Flag::soa> { using type = S<F>; };

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

template <class T>
struct always_false : false_type {};

template<class T, class U>
struct is_same : false_type {};

template<class T>
struct is_same<T, T> : true_type {};

}  // namespace type_traits

template<class T_left, class T_right>
using enable_if_equal = type_traits::enable_if_t<type_traits::is_same<T_left, T_right>::value>;

template<class T_left, class T_right>
using disable_if_equal = type_traits::enable_if_t<!type_traits::is_same<T_left, T_right>::value>;

template<class T>
using disable_if_scalar = type_traits::enable_if_t<
    !type_traits::is_same<T, value<int>>::value &&
    !type_traits::is_same<T, reference<int>>::value &&
    !type_traits::is_same<T, reference_restrict<int>>::value &&
    !type_traits::is_same<T, const_reference_restrict<int>>::value
>;

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

template <
    template <template <class> class> class S,
    template <class> class F_out,
    class... Args
>
constexpr S<F_out> eval_at(size_t i, Args& ...args) { return {(args[i])...}; }

template <
    template <template <class> class> class S,
    template <class> class F_out,
    class... Args
>
constexpr S<F_out> eval_at(size_t i, const Args& ...args) { return {(args[i])...}; }

}  // namespace MemLayout

#define MEMLAYOUT_MEMBERFUNCTIONS(STRUCT, CONTAINER, ...)                                 \
    template <template <class> class F_out>                                               \
    constexpr operator STRUCT<F_out>() { return { __VA_ARGS__ }; }               \
    template <template <class> class F_out>                                               \
    constexpr operator STRUCT<F_out>() const { return { __VA_ARGS__ }; }         \
    template<class T = int, class R = MemLayout::disable_if_scalar<CONTAINER<T>>>         \
    constexpr STRUCT<MemLayout::reference> operator[] (MemLayout::size_t i) {    \
        return MemLayout::eval_at<STRUCT, MemLayout::reference>(i, __VA_ARGS__);          \
    }                                                                                     \
    template<class T = int, class R = MemLayout::disable_if_scalar<CONTAINER<T>>>         \
    constexpr STRUCT<MemLayout::const_reference> operator[] (MemLayout::size_t i) const { \
        return MemLayout::eval_at<STRUCT, MemLayout::const_reference>(i, __VA_ARGS__);    \
    }                                                                                     \

#endif // MEMLAYOUT_H
