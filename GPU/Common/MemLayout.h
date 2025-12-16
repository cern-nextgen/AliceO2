#ifndef MEMLAYOUT_H
#define MEMLAYOUT_H

#include "GPUCommonDefAPI.h"

namespace MemLayout {

template <class T> using value = T;

template <class T> using reference = T&;
template <class T> using reference_restrict = T& GPUrestrict();

template <class T> using const_reference = const T&;
template <class T> using const_reference_restrict = const T& GPUrestrict();

template <class T> using pointer = T*;
template <class T> using pointer_restrtict = T* GPUrestrict();

template <class T> using const_pointer = const T*;
template <class T> using const_pointer_restrict = const T* GPUrestrict();

using size_t = decltype(sizeof 0);
using ptrdiff_t = decltype(static_cast<int*>(nullptr) - static_cast<int*>(nullptr));

enum Flag { soa, aos };

// The types S<value>, S<reference>, and S<const_reference> need to be aggregate constructible
template <template <template <class> class> class S, template <class> class F, Flag L>
struct wrapper;

template <template <template <class> class> class S, template <class> class F>
struct wrapper<S, F, Flag::aos> { using type = F<S<value>>; };

template <template <template <class> class> class S, template <class> class F>
struct wrapper<S, F, Flag::soa> { using type = S<F>; };

namespace type_traits {

template<class T> struct remove_reference { using type = T; };
template<class T> struct remove_reference<T&> { using type = T; };
template<class T> struct remove_reference<T&&> { using type = T; };

}  // namespace type_traits

template< class T >
constexpr type_traits::remove_reference<T>::type&& move(T&& t) noexcept {
    return static_cast<typename type_traits::remove_reference<T>::type&&>(t);
}

template <class SF>
struct RandomAccessAt {
    size_t i;
    template <class... Args>
    constexpr SF operator()(Args& ...args) const { return {{}, args[i]...}; }
};

template <class SF>
struct GetPointer {
    template <class... Args>
    constexpr SF operator()(Args& ...args) const { return {{}, &args...}; }
};

template <class SF>
struct AggregateConstructor {
    template <class... Args>
    constexpr SF operator()(Args& ...args) const { return {{}, args...}; } 
};

template <
    template <class> class F_left,
    template <class> class F_right
>
struct CopyAssignment {
    template <class T>
    constexpr void operator()(F_left<T>& left, F_right<T>& right) const { left = right; }
};

template <
    template <class> class F_left,
    template <class> class F_right
>
struct MoveAssignment {
    template <class T>
    constexpr void operator()(F_left<T>& left, F_right<T>& right) const { left = move(right); }
};

template <
    template <template <class> class> class S,
    template <class> class F
>
struct CRTP {
    using Derived = S<F>;
    template <template <class> class F_out>
    constexpr operator S<F_out>() {
        return static_cast<Derived*>(this)->apply(AggregateConstructor<S<F_out>>{});
    }
    template <template <class> class F_out>
    constexpr operator S<F_out>() const {
        return static_cast<const Derived*>(this)->apply(AggregateConstructor<S<F_out>>{});
    }
    constexpr S<reference> operator[] (size_t i) {
        return static_cast<Derived*>(this)->apply(RandomAccessAt<S<reference>>{i});
    }
    constexpr S<const_reference> operator[] (size_t i) const {
        return static_cast<const Derived*>(this)->apply(RandomAccessAt<S<const_reference>>{i});
    }
    constexpr S<pointer> operator& () {
        return static_cast<Derived*>(this)->apply(GetPointer<S<pointer>>{});
    }
    constexpr S<const_pointer> operator& () const {
        return static_cast<const Derived*>(this)->apply(GetPointer<S<const_pointer>>{});
    }
    constexpr S<reference> operator*() {
        return static_cast<Derived*>(this)->operator[](0);
    }
    constexpr S<const_reference> operator*() const {
        return static_cast<const Derived*>(this)->operator[](0);
    }
};

template <template <template <class> class> class S>
struct CRTP<S, value> {
    using Derived = S<value>;
    template <template <class> class F_out>
    constexpr operator S<F_out>() {
        return static_cast<Derived*>(this)->apply(AggregateConstructor<S<F_out>>{});
    }
    template <template <class> class F_out>
    constexpr operator S<F_out>() const {
        return static_cast<const Derived*>(this)->apply(AggregateConstructor<S<F_out>>{});
    }
};

template <template <template <class> class> class S>
struct CRTP<S, reference> {
    using Derived = S<reference>;
    template <template <class> class F_out>
    constexpr operator S<F_out>() {
        return static_cast<Derived*>(this)->apply(AggregateConstructor<S<F_out>>{});
    }
    template <template <class> class F_out>
    constexpr operator S<F_out>() const {
        return static_cast<const Derived*>(this)->apply(AggregateConstructor<S<F_out>>{});
    }
    template <template <class> class F_other>
    constexpr Derived& operator=(S<F_other>& other) {
        memberwise(*static_cast<Derived*>(this), other, CopyAssignment<reference, F_other>{});
        return *static_cast<Derived*>(this);
    }
    template <template <class> class F_other>
    constexpr Derived& operator=(S<F_other>&& other) {
        memberwise(*static_cast<Derived*>(this), other, MoveAssignment<reference, F_other>{});
        return *static_cast<Derived*>(this);
    }
    constexpr S<pointer> operator& () {
        return static_cast<Derived*>(this)->apply(GetPointer<S<pointer>>{});
    }
    constexpr S<const_pointer> operator& () const {
        return static_cast<const Derived*>(this)->apply(GetPointer<S<const_pointer>>{});
    }
};

template <template <template <class> class> class S>
struct CRTP<S, const_reference> {
    using Derived = S<const_reference>;
    template <template <class> class F_out>
    constexpr operator S<F_out>() const {
        return static_cast<const Derived*>(this)->apply(AggregateConstructor<S<F_out>>{});
    }
    constexpr S<const_pointer> operator& () const {
        return static_cast<const Derived*>(this)->apply(GetPointer<S<const_pointer>>{});
    }
};

template <template <template <class> class> class S>
struct CRTP<S, reference_restrict> {
    using Derived = S<reference_restrict>;
    template <template <class> class F_out>
    constexpr operator S<F_out>() {
        return static_cast<Derived*>(this)->apply(AggregateConstructor<S<F_out>>{});
    }
    template <template <class> class F_out>
    constexpr operator S<F_out>() const {
        return static_cast<const Derived*>(this)->apply(AggregateConstructor<S<F_out>>{});
    }
    template <template <class> class F_other>
    constexpr Derived& operator=(S<F_other>& other) {
        memberwise(*static_cast<Derived*>(this), other, CopyAssignment<reference_restrict, F_other>{});
        return *static_cast<Derived*>(this);
    }
    template <template <class> class F_other>
    constexpr Derived& operator=(S<F_other>&& other) {
        memberwise(*static_cast<Derived*>(this), other, MoveAssignment<reference_restrict, F_other>{});
        return *static_cast<Derived*>(this);
    }
    constexpr S<pointer_restrtict> operator& () {
        return static_cast<Derived*>(this)->apply(GetPointer<S<pointer_restrtict>>{});
    }
    constexpr S<const_pointer_restrict> operator& () const {
        return static_cast<const Derived*>(this)->apply(GetPointer<S<const_pointer_restrict>>{});
    }
};

template <template <template <class> class> class S>
struct CRTP<S, const_reference_restrict> {
    using Derived = S<const_reference_restrict>;
    template <template <class> class F_out>
    constexpr operator S<F_out>() const {
        return static_cast<const Derived*>(this)->apply(AggregateConstructor<S<F_out>>{});
    }
    constexpr S<const_pointer_restrict> operator& () const {
        return static_cast<const Derived*>(this)->apply(GetPointer<S<const_pointer_restrict>>{});
    }
};

template <
    template <template <class> class> class S,
    template <class> class F
>
struct iterator {
    //using iterator_category = std::random_access_iterator_tag;
    using difference_type = ptrdiff_t;
    using value_type = S<value>;
    using pointer = S<F>;
    using reference = S<reference>;

    difference_type index;
    pointer handle;

    constexpr bool operator==(iterator const& other) const { return index == other.index; }
    constexpr bool operator!=(iterator const& other) const { return index != other.index; }
    constexpr bool operator<(iterator const& other) const { return index < other.index; }

    constexpr iterator operator+(difference_type i) const { return {index + i, handle}; }
    constexpr iterator operator-(difference_type i) const { return {index - i, handle}; }
    
    constexpr difference_type operator-(iterator const& other) const { 
        return difference_type(index) - difference_type(other.index); 
    }
    
    constexpr iterator& operator++() { ++index; return *this; }
    constexpr iterator& operator--() { --index; return *this; }

    constexpr reference operator*() { return handle[index]; }
};

}  // namespace MemLayout

#define MEMLAYOUT_MEMBERFUNCTIONS(STRUCT, CONTAINER, ...)\
    template <class Function>\
    constexpr auto apply(Function&& f) { return f(__VA_ARGS__); }\
    template <class Function>\
    constexpr auto apply(Function&& f) const { return f(__VA_ARGS__); }\
    template <template <class> class F_left, template <class> class F_right, class FunctionObject>\
    constexpr friend void memberwise(STRUCT<F_left>& left, STRUCT<F_right>& right, FunctionObject&& f);\

#endif // MEMLAYOUT_H
