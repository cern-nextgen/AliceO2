#ifndef MEMLAYOUT_H
#define MEMLAYOUT_H

namespace MemLayout {

using size_t = decltype(sizeof(0));
using ptrdiff_t = decltype(static_cast<int*>(nullptr) - static_cast<int*>(nullptr));

template <class T> using value = T;
template <class T> using reference = T&;
template <class T> using const_reference = const T&;
template <class T> using pointer = T*;
template <class T> using const_pointer = const T*;

template <class T> using reference_restrict = T& __restrict__;
template <class T> using const_reference_restrict = const T& __restrict__;
template <class T> using pointer_restrict = T* __restrict__;
template <class T> using const_pointer_restrict = const T* __restrict__;

template <class SF>
struct RandomAccessAt {
    MemLayout::size_t i;
    template <class... Args>
    constexpr SF operator()(Args& ...args) const { return {args[i]...}; }
    template <class... Args>
    constexpr SF operator()(const Args& ...args) const { return {args[i]...}; }
};

template <class SF>
struct GetPointer {
    template <class... Args>
    constexpr SF operator()(Args& ...args) const { return {&args...}; }
    template <class... Args>
    constexpr SF operator()(const Args& ...args) const { return {&args...}; }
};

template <class SF>
struct AggregateConstructor {
    template <class... Args>
    constexpr SF operator()(Args& ...args) const { return {args...}; }
    template <class... Args>
    constexpr SF operator()(const Args& ...args) const { return {args...}; }
};

struct FirstMember {
    template <class T, class... Args>
    constexpr T& operator()(T& t, Args& ...args) const { return t; }
    template <class T, class... Args>
    constexpr const T& operator()(const T& t, const Args& ...args) const { return t; }
};

template <class SF>
struct PreIncrement {
    template <class... Args>
    constexpr SF operator()(Args& ...args) const { return {++args...}; }
};

template <class SF>
struct PreDecrement {
    template <class... Args>
    constexpr SF operator()(Args& ...args) const { return {--args...}; }
};

template <class SF>
struct Advance {
    ptrdiff_t i;
    template <class... Args>
    constexpr SF operator()(const Args& ...args) const { return {(args + i)...}; }
};

struct CopyAssignment {
    template <class Left, class Right>
    constexpr Left& operator()(Left& left, const Right& right) const { return left = right; }
};

template <
    template <template <class> class> class S,
    template <class> class F
>
struct wrapper : public S<F> {
    using Base = S<F>;

    constexpr wrapper() = default;
    constexpr wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    template <template <class> class F_other>
    constexpr wrapper(S<F_other>& other) : Base{other.apply(AggregateConstructor<Base>{})} {}
    template <template <class> class F_other>
    constexpr wrapper(const S<F_other>& other) : Base{other.apply(AggregateConstructor<Base>{})} {}

    constexpr wrapper<S, reference> operator[] (size_t i) { return Base::apply(RandomAccessAt<S<reference>>{i}); }
    constexpr wrapper<S, const_reference> operator[] (size_t i) const { return Base::apply(RandomAccessAt<S<const_reference>>{i}); }

    constexpr wrapper<S, reference> operator*() { return operator[](0); }
    constexpr wrapper<S, const_reference> operator*(ptrdiff_t) const { return operator[](0); }
};

template <template <template <class> class> class S>
struct wrapper<S, value> : public S<value> {
    using Base = S<value>;

    constexpr wrapper() = default;
    constexpr wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    constexpr wrapper(const S<reference>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}
    constexpr wrapper(const S<const_reference>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}

    //constexpr S<MemLayout::pointer> operator& () { return apply(GetPointer<S<MemLayout::pointer>>{}); }
    //constexpr S<MemLayout::const_pointer> operator& () const { return apply(GetPointer<S<MemLayout::const_pointer>>{}); }
};

template <template <template <class> class> class S>
struct wrapper<S, reference> : public S<reference> {
    using Base = S<reference>;

    constexpr wrapper() = delete;
    constexpr wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    constexpr wrapper(S<value>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}
    constexpr wrapper(S<reference_restrict> other) : Base(other.apply(AggregateConstructor<Base>{})) {}
    
    constexpr wrapper(const wrapper& other) = default;

    constexpr wrapper& operator=(const wrapper<S, value>& other) {
        Base::apply(other, CopyAssignment{});
        return *this;
    }
    constexpr wrapper& operator=(const wrapper& other) {
        Base::apply(other, CopyAssignment{});
        return *this;
    }
    constexpr wrapper& operator=(const wrapper<S, const_reference>& other) {
        Base::apply(other, CopyAssignment{});
        return *this;
    }
    constexpr wrapper& operator=(const wrapper<S, reference_restrict>& other) {
        Base::apply(other, CopyAssignment{});
        return *this;
    }
    constexpr wrapper& operator=(const wrapper<S, const_reference_restrict>& other) {
        Base::apply(other, CopyAssignment{});
        return *this;
    }

    constexpr wrapper(wrapper&& other) = default;
    
    constexpr wrapper& operator=(wrapper&& other) { return operator=(other); }

    constexpr wrapper<S, pointer> operator&() { return Base::apply(GetPointer<S<pointer>>{}); }
    //constexpr wrapper<S, const_pointer> operator&() const { return Base::apply(GetPointer<S<const_pointer>>{}); }
    constexpr pointer<wrapper<S, reference>> operator->() { return this; }
};

template <template <template <class> class> class S>
struct wrapper<S, reference_restrict> : public S<reference_restrict> {
    using Base = S<reference_restrict>;

    constexpr wrapper() = delete;
    constexpr wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    constexpr wrapper(S<value>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}
    constexpr wrapper(S<reference> other) : Base(other.apply(AggregateConstructor<Base>{})) {}
    
    constexpr wrapper(const wrapper& other) = default;

    constexpr wrapper& operator=(const wrapper<S, value>& other) {
        Base::apply(other, CopyAssignment{});
        return *this;
    }
    constexpr wrapper& operator=(const wrapper& other) {
        Base::apply(other, CopyAssignment{});
        return *this;
    }
    constexpr wrapper& operator=(const wrapper<S, reference>& other) {
        Base::apply(other, CopyAssignment{});
        return *this;
    }
    constexpr wrapper& operator=(const wrapper<S, const_reference>& other) {
        Base::apply(other, CopyAssignment{});
        return *this;
    }
    constexpr wrapper& operator=(const wrapper<S, const_reference_restrict>& other) {
        Base::apply(other, CopyAssignment{});
        return *this;
    }

    constexpr wrapper(wrapper&& other) = default;
    
    constexpr wrapper& operator=(wrapper&& other) { return operator=(other); }

    constexpr wrapper<S, pointer> operator&() { return Base::apply(GetPointer<S<pointer>>{}); }
    //constexpr wrapper<S, const_pointer> operator&() const { return Base::apply(GetPointer<S<const_pointer>>{}); }
    constexpr pointer<wrapper<S, reference>> operator->() { return this; }
};

template <template <template <class> class> class S>
struct wrapper<S, const_reference> : public S<const_reference> {
    using Base = S<const_reference>;

    constexpr wrapper() = delete;
    constexpr wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    constexpr wrapper(const S<value>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}
    constexpr wrapper(const S<reference>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}
    constexpr wrapper(const S<reference_restrict>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}
    constexpr wrapper(const S<const_reference_restrict>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}

    constexpr wrapper<S, const_pointer> operator&() const { return Base::apply(GetPointer<S<const_pointer>>{}); }
    constexpr const_pointer<wrapper<S, const_reference>> operator->() const { return this; }
};

template <template <template <class> class> class S>
struct wrapper<S, const_reference_restrict> : public S<const_reference_restrict> {
    using Base = S<const_reference_restrict>;

    constexpr wrapper() = delete;
    constexpr wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    constexpr wrapper(const S<value>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}
    constexpr wrapper(const S<reference>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}
    constexpr wrapper(const S<reference_restrict>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}
    constexpr wrapper(const S<const_reference>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}

    constexpr wrapper<S, const_pointer> operator&() const { return Base::apply(GetPointer<S<const_pointer>>{}); }
    constexpr const_pointer<wrapper<S, const_reference>> operator->() const { return this; }
};

template <template <template <class> class> class S>
struct wrapper<S, pointer> : public S<pointer> {
    using Base = S<pointer>;

    constexpr wrapper() = default;
    constexpr wrapper(Base b) : Base{static_cast<Base&&>(b)} {}

    constexpr wrapper<S, reference> operator[] (size_t i) { return Base::apply(RandomAccessAt<S<reference>>{i}); }
    constexpr const wrapper<S, const_reference> operator[] (size_t i) const { return Base::apply(RandomAccessAt<S<const_reference>>{i}); }

    constexpr wrapper<S, reference> operator*() { return operator[](0); }
    constexpr wrapper<S, const_reference> operator*() const { return operator[](0); }
    constexpr wrapper<S, reference> operator->() { return operator[](0); }
    constexpr wrapper<S, const_reference> operator->() const { return operator[](0); }

    constexpr bool operator==(const wrapper& other) const { return Base::apply(FirstMember{}) == other.apply(FirstMember{}); }
    constexpr bool operator!=(const wrapper& other) const { return !this->operator==(other); }
    constexpr bool operator<(const wrapper& other) const { return Base::apply(FirstMember{}) < other.apply(FirstMember{}); }

    constexpr wrapper operator+(ptrdiff_t i) const { return Base::apply(Advance<Base>{i}); }
    constexpr wrapper operator-(ptrdiff_t i) const { return operator+(-i); }
    constexpr ptrdiff_t operator-(const wrapper& other) const { return Base::apply(FirstMember{}) - other.apply(FirstMember{}); }

    constexpr wrapper& operator++() { Base::apply(PreIncrement<Base>{}); return *this; }
    constexpr wrapper& operator+=(ptrdiff_t i) { return *this = *this + i; }
    constexpr wrapper& operator--() { Base::apply(PreDecrement<Base>{}); return *this; }
    constexpr wrapper& operator-=(ptrdiff_t i) { return *this = *this - i; }
};

template <template <template <class> class> class S>
struct wrapper<S, const_pointer> : public S<const_pointer> {
    using Base = S<const_pointer>;

    constexpr wrapper() = default;
    constexpr wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    constexpr wrapper(const S<pointer>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}

    constexpr wrapper<S, const_reference> operator[] (size_t i) const { return Base::apply(RandomAccessAt<S<const_reference>>{i}); }
    constexpr wrapper<S, const_reference> operator*() const { return operator[](0); }
    constexpr wrapper<S, const_reference> operator->() const { return operator[](0); }

    constexpr bool operator==(const wrapper& other) const { return Base::apply(FirstMember{}) == other.apply(FirstMember{}); }
    constexpr bool operator!=(const wrapper& other) const { return !this->operator==(other); }
    constexpr bool operator<(const wrapper& other) const { return Base::apply(FirstMember{}) < other.apply(FirstMember{}); }

    constexpr wrapper operator+(ptrdiff_t i) const { return Base::apply(Advance<Base>{i}); }
    constexpr wrapper operator-(ptrdiff_t i) const { return operator+(-i); }
    constexpr ptrdiff_t operator-(const wrapper& other) const { return Base::apply(FirstMember{}) - other.apply(FirstMember{}); }

    constexpr wrapper& operator++() { Base::apply(PreIncrement<Base>{}); return *this; }
    constexpr wrapper& operator+=(ptrdiff_t i) { return *this = *this + i; }
    constexpr wrapper& operator--() { Base::apply(PreDecrement<Base>{}); return *this; }
    constexpr wrapper& operator-=(ptrdiff_t i) { return *this = *this - i; }
};

enum Flag { soa, aos };

template <template <template <class> class> class S, template <class> class F, Flag L>
struct interface;

template <template <template <class> class> class S, template <class> class F>
struct interface<S, F, Flag::aos> { using type = F<S<value>>; };

template <template <template <class> class> class S, template <class> class F>
struct interface<S, F, Flag::soa> { using type = wrapper<S, F>; };

}  // namespace MemLayout

#define MEMLAYOUT_APPLY_UNARY(...)\
    template <class Function>\
    constexpr auto apply(Function&& f) { return f(__VA_ARGS__); }\
    template <class Function>\
    constexpr auto apply(Function&& f) const { return f(__VA_ARGS__); }\

#define MEMLAYOUT_EXPAND(m) f(m, other.m)

#define MEMLAYOUT_APPLY_BINARY(STRUCT_NAME, ...)\
    template <template <class> class F_other, class Function>\
    constexpr STRUCT_NAME apply(STRUCT_NAME<F_other>& other, Function&& f) { return {__VA_ARGS__}; }\
    template <template <class> class F_other, class Function>\
    constexpr STRUCT_NAME apply(STRUCT_NAME<F_other>& other, Function&& f) const { return {__VA_ARGS__}; }\
    template <template <class> class F_other, class Function>\
    constexpr STRUCT_NAME apply(const STRUCT_NAME<F_other>& other, Function&& f) { return {__VA_ARGS__}; }\
    template <template <class> class F_other, class Function>\
    constexpr STRUCT_NAME apply(const STRUCT_NAME<F_other>& other, Function&& f) const { return {__VA_ARGS__}; }\

#endif // MEMLAYOUT_H
