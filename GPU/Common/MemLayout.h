#ifndef MEMLAYOUT_H
#define MEMLAYOUT_H

#ifdef __clang__
#include <experimental/meta>
#else
#include <meta>
#endif

#include <type_traits>

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

//////////////// Reflection utilities

template <typename S>
constexpr std::size_t count_members() {
    return nonstatic_data_members_of(^^S, std::meta::access_context::current()).size();
}

consteval auto nsdms(std::meta::info type) -> std::vector<std::meta::info> {
    return nonstatic_data_members_of(type, std::meta::access_context::current());
}

//////////////// Operations on members

template <class SF>
struct RandomAccessAt {
    MemLayout::size_t i;
    template <class... Args>
    [[gnu::always_inline]] constexpr SF operator()(Args& ...args) const { return {args[i]...}; }
    template <class... Args>
    [[gnu::always_inline]] constexpr SF operator()(const Args& ...args) const { return {args[i]...}; }
};

template <class SF>
struct GetPointer {
    template <class... Args>
    [[gnu::always_inline]] constexpr SF operator()(Args& ...args) const { return {&args...}; }
    template <class... Args>
    [[gnu::always_inline]] constexpr SF operator()(const Args& ...args) const { return {&args...}; }
};

template <class SF>
struct AggregateConstructor {
    template <class... Args>
    [[gnu::always_inline]] constexpr SF operator()(Args& ...args) const { return {args...}; }
    template <class... Args>
    [[gnu::always_inline]] constexpr SF operator()(const Args& ...args) const { return {args...}; }
};

template <class SF>
struct PreIncrement {
    template <class... Args>
    [[gnu::always_inline]] constexpr SF operator()(Args& ...args) const { return {++args...}; }
};

template <class SF>
struct PreDecrement {
    template <class... Args>
    [[gnu::always_inline]] constexpr SF operator()(Args& ...args) const { return {--args...}; }
};

template <class SF>
struct Advance {
    ptrdiff_t i;
    template <class... Args>
    [[gnu::always_inline]] constexpr SF operator()(const Args& ...args) const { return {(args + i)...}; }
};

struct CopyAssignment {
    template <class Left, class Right>
    [[gnu::always_inline]] constexpr Left& operator()(Left& left, const Right& right) const { return left = right; }
};

//////////////// apply to members methods

struct apply_unary_helper {
    template <class Self, class FunctionObject, size_t... Is>
    [[gnu::always_inline]] constexpr auto operator()(Self& self, FunctionObject&& f, std::index_sequence<Is...>) const {
        return f(self.[:nsdms(^^Self)[Is]:]...);
    }
};

template <class Self, class FunctionObject>
[[gnu::always_inline]] constexpr auto apply_unary(Self &self, FunctionObject&& f) {
    constexpr auto indices = std::make_index_sequence<count_members<Self>()>{};
    return apply_unary_helper{}(self, std::forward<FunctionObject>(f), indices);
}

// apply on skeleton struct S<F>
template <class FunctionObject, template <template <class> class> class S, template <class> class F>
[[gnu::always_inline]] constexpr auto apply(S<F> &self, FunctionObject&& f) {
    return apply_unary(self, std::forward<FunctionObject&&>(f));
}

template <class FunctionObject, template <template <class> class> class S, template <class> class F>
[[gnu::always_inline]] constexpr auto apply(const S<F> &self, FunctionObject&& f) {
    return apply_unary(self, std::forward<FunctionObject&&>(f));
}

// apply on wrappers, forwarding to the base type
template <class FunctionObject, class Self> 
    requires requires { typename Self::Base; }
[[gnu::always_inline]] constexpr auto apply(Self &self, FunctionObject&& f) {
    return apply_unary<typename Self::Base>(self, std::forward<FunctionObject&&>(f));
}

template <class FunctionObject, class Self> 
    requires requires { typename Self::Base; }
[[gnu::always_inline]] constexpr auto apply(const Self &self, FunctionObject&& f) {
    return apply_unary<const typename Self::Base>(self, std::forward<FunctionObject&&>(f));
}

struct apply_binary_helper {
    template <class Self, class Other, class FunctionObject, size_t... Is>
    [[gnu::always_inline]] constexpr Self operator()(Self& self, Other& other, FunctionObject&& f, std::index_sequence<Is...>) const {
        return {f(self.[:nsdms(^^Self)[Is]:], other.[:nsdms(^^Other)[Is]:])...};
    }
};

// template <class FunctionObject, class Self, class Other>
// constexpr auto apply(Self &self, Other &other, FunctionObject&& f) {
template <class Self, class Other, class FunctionObject>
[[gnu::always_inline]] constexpr auto apply_binary(Self &self, Other &other, FunctionObject&& f) {
    constexpr auto indices = std::make_index_sequence<count_members<Self>()>{};
    return apply_binary_helper{}(self, other, std::forward<FunctionObject&&>(f), indices);
}

template <class FunctionObject, template <template <class> class> class S, template <class> class F_self, template <class> class F_other>
[[gnu::always_inline]] constexpr auto apply(S<F_self> &self, S<F_other> &other, FunctionObject&& f) {
    return apply_binary(self, other, std::forward<FunctionObject&&>(f));
}

template <class FunctionObject, template <template <class> class> class S, template <class> class F_self, template <class> class F_other>
[[gnu::always_inline]] constexpr auto apply(S<F_self> &self, const S<F_other> &other, FunctionObject&& f) {
    return apply_binary(self, other, std::forward<FunctionObject&&>(f));
}

template <class Self, class Other, class FunctionObject>
    requires requires { typename Self::Base; typename Other::Base; }
[[gnu::always_inline]] constexpr auto apply(Self &self, Other &other, FunctionObject&& f) {
    return apply_binary<typename Self::Base, typename Other::Base>(self, other, std::forward<FunctionObject&&>(f));
}

template <class Self, class Other, class FunctionObject>
    requires requires { typename Self::Base; typename Other::Base; }
[[gnu::always_inline]] constexpr auto apply(Self &self, const Other &other, FunctionObject&& f) {
    static_assert(count_members<typename Self::Base>() == 4);
    return apply_binary<typename Self::Base, const typename Other::Base>(self, other, std::forward<FunctionObject&&>(f));
}

//////////////// wrapper

template <
    template <template <class> class> class S,
    template <class> class F
>
struct wrapper : public S<F> {
    using Base = S<F>;

    constexpr wrapper() = default;
    constexpr wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    template <template <class> class F_other>
    constexpr wrapper(S<F_other>& other) : Base{apply(other, AggregateConstructor<Base>{})} {}
    template <template <class> class F_other>
    constexpr wrapper(const S<F_other>& other) : Base{apply(other, AggregateConstructor<Base>{})} {}

    [[gnu::always_inline]] constexpr wrapper<S, reference> operator[] (size_t i) { 
        return apply(*this, RandomAccessAt<S<reference>>{i}); }
    [[gnu::always_inline]] constexpr wrapper<S, const_reference> operator[] (size_t i) const { 
        return apply(*this, RandomAccessAt<S<const_reference>>{i}); }

    [[gnu::always_inline]] constexpr wrapper<S, reference> operator*() { return operator[](0); }
    [[gnu::always_inline]] constexpr wrapper<S, const_reference> operator*(ptrdiff_t) const { return operator[](0); }
};

template <template <template <class> class> class S>
struct wrapper<S, value> : public S<value> {
    using Base = S<value>;

    constexpr wrapper() = default;
    constexpr wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    constexpr wrapper(const S<reference>& other) : Base(apply(other, AggregateConstructor<Base>{})) {}
    constexpr wrapper(const S<const_reference>& other) : Base(apply(other, AggregateConstructor<Base>{})) {}

    //constexpr S<MemLayout::pointer> operator& () { return apply(GetPointer<S<MemLayout::pointer>>{}); }
    //constexpr S<MemLayout::const_pointer> operator& () const { return apply(GetPointer<S<MemLayout::const_pointer>>{}); }
};

template <template <template <class> class> class S>
struct wrapper<S, reference> : public S<reference> {
    using Base = S<reference>;

    constexpr wrapper() = delete;
    constexpr wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    constexpr wrapper(S<value>& other) : Base(apply(other, AggregateConstructor<Base>{})) {}
    constexpr wrapper(S<reference_restrict> other) : Base(apply(other, AggregateConstructor<Base>{})) {}
    
    constexpr wrapper(const wrapper& other) = default;

    constexpr wrapper& operator=(const wrapper<S, value>& other) {
        apply(*this, other, CopyAssignment{});
        return *this;
    }
    constexpr wrapper& operator=(const wrapper& other) {
        apply(*this, other, CopyAssignment{});
        return *this;
    }
    constexpr wrapper& operator=(const wrapper<S, const_reference>& other) {
        apply(*this, other, CopyAssignment{});
        return *this;
    }
    constexpr wrapper& operator=(const wrapper<S, reference_restrict>& other) {
        apply(*this, other, CopyAssignment{});
        return *this;
    }
    constexpr wrapper& operator=(const wrapper<S, const_reference_restrict>& other) {
        apply(*this, other, CopyAssignment{});
        return *this;
    }

    constexpr wrapper(wrapper&& other) = default;
    
    constexpr wrapper& operator=(wrapper&& other) { return operator=(other); }

    constexpr wrapper<S, pointer> operator&() { return apply(*this, GetPointer<S<pointer>>{}); }
    //constexpr wrapper<S, const_pointer> operator&() const { return apply(*this, GetPointer<S<const_pointer>>{}); }
    constexpr pointer<wrapper<S, reference>> operator->() { return this; }
};

template <template <template <class> class> class S>
struct wrapper<S, reference_restrict> : public S<reference_restrict> {
    using Base = S<reference_restrict>;

    constexpr wrapper() = delete;
    constexpr wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    constexpr wrapper(S<value>& other) : Base(apply(other, AggregateConstructor<Base>{})) {}
    constexpr wrapper(S<reference> other) : Base(apply(other, AggregateConstructor<Base>{})) {}
    
    constexpr wrapper(const wrapper& other) = default;

    constexpr wrapper& operator=(const wrapper<S, value>& other) {
        apply(*this, other, CopyAssignment{});
        return *this;
    }
    constexpr wrapper& operator=(const wrapper& other) {
        apply(*this, other, CopyAssignment{});
        return *this;
    }
    constexpr wrapper& operator=(const wrapper<S, reference>& other) {
        apply(*this, other, CopyAssignment{});
        return *this;
    }
    constexpr wrapper& operator=(const wrapper<S, const_reference>& other) {
        apply(*this, other, CopyAssignment{});
        return *this;
    }
    constexpr wrapper& operator=(const wrapper<S, const_reference_restrict>& other) {
        apply(*this, other, CopyAssignment{});
        return *this;
    }

    constexpr wrapper(wrapper&& other) = default;
    
    constexpr wrapper& operator=(wrapper&& other) { return operator=(other); }

    constexpr wrapper<S, pointer> operator&() { return apply(*this, GetPointer<S<pointer>>{}); }
    //constexpr wrapper<S, const_pointer> operator&() const { return apply(*this, GetPointer<S<const_pointer>>{}); }
    constexpr pointer<wrapper<S, reference>> operator->() { return this; }
};

template <template <template <class> class> class S>
struct wrapper<S, const_reference> : public S<const_reference> {
    using Base = S<const_reference>;

    constexpr wrapper() = delete;
    constexpr wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    constexpr wrapper(const S<value>& other) : Base(apply(other, AggregateConstructor<Base>{})) {}
    constexpr wrapper(const S<reference>& other) : Base(apply(other, AggregateConstructor<Base>{})) {}
    constexpr wrapper(const S<reference_restrict>& other) : Base(apply(other, AggregateConstructor<Base>{})) {}
    constexpr wrapper(const S<const_reference_restrict>& other) : Base(apply(other, AggregateConstructor<Base>{})) {}

    constexpr wrapper<S, const_pointer> operator&() const { return apply(*this, GetPointer<S<const_pointer>>{}); }
    constexpr const_pointer<wrapper<S, const_reference>> operator->() const { return this; }
};

template <template <template <class> class> class S>
struct wrapper<S, const_reference_restrict> : public S<const_reference_restrict> {
    using Base = S<const_reference_restrict>;

    constexpr wrapper() = delete;
    constexpr wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    constexpr wrapper(const S<value>& other) : Base(apply(other, AggregateConstructor<Base>{})) {}
    constexpr wrapper(const S<reference>& other) : Base(apply(other, AggregateConstructor<Base>{})) {}
    constexpr wrapper(const S<reference_restrict>& other) : Base(apply(other, AggregateConstructor<Base>{})) {}
    constexpr wrapper(const S<const_reference>& other) : Base(apply(other, AggregateConstructor<Base>{})) {}

    constexpr wrapper<S, const_pointer> operator&() const { return apply(*this, GetPointer<S<const_pointer>>{}); }
    constexpr const_pointer<wrapper<S, const_reference>> operator->() const { return this; }
};

template <template <template <class> class> class S>
struct wrapper<S, pointer> : public S<pointer> {
    using Base = S<pointer>;

    constexpr wrapper() = default;
    constexpr wrapper(Base b) : Base{static_cast<Base&&>(b)} {}

    [[gnu::always_inline]] constexpr wrapper<S, reference> operator[] (size_t i) { 
        return apply(*this, RandomAccessAt<S<reference>>{i}); }
    [[gnu::always_inline]] constexpr const wrapper<S, const_reference> operator[] (size_t i) const { 
        return apply(*this, RandomAccessAt<S<const_reference>>{i}); }

    [[gnu::always_inline]] constexpr wrapper<S, reference> operator*() { return operator[](0); }
    [[gnu::always_inline]] constexpr wrapper<S, const_reference> operator*() const { return operator[](0); }
    [[gnu::always_inline]] constexpr wrapper<S, reference> operator->() { return operator[](0); }
    [[gnu::always_inline]] constexpr wrapper<S, const_reference> operator->() const { return operator[](0); }

    constexpr bool operator==(const wrapper& other) const { 
        return this->[:nsdms(^^Base)[0]:] == other.[:nsdms(^^Base)[0]:]; }
    constexpr bool operator!=(const wrapper& other) const { 
        return !this->operator==(other); }
    constexpr bool operator<(const wrapper& other) const { 
        return this->[:nsdms(^^Base)[0]:] < other.[:nsdms(^^Base)[0]:]; }
    constexpr bool operator<=(const wrapper& other) const { 
        return this->[:nsdms(^^Base)[0]:] <= other.[:nsdms(^^Base)[0]:]; }
    constexpr bool operator>(const wrapper& other) const { 
        return this->[:nsdms(^^Base)[0]:] > other.[:nsdms(^^Base)[0]:]; }
    constexpr bool operator>=(const wrapper& other) const { 
        return this->[:nsdms(^^Base)[0]:] >= other.[:nsdms(^^Base)[0]:]; }

    constexpr wrapper operator+(ptrdiff_t i) const { return apply(*this, Advance<Base>{i}); }
    constexpr wrapper operator-(ptrdiff_t i) const { return operator+(-i); }
    constexpr ptrdiff_t operator-(const wrapper& other) const { 
        return this->[:nsdms(^^Base)[0]:] - other.[:nsdms(^^Base)[0]:]; }

    constexpr wrapper& operator++() { apply(*this, PreIncrement<Base>{}); return *this; }
    constexpr wrapper& operator+=(ptrdiff_t i) { return *this = *this + i; }
    constexpr wrapper& operator--() { apply(*this, PreDecrement<Base>{}); return *this; }
    constexpr wrapper& operator-=(ptrdiff_t i) { return *this = *this - i; }
};

template <template <template <class> class> class S>
struct wrapper<S, const_pointer> : public S<const_pointer> {
    using Base = S<const_pointer>;

    constexpr wrapper() = default;
    constexpr wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    constexpr wrapper(const S<pointer>& other) : Base(apply(other, AggregateConstructor<Base>{})) {}

    [[gnu::always_inline]] constexpr wrapper<S, const_reference> operator[] (size_t i) const { 
        return apply(*this, RandomAccessAt<S<const_reference>>{i}); }
    [[gnu::always_inline]] constexpr wrapper<S, const_reference> operator*() const { return operator[](0); }
    [[gnu::always_inline]] constexpr wrapper<S, const_reference> operator->() const { return operator[](0); }

    [[gnu::always_inline]] constexpr bool operator==(const wrapper& other) const { 
        return this->[:nsdms(^^Base)[0]:] == other.[:nsdms(^^Base)[0]:]; }
    [[gnu::always_inline]] constexpr bool operator!=(const wrapper& other) const { 
        return !this->operator==(other); }
    [[gnu::always_inline]] constexpr bool operator<(const wrapper& other) const { 
        return this->[:nsdms(^^Base)[0]:] < other.[:nsdms(^^Base)[0]:]; }

    constexpr wrapper operator+(ptrdiff_t i) const { 
        return apply(*this, Advance<Base>{i}); }
    constexpr wrapper operator-(ptrdiff_t i) const { return operator+(-i); }
    constexpr ptrdiff_t operator-(const wrapper& other) const { 
        return this->[:nsdms(^^Base)[0]:] - other.[:nsdms(^^Base)[0]:]; }

    constexpr wrapper& operator++() { apply(*this, PreIncrement<Base>{}); return *this; }
    constexpr wrapper& operator+=(ptrdiff_t i) { return *this = *this + i; }
    constexpr wrapper& operator--() { apply(*this, PreDecrement<Base>{}); return *this; }
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

#endif // MEMLAYOUT_H
