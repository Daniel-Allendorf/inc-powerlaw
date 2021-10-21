#pragma once

#include <cassert>
#include <random>
#include <tlx/define.hpp>
#include <tlx/math.hpp>
#include <shuffle/Concepts.hpp>

namespace shuffle {

template <typename T>
class UniformRangeDivision {
public:
    using value_type = T;

    UniformRangeDivision() = delete;

    constexpr UniformRangeDivision(value_type max)
        : scaler_(kMaxValue / max), scaled_max_(max * scaler_), max_(max) {
        assert(max > 0);
    }

    template <typename URNG>
    constexpr value_type operator()(URNG &gen) const {
        // static_assert(gen.max() >= kMaxValue, "Generator not supported");

        while (true) {
            const auto value = static_cast<value_type>(gen());
            if (TLX_LIKELY(value <= scaled_max_))
                return value / scaler_;
        }
    }

    constexpr void decrement_max() {
        scaled_max_ -= scaler_;
        max_--;

        if (TLX_UNLIKELY(kMaxValue / 5 > scaled_max_ / 4)) {
            scaler_ = kMaxValue / max_;
            scaled_max_ = max_ * scaler_;
        }
    }

    constexpr value_type max() const { return max_; }

private:
    static constexpr value_type kMaxValue = std::numeric_limits<value_type>::max();
    value_type scaler_;
    value_type scaled_max_;
    value_type max_;
};

template <typename T>
class UniformRangeSTL {
public:
    using value_type = T;

    UniformRangeSTL() = delete;

    constexpr UniformRangeSTL(value_type max) : distr_{0, max} {}

    template <typename URNG>
    constexpr value_type operator()(URNG &gen) {
        return distr_(gen);
    }

    constexpr void decrement_max() {
        const auto max = distr_.max();
        assert(max >= 1);
        distr_ = std::uniform_int_distribution<value_type>{0, max - 1};
    }

    constexpr value_type max() const { return distr_.max(); }

private:
    std::uniform_int_distribution<value_type> distr_;
};

template <typename T>
class UniformRangeMask {
public:
    using value_type = T;

    UniformRangeMask() = delete;

    constexpr UniformRangeMask(value_type max) : max_(max) {
        mask_ = max_;
        mask_ |= mask_ >> 1;
        mask_ |= mask_ >> 2;
        mask_ |= mask_ >> 4;
        mask_ |= mask_ >> 8;

        if constexpr (sizeof(value_type) >= 4)
            mask_ |= mask_ >> 16;

        if constexpr (sizeof(value_type) >= 8)
            mask_ |= mask_ >> 32;
    }

    template <typename URNG>
    constexpr value_type operator()(URNG &gen) const {
        // static_assert(gen.max() >= kMaxValue, "Generator not supported");

        while (true) {
            const auto value = static_cast<value_type>(gen()) & mask_;
            if (TLX_LIKELY(value <= max_))
                return value;
        }
    }

    constexpr void decrement_max() {
        max_--;
        assert(max_ > 0);
        mask_ >>= (max_ <= mask_ / 2);
    }

    constexpr value_type max() const { return max_; }

private:
    static constexpr value_type kMaxValue = std::numeric_limits<value_type>::max();
    value_type max_;
    value_type mask_;
};

template <typename T, bool Recycle = false>
class UniformRangeMask23 {
public:
    using value_type = T;

    UniformRangeMask23() = delete;

    constexpr UniformRangeMask23(value_type max) : max_(max) {
        assert(max > 0);

        bits_ = tlx::integer_log2_ceil(max);
    }

    template <typename URNG>
    constexpr value_type operator()(URNG &gen) {
        auto mask = mask_value(bits_);
        auto mask2 = 2 * mask | 1;
        auto max3 = max_ * 3 + 2;

        if (mask2 < max3) {
            while (true) {
                const auto value = random_bits(gen, bits_);
                if (TLX_LIKELY(value <= max_))
                    return value;
            }
        } else {
            while (true) {
                const auto value = random_bits(gen, bits_ + 1);
                if (TLX_LIKELY(value <= max3))
                    return value / 3;
            }
        }
    }

    constexpr void increment_max() {
        max_++;

        const bool grow = (max_ >> bits_) > 0;
        bits_ += grow;
    }

    constexpr void decrement_max() {
        max_--;

        assert(max_ > 0);
        const bool shrink = max_ < (1llu << (bits_ - 1));
        bits_ -= shrink;
    }

    constexpr value_type max() const { return max_; }

private:
    static constexpr value_type kMaxValue = std::numeric_limits<value_type>::max();
    value_type max_;
    unsigned bits_;
    bool scaled_;

    uint64_t buffer_;
    uint64_t valid_;

    constexpr value_type mask_value(unsigned bits) const {
        return static_cast<value_type>((1llu << bits) - 1);
    }

    template <typename URNG>
    value_type random_bits(URNG &gen, unsigned bits) {
        const auto mask = (1llu << bits) - 1;

        if constexpr (Recycle) {
            if (valid_ < mask) {
                valid_ = gen.max();
                buffer_ = gen();
            }

            auto result = static_cast<value_type>(buffer_ & mask);
            buffer_ >>= bits;
            valid_ >>= bits;
            return result;
        }

        return static_cast<value_type>(gen() & mask);
    }
};

template <typename Gen>
uint64_t nearlydivisionless(uint64_t s, Gen &gen) {
    uint64_t x = gen();
    __uint128_t m = (__uint128_t)x * (__uint128_t)s;
    uint64_t l = (uint64_t)m;
    if (TLX_UNLIKELY(l < s)) {
        uint64_t t = -s % s;
        while (l < t) {
            x = gen();
            m = (__uint128_t)x * (__uint128_t)s;
            l = (uint64_t)m;
        }
    }
    return m >> 64;
}

template <typename Gen>
uint64_t nearlydivisionless(uint64_t s, uint64_t s_mod_s, Gen &gen) {
    uint64_t x = gen();
    __uint128_t m = (__uint128_t)x * (__uint128_t)s;
    uint64_t l = (uint64_t)m;
    if (TLX_UNLIKELY(l < s)) {
        while (l < s_mod_s) {
            x = gen();
            m = (__uint128_t)x * (__uint128_t)s;
            l = (uint64_t)m;
        }
    }
    return m >> 64;
}

template <typename Gen>
uint32_t nearlydivisionless(uint32_t s, Gen &gen) {
#if 0
    uint32_t x = gen();
    uint64_t m = static_cast<uint64_t>(x) * static_cast<uint64_t>(s);
    uint32_t l = static_cast<uint32_t>(m);
    if (TLX_UNLIKELY(l < s)) {
        uint64_t t = -s % s;
        while (l < t) {
            x = gen();
            m = static_cast<uint64_t>(x) * static_cast<uint64_t>(s);
            l = static_cast<uint32_t>(m);
        }
    }
    return m >> 32;
#else
    return static_cast<uint32_t>(nearlydivisionless(static_cast<uint64_t>(s), gen));
#endif
}

template <typename T>
class UniformRangeNoDiv {
public:
    using value_type = T;

    UniformRangeNoDiv() = delete;

    constexpr UniformRangeNoDiv(value_type max) : max_(max) {}

    template <typename URNG>
    constexpr value_type operator()(URNG &gen) {
        const auto res = nearlydivisionless(max_ + 1, gen);
        assert(res <= max_);
        return res;
    }

    constexpr void decrement_max() {
        --max_;
        assert(max_ >= 1);
    }

    constexpr void increment_max() { ++max_; }

    constexpr value_type max() const { return max_; }

private:
    T max_;

};

} // namespace shuffle
