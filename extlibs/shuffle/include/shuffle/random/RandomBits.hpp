#pragma once

#include <cassert>
#include <random>
#include <shuffle/Concepts.hpp>

namespace shuffle {

template <typename T = uint64_t>
class RandomBits {
public:
    using result_type = T;
    constexpr static auto kMaxBits = sizeof(result_type) * 8;

    template <typename Gen>
    result_type operator()(Gen &gen) REQUIRES(std::uniform_random_bit_generator<Gen>) {
        return get<1, Gen>(gen);
    }

    template <typename Gen>
    result_type operator()(Gen &gen, unsigned bits)
        REQUIRES(std::uniform_random_bit_generator<Gen>) {
        return get(gen, bits);
    }

    template <typename Gen>
    result_type get(Gen &gen, unsigned bits) REQUIRES(std::uniform_random_bit_generator<Gen>) {
        assert(bits <= kMaxBits);
        const auto mask = (result_type(1) << bits) - 1;
        if (valid_ < mask) {
            buffer_ = gen();
            valid_ = gen.max();

            // we assume that gen.max()+1 is a power-of-two
            assert((valid_ & mask) == mask);
            assert(valid_ == (valid_ | (valid_ >> 1)));
        }

        const auto result = buffer_ & mask;
        buffer_ >>= bits;
        valid_ >>= bits;
        return result;
    }

    template <unsigned Bits, typename Gen>
    result_type get(Gen &gen) REQUIRES(std::uniform_random_bit_generator<Gen>) {
        if constexpr (!Bits)
            return 0;

        static_assert(Bits <= kMaxBits);

        constexpr auto mask = (result_type(1) << Bits) - 1;
        if (valid_ < mask) {
            buffer_ = gen();
            valid_ = gen.max();

            // we assume that gen.max()+1 is a power-of-two
            assert((valid_ & mask) == mask);
            assert(valid_ == (valid_ | (valid_ >> 1)));
        }

        const auto result = buffer_ & mask;
        buffer_ >>= Bits;
        valid_ >>= Bits;
        return result;
    }

private:
    result_type buffer_{0};
    result_type valid_{0};
};

using FairCoin = RandomBits<>;

} // namespace shuffle