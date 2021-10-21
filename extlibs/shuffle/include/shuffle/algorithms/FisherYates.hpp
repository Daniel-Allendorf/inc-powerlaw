#pragma once

#include <array>
#include <cassert>
#include <iterator>
#include <random>

#include <shuffle/random/UniformRange.hpp>

#include <tlx/define.hpp>
#include <tlx/math.hpp>

namespace shuffle {
namespace details {

template <unsigned kBits, typename Gen>
uint32_t uniform_random_arb_bits(uint32_t max_exclusive, uint64_t random, Gen &gen) {
    constexpr auto kMask = (uint64_t(1) << kBits) - 1;
    static_assert(kBits > 1, "need at least on two bits");
    assert(max_exclusive < kMask);
    assert(max_exclusive > 0);

    auto r = random & kMask;
    auto m = r * max_exclusive;

    if (TLX_UNLIKELY(m & kMask < max_exclusive)) {
        uint64_t t = (kMask + 1 - max_exclusive) % max_exclusive;
        while ( (m & kMask) < t ) {
            m = (gen() & kMask) * static_cast<uint64_t>(max_exclusive);
        }
    }

    return m >> kBits;
}


template<typename Gen>
static std::tuple<uint32_t, uint32_t> uniform2(Gen &gen, uint32_t max_exclusive) {
    auto r = gen();

    return {
        uniform_random_arb_bits<32>(max_exclusive    , r      , gen),
        uniform_random_arb_bits<32>(max_exclusive - 1, r >> 32, gen)
    };
}


template<typename Gen>
static std::tuple<uint32_t, uint32_t, uint32_t> uniform3(Gen &gen, uint32_t max_exclusive) {
    auto r = gen();

    return {
        uniform_random_arb_bits<22>(max_exclusive    , r      , gen),
        uniform_random_arb_bits<21>(max_exclusive - 1, r >> 22, gen),
        uniform_random_arb_bits<21>(max_exclusive - 2, r >> 43, gen)
    };

}

template <typename UniformRange, unsigned Prefetch, typename Iter, typename Gen>
void fisher_yates_impl(Iter begin, Iter end, Gen &gen) {
    const auto n = std::distance(begin, end);
    if (TLX_UNLIKELY(n < 2))
        return;

    if (TLX_UNLIKELY(n < 2 * Prefetch))
        return fisher_yates_impl<UniformRange, 0, Iter, Gen>(begin, end, gen);

    UniformRange range(n);
    using std::swap;

    if constexpr (Prefetch == 0) {
        for (; range.max() > 1; ++begin) {
            range.decrement_max();
            auto rnd = range(gen);
            assert(static_cast<ptrdiff_t>(rnd) < std::distance(begin, end));
            auto partner = std::next(begin, rnd);
            swap(*begin, *partner);
        }

    } else {
        constexpr unsigned kPrefetchMask = Prefetch - 1;
        static_assert(kPrefetchMask == (kPrefetchMask | (kPrefetchMask >> 1)),
                      "Prefetch needs to be power of two");

        std::array<Iter, Prefetch> indices;

        auto sample_and_prefetch = [&](auto i, Iter begin) {
            range.decrement_max();
            indices[i] = begin + range(gen);
            __builtin_prefetch(&*indices[i]);
        };

        for (unsigned i = 0; i < Prefetch; ++i) {
            sample_and_prefetch(i, begin + i);
        }

        unsigned i = 0;
        auto prefetch_end = begin + (n - Prefetch - 1);
        for (; begin != prefetch_end; ++begin) {
            swap(*begin, *indices[i]);
            sample_and_prefetch(i, begin + Prefetch);
            i = (i + 1) & kPrefetchMask;
        }

        for (; begin + 1 != end; ++begin) {
            swap(*begin, *indices[i]);
            i = (i + 1) & kPrefetchMask;
        }
    }

    assert(begin + 1 == end);
}

template <unsigned Prefetch, typename Iter, typename Gen>
void fisher_yates_impl_multi(Iter begin, Iter end, Gen &gen) {
    auto n = static_cast<size_t>(std::distance(begin, end));
    if (TLX_UNLIKELY(n < 2))
        return;

    if (TLX_UNLIKELY(n < 2 * Prefetch))
        return fisher_yates_impl_multi<0, Iter, Gen>(begin, end, gen);

    using std::swap;

    constexpr auto kRandom2Upper = size_t(1) << 29;
    constexpr auto kRandom3Upper = size_t(1) << 18;

    if constexpr (Prefetch == 0) {
        for (; n > kRandom2Upper; n -= 1, begin += 1) {
            auto rand = shuffle::nearlydivisionless(n, gen);
            swap(begin[0], begin[rand]);
        }

        for (; n > kRandom3Upper; n -= 2, begin += 2) {
            auto rand = uniform2(gen, n);
            swap(begin[0], begin[std::get<0>(rand)]);
            swap(begin[1], begin[std::get<1>(rand)+1]);
        }

        for (; n > 4; n -= 3, begin += 3) {
            auto rand = uniform3(gen, n);
            swap(begin[0], begin[std::get<0>(rand)]);
            swap(begin[1], begin[std::get<1>(rand)+1]);
            swap(begin[2], begin[std::get<2>(rand)+2]);
        }

        for (; n > 1; n -= 1, begin += 1) {
            auto rand = shuffle::nearlydivisionless(n, gen);
            swap(begin[0], begin[rand]);
        }

    } else {
        /*
        constexpr unsigned kPrefetchMask = Prefetch - 1;
        static_assert(kPrefetchMask == (kPrefetchMask | (kPrefetchMask >> 1)),
                      "Prefetch needs to be power of two");

        std::array<Iter, Prefetch> indices;

        auto sample_and_prefetch = [&](auto i, Iter begin) {
            range.decrement_max();
            indices[i] = begin + range(gen);
            __builtin_prefetch(&*indices[i]);
        };

        for (unsigned i = 0; i < Prefetch; ++i) {
            sample_and_prefetch(i, begin + i);
        }

        unsigned i = 0;
        auto prefetch_end = begin + (n - Prefetch - 1);
        for (; begin != prefetch_end; ++begin) {
            swap(*begin, *indices[i]);
            sample_and_prefetch(i, begin + Prefetch);
            i = (i + 1) & kPrefetchMask;
        }

        for (; begin + 1 != end; ++begin) {
            swap(*begin, *indices[i]);
            i = (i + 1) & kPrefetchMask;
        }
        */
    }

    assert(begin + 1 == end);
}


template <typename UniformRange, unsigned BlockSize, bool Prefetch, typename Iter, typename Gen>
void fisher_yates_impl_blocked(Iter begin, Iter end, Gen &gen) {
    const auto n = std::distance(begin, end);
    if (TLX_UNLIKELY(n < 2))
        return;

    if (TLX_UNLIKELY(n < 2 * BlockSize))
        return fisher_yates_impl<UniformRange, 0, Iter, Gen>(begin, end, gen);

    UniformRange range(n);
    using std::swap;

    std::array<unsigned, BlockSize> block;
    unsigned i = 0;

    for (; i + BlockSize < n; i += BlockSize) {
        for (unsigned j = 0; j < BlockSize; ++j) {
            range.decrement_max();
            block[j] = range(gen);

            if constexpr (Prefetch)
                __builtin_prefetch(&*(begin + block[j]));
        }

        for (unsigned j = 0; j < BlockSize; ++j, ++begin) {
            auto partner = std::next(begin, block[j]);
            swap(*begin, *partner);
        }
    }

    for (; range.max() > 1; ++begin) {
        range.decrement_max();
        auto partner = std::next(begin, range(gen));
        swap(*begin, *partner);
    }

    assert(begin + 1 == end);
}

} // namespace details

template <typename Iter, typename Gen>
void fisher_yates(Iter begin, Iter end, Gen &gen) {
    if (std::distance(begin, end) < std::numeric_limits<uint32_t>::max())
        return details::fisher_yates_impl<UniformRangeNoDiv<uint32_t>, 0, Iter, Gen>(begin, end,
                                                                                     gen);

    return details::fisher_yates_impl_blocked<UniformRangeNoDiv<uint64_t>, 16, true, Iter, Gen>(
        begin, end, gen);
}

} // namespace shuffle
