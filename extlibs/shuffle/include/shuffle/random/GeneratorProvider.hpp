#pragma once

#include <array>
#include <random>
#include <vector>

#include <omp.h>

#include <shuffle/Concepts.hpp>

#ifdef SHUFFLE_USE_RANDUTILS
#include <shuffle/randutils.hpp>
#endif

namespace shuffle {

template <typename Gen, size_t CacheLineBytes = 64>
REQUIRES(std::uniform_random_bit_generator<Gen>)
class GeneratorProvider {
    static constexpr size_t kUsedCacheLines = (sizeof(Gen) + CacheLineBytes - 1) / CacheLineBytes;
    static constexpr size_t kSpacingSize = kUsedCacheLines * CacheLineBytes + 1;

    union spaced_type {
        Gen payload;
        std::array<uint8_t, kSpacingSize> dummy;
    };

public:
    using generator_type = Gen;

    explicit GeneratorProvider(unsigned num_threads = omp_get_max_threads()) {
        generators_.reserve(num_threads);
#ifdef SHUFFLE_USE_RANDUTILS
        while (generators_.size() < num_threads) {
            generators_.emplace_back(spaced_type{Gen{randutils::auto_seed_128{}.base()}});
#else
        std::random_device rd;
        while (generators_.size() < num_threads) {
            generators_.emplace_back(spaced_type{Gen{rd()}});
#endif
        }
    }

    explicit GeneratorProvider(Gen &gen, unsigned num_threads = omp_get_max_threads()) {
        generators_.reserve(num_threads);
        while (generators_.size() < num_threads)
            generators_.emplace_back(
                spaced_type{Gen{gen()}}); // this is a somewhat suboptimal seeding as we provide
                                          // at most 64bits of entropy
    }

    GeneratorProvider(const GeneratorProvider &) = delete;
    GeneratorProvider &operator=(const GeneratorProvider &) = delete;

    // ONLY USE IT FOR TESTING
    void seed(unsigned seed) {
        for (auto &gen : generators_)
            gen.payload.seed(++seed);
    }

    generator_type &operator()(unsigned thread_id = omp_get_thread_num()) {
        return generators_.at(thread_id).payload;
    }

    size_t num_threads() const { return generators_.size(); }

private:
    std::vector<spaced_type> generators_;
};

template <typename Gen>
struct IsGeneratorProvider {
    static constexpr bool value = false;
};

template <typename Gen, unsigned K>
struct IsGeneratorProvider<GeneratorProvider<Gen, K>> {
    static constexpr bool value = true;
};

} // namespace shuffle