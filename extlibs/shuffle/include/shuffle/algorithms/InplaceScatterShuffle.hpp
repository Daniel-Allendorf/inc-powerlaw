/*******************************************************************************
 * InplaceScatterShuffle.hpp
 *
 * Copyright (C) 2021 Manuel Penschuck <manuel@algorithm.engineering>
 *
 * All rights reserved. Published under the Boost Software License, Version 1.0
 ******************************************************************************/

#pragma once

#ifndef INPLACE_SCATTER_SHUFFLE_HPP
#define INPLACE_SCATTER_SHUFFLE_HPP

#include <cassert>
#include <numeric>
#include <type_traits>

#include <omp.h>

#include <tlx/define.hpp>
#include <tlx/math.hpp>

#include <shuffle/algorithms/FisherYates.hpp>

#include <shuffle/random/GeneratorProvider.hpp>
#include <shuffle/random/RandomBits.hpp>
#include <shuffle/random/UniformRange.hpp>

//////////////////////////////////////
///// WARNING: WE DO ONLY SHUFFLE THIS FIRST HALF !!!
//////////////////////////////////////

namespace shuffle {
namespace iss {

enum class ScatterAlgorithm {
    Direct,
    DirectBuffered,
    BinaryScatter,
    BinaryScatterAVX,
};

template <unsigned DefaultNumSegements = 128,                      //
          size_t DefaultCutOff = 1llu << 18,                       //
          ScatterAlgorithm ScatterAlgo = ScatterAlgorithm::Direct, //
          bool SwapRangeAVX = false>
struct Config {
    static constexpr unsigned kDefaultNumSegments = DefaultNumSegements;
    static constexpr size_t kDefaultCutOff = DefaultCutOff;
    static constexpr ScatterAlgorithm kScatterAlgorithm = ScatterAlgo;
    static constexpr bool kSwapRangeAVX = SwapRangeAVX;
};

template <typename Iter, typename Config = Config<>>
struct Impl {
    using iterator_type = Iter;
    using value_type = typename std::iterator_traits<Iter>::value_type;

    struct BlockDescriptor {
        Iter begin;
        Iter placed_end;
        Iter stash_begin;
        Iter end;
        size_t overflow{0};

        // | placed items | unplaced items | stashed items |
        // ^- begin
        //                ^- placed_end
        //                                 ^- stashed_begin
        //                                                 ^- end
        BlockDescriptor() = default;

        BlockDescriptor(Iter begin, Iter end)
            : begin(begin), placed_end(begin), stash_begin(end), end(end) {}

        void swap_into_placed(value_type &other) {
            using std::swap;
            swap(*placed_end, other);
            ++placed_end;
            __builtin_prefetch(&*placed_end);
        }

        void swap_into_stash(value_type &other) {
            stash_begin--;
            using std::swap;
            swap(*stash_begin, other);
        }

        void unstash_and_swap_into_placed(value_type &other) {
            assert(!has_unplaced());

            using std::swap;
            swap(*placed_end, other);
            ++placed_end;
            ++stash_begin;
        }

        value_type &placed_item() {
            assert(has_placed());
            return *(placed_end - 1);
        }

        value_type &unplaced_item() {
            assert(has_unplaced());
            return *placed_end;
        }

        value_type &stashed_item() {
            assert(has_stashed());
            return *stash_begin;
        }

        void stash_last_placed_item() {
            assert(has_placed());
            assert(!has_unplaced());
            --placed_end;
            --stash_begin;
        }

        [[nodiscard]] size_t num_unplaced() const noexcept {
            return std::distance(placed_end, stash_begin);
        }

        [[nodiscard]] size_t num_placed() const noexcept { //
            return std::distance(begin, placed_end);
        }

        [[nodiscard]] size_t num_stashed() const noexcept { //
            return std::distance(stash_begin, end);
        }

        [[nodiscard]] bool has_unplaced() const noexcept { return placed_end != stash_begin; }

        [[nodiscard]] bool has_placed() const noexcept { return begin != placed_end; }

        [[nodiscard]] bool has_stashed() const noexcept { return stash_begin != end; }

        void take_from_left_stash(size_t num_elems, BlockDescriptor &left) {
            assert(!has_unplaced());
            assert(left.end == begin);
            assert(left.num_stashed() >= num_elems);

            while (num_elems--) {
                using std::swap;

                --left.end;
                --stash_begin;
                swap(*stash_begin, *left.end);
            }

            begin = left.end;
            placed_end = stash_begin;
        }

        void take_from_right_stash(size_t num_elems, BlockDescriptor &right) {
            assert(end == right.begin);
            assert(num_elems <= right.num_stashed());
            assert(!right.has_unplaced());

            while (num_elems--) {
                using std::swap;

                swap(*right.begin, *right.stash_begin);
                ++right.begin;
                ++right.stash_begin;
            }

            right.placed_end = right.stash_begin;
            end = right.begin;
        }
    };

    using block_iterator = typename std::vector<BlockDescriptor>::iterator;

    ////////////////// DIRECT SELECTOR
    template <typename Gen>
    static void rough_scatter(block_iterator block_begin, block_iterator block_end, Gen &gen) {
        const auto num_blocks = static_cast<size_t>(std::distance(block_begin, block_end));
        if (TLX_UNLIKELY(num_blocks > 256 || !tlx::is_power_of_two(num_blocks)))
            return rough_scatter_generic(block_begin, block_end, gen);

        switch (tlx::integer_log2_ceil(num_blocks)) {
        case 1:
            return rough_scatter_pot<1>(block_begin, block_end, gen);
        case 2:
            return rough_scatter_pot<2>(block_begin, block_end, gen);
        case 3:
            return rough_scatter_pot<3>(block_begin, block_end, gen);
        case 4:
            return rough_scatter_pot<4>(block_begin, block_end, gen);
        case 5:
            return rough_scatter_pot<5>(block_begin, block_end, gen);
        case 6:
            return rough_scatter_pot<6>(block_begin, block_end, gen);
        case 7:
            return rough_scatter_pot<7>(block_begin, block_end, gen);
        case 8:
            return rough_scatter_pot<8>(block_begin, block_end, gen);
        }

        abort();
    }

    ////////////////// DIRECT SCATTER
    template <typename BlockIter, typename Gen>
    static void rough_scatter_generic(BlockIter block_begin, BlockIter block_end, Gen &gen) {
        auto do_it = [&](auto uniform_variate) {
            for (auto active_block_iter = block_begin; active_block_iter != block_end;
                 ++active_block_iter) {
                auto &active_block = *active_block_iter;

                while (active_block.has_unplaced()) {
                    const auto partner_index = uniform_variate();
                    assert(partner_index
                           < static_cast<size_t>(std::distance(block_begin, block_end)));
                    auto &partner_segment = block_begin[partner_index];

                    if (TLX_UNLIKELY(!partner_segment.has_unplaced())) {
                        // segment is full, we've to stash the overflow at the end of our active
                        // segment
                        active_block.swap_into_stash(active_block.unplaced_item());
                        ++partner_segment.overflow;
                        continue;
                    }

                    partner_segment.swap_into_placed(active_block.unplaced_item());
                }
            }
        };

        const auto num_segments = std::distance(block_begin, block_end);
        if (tlx::is_power_of_two(num_segments)) {
            const auto num_bits = tlx::integer_log2_ceil(num_segments);
            RandomBits<> random_bits;
            do_it([&] { return random_bits.get(gen, num_bits); });

        } else {
            do_it([&] { return nearlydivisionless(static_cast<uint64_t>(num_segments), gen); });
        }
    }

    template <unsigned kBits, typename BlockIter, typename Gen>
    static void rough_scatter_pot(BlockIter block_begin, BlockIter block_end, Gen &gen) {
        std::array<std::pair<Iter, Iter>, 1llu << kBits> pairs;
        std::transform(block_begin, block_end, pairs.begin(),
                       [](auto &block) -> std::pair<Iter, Iter> {
                           return {block.placed_end, block.stash_begin};
                       });

        shuffle::RandomBits random_bits;

        for (auto &active_pair : pairs) {
            while (active_pair.first != active_pair.second) {
                const auto partner_index = random_bits.get<kBits>(gen);
                assert(partner_index < pairs.size());
                auto &partner_pair = pairs[partner_index];

                using std::swap;

                if (TLX_UNLIKELY(partner_pair.first == partner_pair.second)) {
                    // segment is full, we've to stash the overflow at the end of our active segment
                    --active_pair.second;
                    swap(*active_pair.first, *active_pair.second);
                    ++block_begin[partner_index].overflow;
                } else {
                    // normal swap
                    swap(*partner_pair.first, *active_pair.first);
                    partner_pair.first++;
                    __builtin_prefetch(&*partner_pair.first);
                }
            }
        }

        for (auto p : pairs) {
            block_begin->placed_end = p.first;
            block_begin->stash_begin = p.second;
            ++block_begin;
        }
    }

    static BlockDescriptor compact_segment(block_iterator seg_begin, block_iterator seg_end) {
        auto read_block_it = seg_end - 1;
        auto write_block_it = seg_begin;

        auto compacted_descriptor = [&] {
            BlockDescriptor compacted(seg_begin->begin, (seg_end - 1)->end);
            compacted.placed_end = write_block_it->placed_end;
            compacted.stash_begin = write_block_it->placed_end;
            compacted.overflow = std::accumulate(
                seg_begin, seg_end, 0ull, [](size_t s, const auto &b) { return s + b.overflow; });
            return compacted;
        };

        for (; write_block_it != read_block_it; ++write_block_it) {
            while (write_block_it->has_stashed()) {
                while (!read_block_it->has_placed()) {
                    --read_block_it;
                    if (read_block_it == write_block_it)
                        return compacted_descriptor();
                }

                write_block_it->unstash_and_swap_into_placed(read_block_it->placed_item());
                read_block_it->stash_last_placed_item();
            }
        }

        return compacted_descriptor();
    }

    static void redistribute_stashes(block_iterator block_begin, block_iterator block_end) {
        for (auto block = block_begin + 1; block != block_end; ++block) {
            auto &block_to_left = *(block - 1);
            if (block_to_left.num_stashed() > block_to_left.overflow) {
                block->take_from_left_stash(block_to_left.num_stashed() - block_to_left.overflow,
                                            block_to_left);
            }
        }

        for (auto block = block_end - 2;; --block) {
            auto &block_to_right = *(block + 1);
            if (block_to_right.num_stashed() > block_to_right.overflow) {
                block->take_from_right_stash(block_to_right.num_stashed() - block_to_right.overflow,
                                             block_to_right);
            }

            if (block == block_begin)
                break;
        }
    }

    static Iter pull_stashes_to_front(block_iterator block_begin, block_iterator block_end) {
        auto buffer = block_begin->begin;

        for (auto read_from = block_begin; read_from != block_end; ++read_from) {
            auto &read_block = *read_from;

            for (auto r = read_block.stash_begin; r != read_block.end; ++r)
                std::swap(*r, *buffer++);
        }

        return buffer;
    }

    static void restore_stashes_from_front(block_iterator block_begin, block_iterator block_end,
                                           Iter buffer_end) {
        for (auto i = std::distance(block_begin, block_end) - 1; i >= 0; --i) {
            const auto &write_block = block_begin[i];

            for (auto j = std::distance(write_block.stash_begin, write_block.end) - 1; j >= 0; --j)
                std::swap(*--buffer_end, write_block.stash_begin[j]);
        }

        assert(buffer_end == block_begin->begin);
    }

    template <typename Gen>
    static void sequential_shuffle(Iter begin, Iter end, Gen &gen, unsigned max_num_segments,
                                   size_t cutoff) {
        auto n = static_cast<size_t>(std::distance(begin, end));
        if (n <= cutoff)
            return shuffle::fisher_yates(begin, end, gen);

        // compute num_segments
        const auto num_segments = std::min(
            max_num_segments, 1u << [&] {
                auto binary_depth = tlx::integer_log2_ceil(n / cutoff) + 1;
                auto max_depth_per_layer = tlx::integer_log2_ceil(max_num_segments);
                auto layers = tlx::div_ceil(binary_depth, max_depth_per_layer);
                return std::max(4u, tlx::div_ceil(binary_depth, layers));
            }());

        std::vector<BlockDescriptor> segments;
        segments.reserve(num_segments);

        auto prev_end = begin;
        for (unsigned i = 0; i < num_segments; ++i) {
            auto end = begin + (n * (i + 1) / num_segments);
            segments.emplace_back(prev_end, end);
            prev_end = end;
        }

        rough_scatter(segments.begin(), segments.end(), gen);
        redistribute_stashes(segments.begin(), segments.end());
        auto buffer_end = pull_stashes_to_front(segments.begin(), segments.end());
        sequential_shuffle(segments.front().begin, buffer_end, gen, num_segments, cutoff);
        restore_stashes_from_front(segments.begin(), segments.end(), buffer_end);

        for (auto &segment : segments) {
            sequential_shuffle(segment.begin, segment.end, gen, max_num_segments, cutoff);
        }
    }

    template <typename GenProv>
    static void parallel_shuffle(Iter begin, Iter end, GenProv &gen_prov, unsigned num_segments,
                                 size_t cutoff) {

        using block_vector = std::vector<BlockDescriptor>;

        const auto n = static_cast<size_t>(std::distance(begin, end));
        if (n <= 4 * cutoff) {
            auto &gen = gen_prov(0);
            return shuffle::fisher_yates(begin, end, gen);
        }

        if (gen_prov.num_threads() == 1)
            return sequential_shuffle(begin, end, gen_prov(0), num_segments, cutoff);

        std::vector<block_vector> blocks_of_thread;
        block_vector segments;
        const auto requested_num_segments = num_segments;

#pragma omp parallel num_threads(gen_prov.num_threads())
        {
            const auto num_threads = static_cast<unsigned>(omp_get_max_threads());
            const auto thread_id = static_cast<unsigned>(omp_get_thread_num());
            num_segments = std::max(std::min(num_segments, 3 * num_threads), num_segments);
            auto &gen = gen_prov();

#pragma omp single
            {
                blocks_of_thread.resize(num_threads);
                segments.resize(num_segments);
            }

            // compute data regions for this thread
            auto &descriptors = blocks_of_thread[thread_id];
            descriptors.reserve(num_segments);

            size_t thread_len = 0;
            for (unsigned seg_id = 0; seg_id < num_segments; ++seg_id) {
                auto b = n * (seg_id * num_threads + thread_id) / (num_threads * num_segments);
                auto e = n * (seg_id * num_threads + thread_id + 1) / (num_threads * num_segments);
                thread_len += e - b;
                descriptors.emplace_back(begin + b, begin + e);
            }

            rough_scatter(descriptors.begin(), descriptors.end(), gen);

            block_vector blocks;
            blocks.reserve(num_threads);

#pragma omp barrier

#pragma omp for schedule(dynamic, 1)
            for (unsigned seg_id = 0; seg_id < num_segments; ++seg_id) {
                blocks.clear();
                for (auto &b : blocks_of_thread)
                    blocks.emplace_back(b[seg_id]);

                segments[seg_id] = compact_segment(blocks.begin(), blocks.end());
            }

#pragma omp single
            {
                redistribute_stashes(segments.begin(), segments.end());
                auto buffer_end = pull_stashes_to_front(segments.begin(), segments.end());
                sequential_shuffle(segments.front().begin, buffer_end, gen, num_segments, cutoff);
                restore_stashes_from_front(segments.begin(), segments.end(), buffer_end);
            }

            // we must shuffle all elements including the middle one. so find the segment
            // that includes it
            auto end_seg = num_segments / 2;
            for(; end_seg < num_segments && segments[end_seg].begin < begin + n/2; ++end_seg);

#pragma omp for nowait
            for (unsigned seg_id = 0; seg_id < end_seg; ++seg_id)
                sequential_shuffle(segments[seg_id].begin, segments[seg_id].end, gen,
                                   requested_num_segments, cutoff);
        }
    }
};
} // namespace iss

template <typename Iter, typename Gen, typename Config = iss::Config<>>
void iss_shuffle(Iter begin, Iter end, Gen &gen,
                 unsigned num_segments = Config::kDefaultNumSegments,
                 size_t cutoff = Config::kDefaultCutOff) {
    iss::Impl<Iter, Config>::sequential_shuffle(begin, end, gen, num_segments, cutoff);
}

namespace parallel {

template <typename Iter, typename Config = iss::Config<>>
void iss_shuffle(Iter begin, Iter end, unsigned num_segments = Config::kDefaultNumSegments,
                 size_t cutoff = Config::kDefaultCutOff) {
    GeneratorProvider<std::mt19937_64> gen_prov;
    iss::Impl<Iter, Config>::parallel_shuffle(begin, end, gen_prov, num_segments, cutoff);
}

template <typename Iter, typename GenProv, typename Config = iss::Config<>>
void iss_shuffle(Iter begin, Iter end, GenProv &gen_prov,
                 unsigned num_segments = Config::kDefaultNumSegments,
                 size_t cutoff = Config::kDefaultCutOff) {
    iss::Impl<Iter, Config>::parallel_shuffle(begin, end, gen_prov, num_segments, cutoff);
}

} // namespace parallel
} // namespace shuffle

#endif // INPLACE_SCATTER_SHUFFLE_HPP
