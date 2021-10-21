#define _REENTRANT
#include <incpwl/ConfigurationModel.hpp>

#include <omp.h>
#include <algorithm>

#include <tlx/die.hpp>

#include <shuffle/algorithms/FisherYates.hpp>
#include <shuffle/algorithms/InplaceScatterShuffle.hpp>
#include <shuffle/random/RandomBits.hpp>

#include <range/v3/view.hpp>
#include <range/v3/algorithm.hpp>
#include <range/v3/numeric.hpp>

#include <ska_sort.hpp>

#include <ips4o/ips4o.hpp>

namespace incpwl {


ConfigurationModel::ConfigurationModel(const std::vector<count> &degree_sequence) //
    : degree_sequence_(degree_sequence), points_{degree_sequence_           //
                                                 | ranges::views::enumerate //
                                                 | ranges::views::for_each([](auto &&x) {
    return ranges::views::repeat_n(static_cast<node>(x.first), x.second);
}) //
                                                 | ranges::to<std::vector>}, //
      keep_going_([]() { return true; }) //
{
    assert(points_.size() % 2 == 0);

    const size_t n_bits = tlx::integer_log2_ceil(degree_sequence_.size());

    if (degree_sequence_.size() > std::numeric_limits<uint32_t>::max() - 1)
        throw std::runtime_error("This configuration model only supports 32 bit node ids");

    if (2 * n_bits >= 32) {
        ranges::fill(distr_begins, 0);
        auto shift = std::max<size_t>(n_bits, kDistrBits) - kDistrBits;
        for (auto[i, d] : ranges::views::enumerate(degree_sequence))
            distr_begins[1 + (i >> shift)] += d;

        for (size_t i = 0; i != kDistrLines; ++i)
            distr_begins[i + 1] += distr_begins[i];
    }
}

void ConfigurationModel::enable_parallel_sampling(std::function<bool()> keep_going) {
    keep_going_ = keep_going;
}

void ConfigurationModel::generate(AdjacencyVector &adj_vec, std::mt19937_64 &gen, bool parallel) {

    assert(adj_vec.num_nodes() == degree_sequence_.size());
    assert(ranges::accumulate(adj_vec.degrees(), 0u) == 0);

    if (parallel && points_.size() > static_cast<size_t>(16 * omp_get_num_threads())) {
        generate_parallel(adj_vec, gen);
    } else {
        generate_sequential(adj_vec, gen);
    }
}

void ConfigurationModel::generate_parallel(AdjacencyVector &adj_vec, std::mt19937_64 &gen) {
    shuffle::GeneratorProvider gen_prov(gen);
    shuffle::parallel::iss_shuffle(points_.begin(), points_.end(), gen_prov);

    constexpr uint64_t kMask = (1llu << 32) - 1;

    const auto mid_size = points_.size() / 2;
#pragma omp parallel for
    for(long long i=0; i < static_cast<long long>(mid_size); ++i) {
        auto u = points_[i] & kMask;
        auto v = points_[mid_size + i] & kMask;
        points_[i] = (u << 32) | v;
        points_[mid_size+i] = (v << 32) | u | ((kMask << 32) * (u == v));
    }

    ips4o::parallel::sort(points_.begin(), points_.end());

    if (!keep_going_())
        return;

    auto first_node_of = [&] (uint64_t idx) {return points_[idx] >> 32;};
    auto second_node_of = [&] (uint64_t idx) {return points_[idx] & kMask;};

#pragma omp parallel
    {
        const auto num_threads = omp_get_num_threads();
        const auto thread_id = omp_get_thread_num();

        auto my_point_begin = (points_.size() * thread_id) / num_threads;
        auto my_point_end = (points_.size() * (thread_id + 1)) / num_threads;

        // find first own node
        if (my_point_begin > 0) {
            auto node_before = first_node_of(my_point_begin - 1);
            for (; my_point_begin < my_point_end && first_node_of(my_point_begin) == node_before; ++my_point_begin);
        }

        // find last own node
        if (my_point_begin != my_point_end && my_point_end < points_.size()) {
            auto last_own_node = first_node_of(my_point_end - 1);
            for (; my_point_end != points_.size() && first_node_of(my_point_end) == last_own_node; ++my_point_end);
        }

        for (size_t i = my_point_begin; i < my_point_end; ++i) {
            if (TLX_UNLIKELY(i % 32 == 0 && !keep_going_()))
                break;

            auto u = first_node_of(i);
            auto v = second_node_of(i);

            if (TLX_UNLIKELY(u == kMask))
                break;

            adj_vec.adj_vec_[adj_vec.boundaries_[u].second++] = v;
        }
    }

#ifndef NDEBUG
    if (!keep_going_()) return;

    for (node u : adj_vec.nodes()) {
        die_unless(ranges::is_sorted(adj_vec.neighbors(u)));
        die_unequal(adj_vec.degree(u) + adj_vec.count_edge(u, u), degree_sequence_[u]);
    }
#endif
}


void ConfigurationModel::generate_sequential(AdjacencyVector &adj_vec, std::mt19937_64 &gen) {
    // shuffle half of the points; the idea is the sample for each point iid with prob of 0.5
    // whether it goes into the left or right half of the sequence. then we shuffle the larger
    // one.
    {
        shuffle::FairCoin coin;
        auto it = std::partition(points_.begin(), points_.end(), [&](auto) -> bool { return coin(gen); });

        if (!keep_going_()) return;

        if (it < points_.begin() + points_.size() / 2)
            shuffle::fisher_yates(it, points_.end(), gen);
        else
            shuffle::fisher_yates(points_.begin(), it, gen);
    }

    if (!keep_going_()) return;

    if (pairs64_.size() < points_.size())
        pairs64_.resize(points_.size());

    const size_t n_bits = tlx::integer_log2_ceil(degree_sequence_.size());

    if (2 * n_bits < 32) {
        auto end = pairs64_.begin();
        {
            auto k = points_.size() / 2;
            for (size_t i = 0; i != k; ++i) {
                auto u = points_[i];
                auto v = points_[k + i];

                *(end++) = (static_cast<uint64_t>(u) << n_bits) | v;
                *end = (static_cast<uint64_t>(v) << n_bits) | u;
                end += (u != v); // add self-loops only once
            }
        }

        ska_sort(pairs64_.begin(), end, [](auto x) -> uint32_t { return x; });
        if (!keep_going_()) return;

        const auto mask = (1llu << n_bits) - 1;
        for (auto it = pairs64_.begin(); it != end; ++it) {
            auto u = *it >> n_bits;
            auto v = *it & mask;
            adj_vec.adj_vec_[adj_vec.boundaries_[u].second++] = v;
        }
    } else {
        std::array<size_t, kDistrLines> distr_ends;
        std::copy_n(distr_begins.begin(), kDistrLines, distr_ends.begin());

        auto line_shift = std::max<size_t>(n_bits, kDistrBits) - kDistrBits;
        {
            auto mid = points_.size() / 2;
            for (size_t i = 0; i != mid; ++i) {
                auto u = points_[i];
                auto v = points_[mid + i];

                auto line1 = u >> line_shift;
                auto line2 = v >> line_shift;
                assert(line1 < kDistrLines);
                assert(line2 < kDistrLines);

                pairs64_[distr_ends[line1]++] = (static_cast<uint64_t>(u) << n_bits) | v;
                pairs64_[distr_ends[line2]] = (static_cast<uint64_t>(v) << n_bits) | u;

                distr_ends[line2] += (u != v); // add self-loops only once
            }
        }

        const auto mask = (1llu << n_bits) - 1;
        for (size_t line = 0; line < kDistrLines; ++line) {
            if (!keep_going_()) return;
            auto begin = pairs64_.begin() + distr_begins[line];
            auto end = pairs64_.begin() + distr_ends[line];

            // we only sort lower 32bits, so u's might not be fully sorted, but that's okay since
            // we're doing a distribution sort below
            ska_sort(begin, end, [](auto x) -> uint32_t { return x; });

            for (; begin != end; ++begin) {
                auto u = *begin >> n_bits;
                auto v = *begin & mask;
                adj_vec.adj_vec_[adj_vec.boundaries_[u].second++] = v;
            }
        }
    }

    // check correctness
#ifndef NDEBUG
    if (!keep_going_()) return;

    for (node u : adj_vec.nodes()) {
        die_unless(ranges::is_sorted(adj_vec.neighbors(u)));
        die_unequal(adj_vec.degree(u) + adj_vec.count_edge(u, u), degree_sequence_[u]);
    }
#endif
}

}