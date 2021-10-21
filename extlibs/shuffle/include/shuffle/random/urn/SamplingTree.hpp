#pragma once

#include <array>
#include <tlx/math.hpp>

#include <shuffle/RandomBits.hpp>
#include <shuffle/UniformRange.hpp>

#include <immintrin.h>

namespace shuffle {
namespace urn {

template <unsigned Log2Lines, typename CountT = unsigned>
class SamplingTree {
public:
    constexpr static auto kLines = unsigned(1) << Log2Lines;
    constexpr static auto kLog2Lines = Log2Lines;

    using label_type = unsigned;
    using count_type = CountT;

    SamplingTree() { std::fill(weights_.begin(), weights_.end(), 0); }

    void bulk_insert(label_type line, count_type count) {
        assert(line < kLines);

        total_weight_ += count;

        line += kLines;
        for (unsigned i = 0; i < kLog2Lines; ++i) {
            auto parent = line / 2;
            auto is_right_child = (line % 2);

            auto &parent_weight = weights_[parent - 1];
            parent_weight += count * !is_right_child;

            line = parent;
        }
    }

    // not needed; just for compatiblity
    void bulk_complete() {}

    template <typename Gen>
    label_type sample(Gen &gen) {
        assert(total_weight_);

        auto random_value =
            static_cast<count_type>(nearlydivisionless(uint64_t(total_weight_), gen));
        unsigned index = 1;

        for (unsigned i = 0; i < kLog2Lines; ++i) {
            auto &node_weight = weights_[index - 1];

            bool go_right = (node_weight <= random_value);
            random_value -= go_right * node_weight;
            node_weight -= !go_right;

            index += index + go_right;
        }

        --total_weight_;
        return index - kLines;
    }

    count_type total_weight() const { return total_weight_; }

private:
    count_type total_weight_{0};
    std::array<count_type, kLines> weights_;
};

} // namespace urn
} // namespace shuffle
