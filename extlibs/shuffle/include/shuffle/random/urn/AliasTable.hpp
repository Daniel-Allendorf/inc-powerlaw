#pragma once

#include <array>
#include <tlx/math.hpp>

#include <shuffle/RandomBits.hpp>
#include <shuffle/UniformRange.hpp>

namespace shuffle {
namespace urn {

template <unsigned Log2Lines, typename CountT = size_t, bool UseEndgame = false>
class AliasTable {
public:
    constexpr static auto kLines = unsigned(1) << Log2Lines;
    constexpr static auto kLog2Lines = Log2Lines;

    using label_type = unsigned;
    using count_type = CountT;

    AliasTable() { std::fill(scratch_.begin(), scratch_.end(), 0); }

    void bulk_insert(label_type line, count_type count) {
        assert(insertion_mode_);
        scratch_[line] += count;
        total_weight_ += count;
    }

    void bulk_complete() {
#ifndef NDEBUG
        assert(insertion_mode_);
        insertion_mode_ = false;
#endif
        if (UseEndgame && total_weight_ <= kLines) {
            switch_to_endgame();
        } else {
            build_from_scratch();
        }
    }

    template <typename Gen>
    label_type sample(Gen &gen) {
        assert(!insertion_mode_);

        if (UseEndgame && total_weight_ <= kLines)
            return sample_endgame(gen);

        return sample_table(gen);
    }

    count_type total_weight() const { return total_weight_; }

private:
#ifndef NDEBUG
    bool insertion_mode_{true};
#endif

    RandomBits<uint64_t> random_bits_;

    struct Line {
        std::array<CountT, 2> counts;
        std::array<label_type, 2> labels;
    };

    std::array<Line, kLines> table_;
    std::array<count_type, kLines> scratch_;

    count_type total_weight_{0};
    count_type max_weight_;
    count_type refresh_if_below_;

    void update_scratch() {
        std::fill(scratch_.begin(), scratch_.end(), 0);
        for (const Line &line : table_) {
            for (unsigned i = 0; i < 2; ++i) {
                assert(line.labels[i] < kLines);
                scratch_[line.labels[i]] += line.counts[i];
            }
        }
    }

    void refresh() {
        count_type min_value = std::numeric_limits<count_type>::max();
        count_type max_value = 0;
        for (const Line &line : table_) {
            const auto w = line.counts[0] + line.counts[1];
            min_value = std::min(min_value, w);
            max_value = std::max(max_value, w);
        }
        //std::cout << min_value << " " << max_value << " " << (1.0 * min_value / max_value) << '\n';

        if (min_value * 16 < max_value * 15) {
            update_scratch();
            return build_from_scratch();
        }

        max_weight_ = max_value;
        update_refresh_limit();
    }

    void transfer_scratch_to_table() {
        for (unsigned i = 0; i < kLines; ++i) {
            table_[i].counts = {scratch_[i], 0};
            table_[i].labels = {i, 0};
        }
    }

    void build_from_scratch() {
        transfer_scratch_to_table();

        max_weight_ = tlx::div_ceil(total_weight_, kLines);
        update_refresh_limit();

        auto it_lower = scratch_.begin();
        auto it_higher = scratch_.begin() + kLines - 1;

        for (size_t i = 0; i < kLines; ++i) {
            const auto count = table_[i].counts[0];
            if (count < max_weight_) {
                *(it_lower++) = i;

            } else if (count > max_weight_) {
                *(it_higher--) = i;
            }
        }

        if (++it_higher == scratch_.end() || --it_lower == scratch_.begin())
            return;

        while (true) {
            Line &line_above = table_[*it_higher];
            Line &line_below = table_[*it_lower];
            const auto weight_to_shift =
                std::min(max_weight_ - line_below.counts[0], line_above.counts[0] - max_weight_);

            line_below.counts[1] = weight_to_shift;
            line_below.labels[1] = line_above.labels[0];

            line_above.counts[0] -= weight_to_shift;

            if (line_above.counts[0] == max_weight_) {
                ++it_higher;
                if (it_higher == scratch_.end())
                    break;
            }

            if (TLX_UNLIKELY(it_lower == scratch_.begin()))
                break;
            --it_lower;
        }
    }

    void update_refresh_limit() {
        refresh_if_below_ = std::max<count_type>(max_weight_ * 0.9, max_weight_ - 10);
    }

    void switch_to_endgame() {
        assert(total_weight_ <= kLines);
        assert(std::accumulate(scratch_.begin(), scratch_.end(), 0u) == total_weight_);

        size_t idx = 0;
        for (unsigned i = 0; i < kLines; ++i) {
            while (scratch_[i]--) {
                table_[idx / 2].counts[idx % 2] = i;
                ++idx;
            }
        }
    }

    template <typename Gen>
    label_type sample_endgame(Gen &gen) {
        auto idx = nearlydivisionless(total_weight_ - 1, gen);
        auto result = static_cast<label_type>(table_[idx / 2].counts[idx % 2]);
        --total_weight_;
        table_[idx / 2].counts[idx % 2] = table_[total_weight_ / 2].counts[total_weight_ % 2];
        return result;
    }

    template <typename Gen>
    label_type sample_table(Gen &gen) {
        while (true) {
            auto line_idx = random_bits_.get<kLog2Lines>(gen);
            auto &line = table_[line_idx];
            auto weight = nearlydivisionless(max_weight_, gen);

            const auto line_weight = line.counts[0] + line.counts[1];
            if (TLX_UNLIKELY(line_weight <= weight))
                continue; // rejection sampling

            bool second = (line.counts[0] <= weight);
            line.counts[0] -= !second;
            line.counts[1] -= second;
            total_weight_--;
            const auto sampled = line.labels[second];

            if (TLX_UNLIKELY(UseEndgame && total_weight_ == kLines)) {
                update_scratch();
                switch_to_endgame();
            } else if (TLX_UNLIKELY(line_weight == refresh_if_below_)) {
                refresh();
            }

            return sampled;
        }
    }
};

} // namespace urn
} // namespace shuffle
