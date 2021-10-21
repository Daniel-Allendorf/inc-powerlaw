#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <shuffle/UniformRange.hpp>
#include <shuffle/RandomBits.hpp>

namespace shuffle {
namespace urn {

template <unsigned Log2Lines, typename CountT = size_t>
class RejectionTable {
public:
    constexpr static auto kLines = unsigned(1) << Log2Lines;
    constexpr static auto kLog2Lines = Log2Lines;

    using label_type = unsigned;
    using count_type = CountT;

    RejectionTable() { std::fill(weights_.begin(), weights_.end(), 0); }

    void bulk_insert(label_type line, count_type count) noexcept {
        weights_[line] += count;
        total_weight_ += count;
        rebuild_at_weight_ = total_weight_;
    }

    void bulk_complete() const noexcept {}

    template <typename Gen>
    label_type sample(Gen &gen) {
        if (total_weight_ <= rebuild_at_weight_)
            rebuild();

        total_weight_--;

        if (uniform_sampling_) {
            auto line = random_bits_.get<Log2Lines>(gen);
            --weights_[line];
            return line;
        }

        while (true) {
            const auto random = shuffle::nearlydivisionless(sampling_max_, s_mod_s_, gen);
            const auto line = random % kLines;
            const auto weight = random / kLines;

            if (TLX_LIKELY(weights_[line] > weight)) {
                --weights_[line];
                return line;
            }
        }
    }

    count_type total_weight() const noexcept { return total_weight_; }

private:
    count_type total_weight_{0};
    count_type rebuild_at_weight_{0};

    std::array<count_type, kLines> weights_;

    count_type sampling_max_;
    uint64_t s_mod_s_;
    bool uniform_sampling_{false};
    RandomBits<> random_bits_;

    void rebuild() {
        sampling_max_ = *std::max_element(weights_.begin(), weights_.end()) * kLines;
        s_mod_s_ = -static_cast<uint64_t>(sampling_max_) % static_cast<uint64_t>(sampling_max_);
        rebuild_at_weight_ = std::max(total_weight_ * 15 / 16, total_weight_ - 50 * kLines);

        //const auto min_v = *std::min_element(weights_.begin(), weights_.end());
        //uniform_sampling_ = (total_weight_ - rebuild_at_weight_ < min_v);
    }
};

} // namespace urn

} // namespace shuffle