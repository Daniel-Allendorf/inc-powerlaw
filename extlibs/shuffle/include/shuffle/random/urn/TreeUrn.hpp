#pragma once

#include <random>
#include <vector>
#include <tlx/math.hpp>

#include <shuffle/UniformRange.hpp>

namespace shuffle {
namespace urns {

class TreeUrn {
public:
    using value_type = int64_t;
    using color_type = size_t;

    explicit TreeUrn(color_type number_of_colors)
        : number_of_colors_(number_of_colors),
          first_leaf_(tlx::round_up_to_power_of_two(number_of_colors_)),
          tree_storage_(first_leaf_, 0) {
        tree_1indexed_ = tree_storage_.data() - 1;
    }

    void add_balls(color_type col, value_type n = 1) {
        number_of_balls_ += n;

        auto i = first_leaf_ + col;
        do {
            auto step = [&] {
                const auto parent = i / 2;
                const auto isLeft = !(i & 1);
                tree_1indexed_[parent] += isLeft * n;
                i = parent;
            };

            if (i >= 8) {
                step();
                step();
                step();
            }

            step();
        } while (i);
    }

    void remove_balls(color_type col, value_type n) { add_balls(col, -n); }

    template <typename Generator>
    std::pair<color_type, value_type> remove_random_ball_with_index(Generator &&gen) noexcept {
        assert(!empty());
        //auto value = std::uniform_int_distribution<value_type>{0, number_of_balls_ - 1}(gen);
        auto value = static_cast<value_type>(nearlydivisionless(static_cast<uint64_t>(number_of_balls_ - 1), gen));

        size_t i = 1;

        do {
            auto step = [&] {
                auto &leftWeight = tree_1indexed_[i];

                auto toRight = (value >= leftWeight);
                value -= toRight * leftWeight;
                leftWeight -= !toRight;

                i = 2 * i + toRight;
            };

            if (8 * i + 7 < first_leaf_) {
                step();
                step();
                step();
            }

            step();
        } while (i < first_leaf_);

        --number_of_balls_;
        const auto col = i - first_leaf_;

        return {col, value};
    }

    template <typename Generator>
    color_type remove_random_ball(Generator &&gen) noexcept {
        return remove_random_ball_with_index(std::forward<Generator>(gen)).first;
    }

    value_type number_of_balls() const noexcept { return number_of_balls_; }

    color_type number_of_colors() const noexcept { return number_of_colors_; }

    bool empty() const noexcept { return !number_of_balls(); }

    void clear() {
        number_of_balls_ = 0;
        std::fill(tree_storage_.begin(), tree_storage_.end(), 0);
    }

private:
    value_type number_of_balls_{0};
    size_t number_of_colors_{0};
    size_t first_leaf_;

    std::vector<value_type> tree_storage_;

    value_type *tree_1indexed_;

};

} // namespace urns
} // namespace shuffle