#pragma once
#ifndef UNIFORM_PLD_SAMPLING_CONFIGURATION_MODEL_H
#define UNIFORM_PLD_SAMPLING_CONFIGURATION_MODEL_H

#include <functional>
#include <random>
#include <vector>

#include <incpwl/defs.hpp>
#include <incpwl/AdjacencyVector.hpp>

namespace incpwl {

class ConfigurationModel {
public:
    explicit ConfigurationModel(const std::vector<count> &degree_sequence);

    void enable_parallel_sampling(std::function<bool()> keep_going);

    void generate(AdjacencyVector &adj_vec, std::mt19937_64 &gen, bool parallel);

private:
    const std::vector<count> &degree_sequence_;
    std::vector<uint64_t> points_;
    std::vector<uint64_t> pairs64_;
    std::function<bool()> keep_going_;

    static constexpr size_t kDistrBits = 8;
    static constexpr size_t kDistrLines = size_t(1) << kDistrBits;
    std::array<size_t, kDistrLines + 1> distr_begins;

    void generate_parallel(AdjacencyVector &adj_vec, std::mt19937_64 &gen);
    void generate_sequential(AdjacencyVector &adj_vec, std::mt19937_64 &gen);


};

}

#endif // UNIFORM_PLD_SAMPLING_CONFIGURATION_MODEL_H
