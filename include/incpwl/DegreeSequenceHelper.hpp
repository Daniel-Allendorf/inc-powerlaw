#pragma once
#ifndef UNIFORM_PLD_SAMPLING_DEGREE_SEQUENCE_HELPER_H
#define UNIFORM_PLD_SAMPLING_DEGREE_SEQUENCE_HELPER_H


#include <iostream>
#include <random>
#include <vector>

#include "defs.hpp"

namespace incpwl {

    std::vector<count> generate_degree_sequence(std::mt19937_64 &gen, std::size_t n, double gamma, count min_degree = 1, count max_degree = 0);
    std::vector<count> read_degree_sequence(std::istream &input, bool require_sorted = true);
    bool is_degree_sequence_graphical(std::vector<count>& deg_seq);
}

#endif
