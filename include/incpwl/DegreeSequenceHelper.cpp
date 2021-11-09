#include <incpwl/DegreeSequenceHelper.hpp>

#include <cassert>
#include <cmath>
#include <algorithm>
#include <functional>

#include <tlx/define/likely.hpp>

#include <incpwl/PowerlawDegreeSequence.hpp>

namespace incpwl {

std::vector<count> generate_degree_sequence(std::mt19937_64 &gen, std::size_t n, double gamma, count min_degree, count max_degree) {
    if (gamma <= 0)
        throw std::range_error("gamma has to be strictly larger than 1");

    if (!max_degree)
        max_degree = std::pow(n, 1. / (gamma - 1));

    std::vector<count> degree_sequence;
    PowerlawDegreeSequence sequence_generator(min_degree, max_degree, -gamma);
    sequence_generator.run();
    while (true) {
        degree_sequence = sequence_generator.getDegreeSequence(n, gen);
        if (is_degree_sequence_graphical(degree_sequence))
            break;
    }
    std::sort(degree_sequence.begin(), degree_sequence.end(), std::greater<count>());
    return degree_sequence;
}

std::vector<count> read_degree_sequence(std::istream &input, bool require_sorted) {
    std::vector<count> degree_sequence;
    std::string line;
    while (getline(input, line) && !line.empty()) {
        count degree = std::stoul(line);
        degree_sequence.push_back(degree);
    }

    if (require_sorted && !std::is_sorted(degree_sequence.begin(), degree_sequence.end(), std::greater<count>()))
        throw std::runtime_error("Degree sequence is not sorted");

    return degree_sequence;
}


///! This function is derived from NetworKit::StaticDegreeSequence::isRealizable()
///! https://networkit.github.io/; CODE UNDER MIT LICENSE
bool is_degree_sequence_graphical(std::vector<count> &seq) {
    count n = seq.size();

    // First inequality
    count deg_sum = 0;
    for (count i = 0; i < n; ++i) {
        if (TLX_UNLIKELY(seq[i] >= n)) {
            return false;
        }
        deg_sum += seq[i];
    }

    if (deg_sum % 2 != 0) {
        return false;
    }

    std::vector<count> partialSeqSum(n + 1);
    std::copy(seq.cbegin(), seq.cend(), partialSeqSum.begin());
    sort(partialSeqSum.begin(), partialSeqSum.end(), std::greater<count>());
    for (size_t i = n - 1; i--;) { // not using std::partial_sum as unclear whether input/output may be identical
        partialSeqSum[i] += partialSeqSum[i + 1];
    }

    auto degreeOf = [&](size_t i) {
        assert(i < n);
        return partialSeqSum[i] - partialSeqSum[i + 1];
    };

    deg_sum = 0;
    for (count j = 0; j < n; ++j) {
        deg_sum += degreeOf(j);
        count min_deg_sum = 0;

        size_t sumFrom = j + 1;
        if (sumFrom < n && degreeOf(sumFrom) >= j + 1) {
            // find the first element right of j that has a value less or equal to j
            const auto it = std::lower_bound(partialSeqSum.data() + sumFrom, partialSeqSum.data() + n, j,
                                             [](const count &x, const count j) { return x - *(&x + 1) > j; });
            sumFrom = std::distance(partialSeqSum.data(), it);
            min_deg_sum += (j + 1) * (sumFrom - j - 1);
        }

        if (sumFrom != n)
            min_deg_sum += partialSeqSum[sumFrom];

        if (TLX_UNLIKELY(deg_sum > (j + 1) * j + min_deg_sum)) {
            return false;
        }
    }

    return true;
}

}