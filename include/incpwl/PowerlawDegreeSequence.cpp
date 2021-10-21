///! This class is a stripped down version of PowerlawDegreeSequence
///! https://networkit.github.io/; CODE UNDER MIT LICENSE

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <incpwl/PowerlawDegreeSequence.hpp>

namespace incpwl {

PowerlawDegreeSequence::PowerlawDegreeSequence(count minDeg, count maxDeg, double gamma) : minDeg(minDeg), maxDeg(maxDeg), gamma(gamma) {
    if (minDeg > maxDeg) throw std::runtime_error("Error: minDeg must not be larger than maxDeg");
    if (gamma > -1) throw std::runtime_error("Error: gamma must be lower than -1");
}

void PowerlawDegreeSequence::run() {
    cumulativeProbability.clear();
    cumulativeProbability.reserve(maxDeg - minDeg + 1);

    double sum = 0;

    for (double d = maxDeg; d >= minDeg; --d) {
        sum += std::pow(d, gamma);
        cumulativeProbability.push_back(sum);
    }

    for (double &prob : cumulativeProbability) {
        prob /= sum;
    }

    cumulativeProbability.back() = 1.0;
}

std::vector<count> PowerlawDegreeSequence::getDegreeSequence(count numNodes, std::mt19937_64 &gen) const {
    std::vector<count> degreeSequence;

    degreeSequence.reserve(numNodes);
    count degreeSum = 0;

    for (count i = 0; i < numNodes; ++i) {
        degreeSequence.push_back(getDegree(gen));
        degreeSum += degreeSequence.back();
    }

    if (degreeSum % 2 != 0) {
        (*std::max_element(degreeSequence.begin(), degreeSequence.end()))--;
    }

    return degreeSequence;
}

count PowerlawDegreeSequence::getDegree(std::mt19937_64 &gen) const {
    auto prob = std::uniform_real_distribution<double>{0, 1}(gen);
    auto lb = std::lower_bound(cumulativeProbability.begin(), cumulativeProbability.end(), prob);
    return maxDeg - std::distance(cumulativeProbability.begin(), lb);
}

}
