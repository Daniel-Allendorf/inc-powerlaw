///! This class is a stripped down version of NetworKit::PowerlawDegreeSequence
///! https://networkit.github.io/; CODE UNDER MIT LICENSE

#ifndef NETWORKIT_GENERATORS_POWERLAW_DEGREE_SEQUENCE_HPP_
#define NETWORKIT_GENERATORS_POWERLAW_DEGREE_SEQUENCE_HPP_

#include <vector>
#include <random>

#include <incpwl/defs.hpp>


namespace incpwl {

class PowerlawDegreeSequence {
public:
    PowerlawDegreeSequence(count minDeg, count maxDeg, double gamma);

    void run();

    std::vector<count> getDegreeSequence(count numNodes, std::mt19937_64&) const;
    count getDegree(std::mt19937_64&) const;

private:
    count minDeg, maxDeg;
    double gamma;
    std::vector<double> cumulativeProbability;
};

}

#endif
