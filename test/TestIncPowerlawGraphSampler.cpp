#include <incpwl/IncPowerlawGraphSampler.hpp>

#include <boost/math/distributions/chi_squared.hpp>
#include <gtest/gtest.h>

using namespace incpwl;

class TestIncPowerlawGraphSampler : public ::testing::Test {};

// Helper class to enhance testability of the sampler
class TestableIncPowerlawGraphSampler : public IncPowerlawGraphSampler {
public:
    explicit TestableIncPowerlawGraphSampler(const std::vector<count> &degree_sequence, double gamma)
            : IncPowerlawGraphSampler(degree_sequence, gamma) {
    }

    bool is_trivial_case() const {
        return M2_ < M1_;
    }

    void set_h(node h) {
        h_ = h;
        H1_ = 0; H2_ = 0; H3_ = 0; H4_ = 0;
        for (node i = 0; i < num_nodes_; ++i) {
            count degree = degree_sequence_[i];
            if (i < h_) {
                H1_ += ordered_choices(degree, 1);
                H2_ += ordered_choices(degree, 2);
                H3_ += ordered_choices(degree, 3);
                H4_ += ordered_choices(degree, 4);
            }
        }
        L2_ = M2_ - H2_;
        L3_ = M3_ - H3_;
        L4_ = M4_ - H4_;
        B2_ = 0; B3_ = 0;
        for (count i = h_; i < std::min(h_ + Delta_, num_nodes_); ++i) {
            count degree = degree_sequence_[i];
            B2_ += ordered_choices(degree, 2);
        }
        for (count i = h_; i < std::min(h_ + Delta_, num_nodes_); ++i) {
            count degree = degree_sequence_[i];
            B3_ += ordered_choices(degree, 3);
        }
    }

    void enable_explicit_bl_GV0_calculation() {
        explicit_bl_GV0_calculation_enabled_ = true;
    }
};

/*
 * Tests for basic correctness.
 */

// Declare helper function for comparing the degrees of generated graphs to a sequence
std::string check_degrees(const AdjacencyList& graph, const std::vector<count>& degree_sequence);

// Test degrees in a trivial case
TEST_F(TestIncPowerlawGraphSampler, Degrees1) {
    std::mt19937_64 gen(0);
    std::vector<count> degree_sequence(100, 1);
    TestableIncPowerlawGraphSampler sampler(degree_sequence, 3);
    for (int t = 0; t < 10000; ++t) {
        auto graph = sampler.sample(gen);
        auto result = check_degrees(graph, degree_sequence);
        if (!result.empty()) FAIL() << result;
    }
}

// Test degrees in a case where only heavy loops are possible.
TEST_F(TestIncPowerlawGraphSampler, Degrees2) {
    std::mt19937_64 gen(0);
    std::vector<count> degree_sequence = {4, 1, 1, 1, 1};
    TestableIncPowerlawGraphSampler sampler(degree_sequence, 2.88103);
    ASSERT_FALSE(sampler.is_trivial_case());
    for (int t = 0; t < 10000; ++t) {
        auto graph = sampler.sample(gen);
        auto result = check_degrees(graph, degree_sequence);
        if (!result.empty()) FAIL() << result;
    }
}

// Test degrees in a case where only heavy loops and heavy multiple edges are possible.
TEST_F(TestIncPowerlawGraphSampler, Degrees3) {
    std::mt19937_64 gen(0);
    std::vector<count> degree_sequence = {6, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    TestableIncPowerlawGraphSampler sampler(degree_sequence, 2.88103);
    ASSERT_FALSE(sampler.is_trivial_case());
    for (int t = 0; t < 10000; ++t) {
        auto graph = sampler.sample(gen);
        auto result = check_degrees(graph, degree_sequence);
        if (!result.empty()) FAIL() << result;
    }
}

// Test degrees in a case where all types of non-simple edges are possible.
TEST_F(TestIncPowerlawGraphSampler, Degrees4) {
    std::mt19937_64 gen(0);
    std::vector<count> degree_sequence = {8, 5, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1};
    TestableIncPowerlawGraphSampler sampler(degree_sequence, 2.88103);
    ASSERT_FALSE(sampler.is_trivial_case());
    for (int t = 0; t < 10000; ++t) {
        auto graph = sampler.sample(gen);
        auto result = check_degrees(graph, degree_sequence);
        if (!result.empty()) FAIL() << result;
    }
}

/*
 * Uniformity tests.
 */

// Declare some helper functions
std::size_t ordered_choices(std::size_t n, std::size_t k);
std::size_t choices(std::size_t n, std::size_t k);
std::string graph_to_string(const AdjacencyList& A);
std::unordered_map<std::string, count> sample_graphs_and_count(std::mt19937_64& gen,
                                                               IncPowerlawGraphSampler& sampler,
                                                               std::size_t iterations);
std::unordered_map<std::string, count> merge_graph_counts(const std::unordered_map<std::string, count>& A,
                                                          const std::unordered_map<std::string, count>& B);
std::string check_sample_uniformity(const std::unordered_map<std::string, count>& counts,
                                    std::size_t samples,
                                    std::size_t bins,
                                    double alpha);

// Test phase 1 heavy multiple-edge reduction
TEST_F(TestIncPowerlawGraphSampler, Uniformity1a) {
    std::mt19937_64 gen(0);
    std::vector<count> degree_sequence = {4, 4, 1, 1, 1, 1, 1, 1};
    TestableIncPowerlawGraphSampler sampler(degree_sequence, 2.88103);
    sampler.set_h(2);
    ASSERT_FALSE(sampler.is_trivial_case());
    std::size_t expected_graphs = 20;
    std::size_t iterations = expected_graphs * 100000;
    auto graph_counts = sample_graphs_and_count(gen, sampler, iterations);
    ASSERT_EQ(graph_counts.size(), expected_graphs);
    double alpha = 0.1;
    auto result = check_sample_uniformity(graph_counts, iterations, expected_graphs, alpha);
    if (!result.empty()) FAIL() << result;
}

// Test phase 2 heavy loop reduction.
TEST_F(TestIncPowerlawGraphSampler, Uniformity1b) {
    std::mt19937_64 gen(0);
    std::vector<count> degree_sequence = {4, 1, 1, 1, 1, 1, 1};
    TestableIncPowerlawGraphSampler sampler(degree_sequence, 2.88103);
    ASSERT_FALSE(sampler.is_trivial_case());
    std::size_t n = degree_sequence.size();
    std::size_t expected_graphs = choices(n - 1, 4);
    std::size_t iterations = expected_graphs * 100000;
    auto graph_counts = sample_graphs_and_count(gen, sampler, iterations);
    ASSERT_EQ(graph_counts.size(), expected_graphs);
    double alpha = 0.1;
    auto result = check_sample_uniformity(graph_counts, iterations, expected_graphs, alpha);
    if (!result.empty()) FAIL() << result;
}

/*
 * Test small sequences with high probability of light non-simple edges.
 */
TEST_F(TestIncPowerlawGraphSampler, Uniformity2a) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::vector<count> degree_sequence = {3, 3, 3, 2, 1};
    TestableIncPowerlawGraphSampler sampler(degree_sequence, 2.88103);
    ASSERT_FALSE(sampler.is_trivial_case());
    sampler.enable_explicit_bl_GV0_calculation();
    std::size_t expected_graphs = 3;
    std::size_t iterations = expected_graphs * 100000;
    auto graph_counts = sample_graphs_and_count(gen, sampler, iterations);
    ASSERT_EQ(graph_counts.size(), expected_graphs);
    double alpha = 0.1;
    auto result = check_sample_uniformity(graph_counts, iterations, expected_graphs, alpha);
    if (!result.empty()) FAIL() << result;
}

TEST_F(TestIncPowerlawGraphSampler, Uniformity2b) {
    std::mt19937_64 gen(0);
    std::vector<count> degree_sequence = {3, 3, 3, 2, 2, 1};
    TestableIncPowerlawGraphSampler sampler(degree_sequence, 2.88103);
    ASSERT_FALSE(sampler.is_trivial_case());
    sampler.enable_explicit_bl_GV0_calculation();
    std::size_t expected_graphs = 27;
    std::size_t iterations = expected_graphs * 100000;
    auto graph_counts = sample_graphs_and_count(gen, sampler, iterations);
    ASSERT_EQ(graph_counts.size(), expected_graphs);
    double alpha = 0.1;
    auto result = check_sample_uniformity(graph_counts, iterations, expected_graphs, alpha);
    if (!result.empty()) FAIL() << result;
}

TEST_F(TestIncPowerlawGraphSampler, Uniformity2c) {
    std::mt19937_64 gen(0);
    std::vector<count> degree_sequence = {4, 4, 4, 4, 3, 3};
    TestableIncPowerlawGraphSampler sampler(degree_sequence, 2.88103);
    ASSERT_FALSE(sampler.is_trivial_case());
    sampler.enable_explicit_bl_GV0_calculation();
    std::size_t expected_graphs = 18;
    std::size_t iterations = expected_graphs * 50000;
    auto graph_counts = sample_graphs_and_count(gen, sampler, iterations);
    ASSERT_EQ(graph_counts.size(), expected_graphs);
    double alpha = 0.1;
    auto result = check_sample_uniformity(graph_counts, iterations, expected_graphs, alpha);
    if (!result.empty()) FAIL() << result;
}

/*
 * Test some mixed cases with both heavy and light multiple edges.
 */
TEST_F(TestIncPowerlawGraphSampler, Uniformity3a) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::vector<count> degree_sequence = {3, 3, 3, 2, 1};
    TestableIncPowerlawGraphSampler sampler(degree_sequence, 2.88103);
    sampler.set_h(2);
    ASSERT_FALSE(sampler.is_trivial_case());
    sampler.enable_explicit_bl_GV0_calculation();
    std::size_t expected_graphs = 3;
    std::size_t iterations = expected_graphs * 100000;
    auto graph_counts = sample_graphs_and_count(gen, sampler, iterations);
    ASSERT_EQ(graph_counts.size(), expected_graphs);
    double alpha = 0.1;
    auto result = check_sample_uniformity(graph_counts, iterations, expected_graphs, alpha);
    if (!result.empty()) FAIL() << result;
}

TEST_F(TestIncPowerlawGraphSampler, Uniformity3b) {
    std::mt19937_64 gen(0);
    std::vector<count> degree_sequence = {4, 4, 4, 4, 3, 3};
    TestableIncPowerlawGraphSampler sampler(degree_sequence, 2.88103);
    sampler.set_h(2);
    ASSERT_FALSE(sampler.is_trivial_case());
    sampler.enable_explicit_bl_GV0_calculation();
    std::size_t expected_graphs = 18;
    std::size_t iterations = expected_graphs * 50000;
    auto graph_counts = sample_graphs_and_count(gen, sampler, iterations);
    ASSERT_EQ(graph_counts.size(), expected_graphs);
    double alpha = 0.1;
    auto result = check_sample_uniformity(graph_counts, iterations, expected_graphs, alpha);
    if (!result.empty()) FAIL() << result;
}

/*
 * Test with some larger sequences.
 * The number of possible graphs is too big for uniformity testing, but check that every graph is generated once.
 */
TEST_F(TestIncPowerlawGraphSampler, Uniformity4a) {
    std::mt19937_64 gen(0);
    std::vector<count> degree_sequence = {4, 2, 1, 1, 1, 1, 1, 1, 1, 1};
    TestableIncPowerlawGraphSampler sampler(degree_sequence, 2.88103);
    ASSERT_FALSE(sampler.is_trivial_case());
    std::size_t n = degree_sequence.size();
    std::size_t expected_graphs = choices(n - 2, 3) * (n - 5) * 3 +
                                  choices(n - 2, 4) * choices(n - 6, 2);
    std::size_t iterations = expected_graphs * 10;
    auto graph_counts = sample_graphs_and_count(gen, sampler, iterations);
    ASSERT_EQ(graph_counts.size(), expected_graphs);
}

TEST_F(TestIncPowerlawGraphSampler, Uniformity4b) {
    std::mt19937_64 gen(0);
    std::vector<count> degree_sequence = {6, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    TestableIncPowerlawGraphSampler sampler(degree_sequence, 2.88103);
    ASSERT_FALSE(sampler.is_trivial_case());
    std::size_t n = degree_sequence.size();
    std::size_t expected_graphs = choices(n - 2, 5) * choices(n - 7, 3) +
                                  choices(n - 2, 6) * choices(n - 8, 4);
    std::size_t iterations = expected_graphs * 20;
    auto graph_counts = sample_graphs_and_count(gen, sampler, iterations);
    ASSERT_EQ(graph_counts.size(), expected_graphs);
}

TEST_F(TestIncPowerlawGraphSampler, Uniformity4c) {
    std::mt19937_64 gen(1);
    std::vector<count> degree_sequence = {5, 4, 4, 3, 3, 2, 2, 1};
    TestableIncPowerlawGraphSampler sampler(degree_sequence, 2.88103);
    ASSERT_FALSE(sampler.is_trivial_case());
    sampler.enable_explicit_bl_GV0_calculation();
    std::size_t expected_graphs = 703;
    std::size_t iterations = expected_graphs * 10;
    auto graph_counts = sample_graphs_and_count(gen, sampler, iterations);
    ASSERT_EQ(graph_counts.size(), expected_graphs);
}

TEST_F(TestIncPowerlawGraphSampler, Uniformity4d) {
    std::mt19937_64 gen(0);
    std::vector<count> degree_sequence = {8, 5, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1};
    TestableIncPowerlawGraphSampler sampler(degree_sequence, 2.88103);
    ASSERT_FALSE(sampler.is_trivial_case());
    std::size_t expected_graphs = 9486;
    std::size_t iterations = expected_graphs * 10;
    auto graph_counts = sample_graphs_and_count(gen, sampler, iterations);
    ASSERT_EQ(graph_counts.size(), expected_graphs);
}

/*
 * Implementation of helper functions.
 */
std::string check_degrees(const AdjacencyList& graph, const std::vector<count>& degree_sequence) {
    if (graph.size() != degree_sequence.size())
        return "Expected graph with " +
               std::to_string(degree_sequence.size()) +
               " nodes but graph with " +
               std::to_string(graph.size()) +
               " nodes was generated.";
    std::size_t n = graph.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (graph[i].size() != degree_sequence[i])
            return "Expected node " +
                   std::to_string(i) +
                   " to have degree " +
                   std::to_string(degree_sequence[i]) +
                   " but had degree " +
                   std::to_string(graph[i].size()) +
                   ".";
    }
    return "";
}

std::string graph_to_string(const AdjacencyList& A) {
    std::string As;
    std::size_t N = A.size();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i; j < N; ++j) {
            if (A[i].find(j) != A[i].end()) {
                As += "1";
            } else {
                As += "0";
            }
        }
    }
    return As;
}

std::unordered_map<std::string, count> sample_graphs_and_count(std::mt19937_64& gen,
                                                               IncPowerlawGraphSampler& sampler,
                                                               std::size_t iterations) {
    std::unordered_map<std::string, count> graph_counts;
    for (std::size_t i = 0; i < iterations; ++i) {
        auto graph = sampler.sample(gen);
        auto graph_string = graph_to_string(graph);
        auto graph_count_iter = graph_counts.find(graph_string);
        if (graph_count_iter == graph_counts.end()) {
            graph_counts[graph_string] = 1;
        } else {
            graph_count_iter->second++;
        }
    }
    return graph_counts;
}

std::unordered_map<std::string, count> merge_graph_counts(const std::unordered_map<std::string, count>& A,
                                                          const std::unordered_map<std::string, count>& B) {
    std::unordered_map<std::string, count> merged_counts = A;
    for (auto [graph, count] : B) {
        assert(A.find(graph) != A.end());
        merged_counts[graph] += count;
    }
    return merged_counts;
}

std::string check_sample_uniformity(const std::unordered_map<std::string, count>& counts,
                                    std::size_t samples,
                                    std::size_t bins,
                                    double alpha) {
    double chi_squared = 0.;
    for (const auto& [graph, bin_count] : counts) {
        double p = 1. / bins;
        count expected_count = samples * p;
        if (bin_count != expected_count) {
            count delta = bin_count < expected_count ? expected_count - bin_count : bin_count - expected_count;
            if (delta >= (std::numeric_limits<double>::max() / delta))
                return "Overflow error. Too many iterations?";
            chi_squared += std::pow(delta, 2) / expected_count;
        }
    }
    double combined_p_value = chi_squared <= 0. ? 1. :
                              cdf(complement(boost::math::chi_squared(bins - 1), chi_squared));
    if (combined_p_value < alpha) {
        return "The combined p-value was " +
               std::to_string(combined_p_value) +
               " but the test requires a probability of at least " +
               std::to_string(alpha) +
               ".";
    }
    return "";
}

std::size_t ordered_choices(std::size_t n, std::size_t k) {
    std::size_t r = 1;
    for (std::size_t i = 0; i < k; ++i) {
        r *= (n - i);
    }
    return r;
}

std::size_t choices(std::size_t n, std::size_t k) {
    std::size_t r = ordered_choices(n, k);
    for (std::size_t i = 1; i <= k; ++i) {
        r /= i;
    }
    return r;
}