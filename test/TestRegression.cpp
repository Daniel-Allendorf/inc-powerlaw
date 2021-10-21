#include <fstream>
#include <sstream>
#include <array>
#include <gtest/gtest.h>

#include <incpwl/IncPowerlawGraphSampler.hpp>
#include <incpwl/DegreeSequenceHelper.hpp>
using namespace incpwl;

class TestRegression : public ::testing::Test {};

constexpr int N_SEQS_GRAPHS[3][3] = {
        {1<<12, 25, 10},
        {1<<16, 10, 5},
        {1<<20, 3, 2}
};
constexpr double GAMMA = 2.88103;
constexpr const char* SEQS_PATH = "regression-testing/seq_";
constexpr const char* GRAPHS_PATH = "regression-testing/graph_";

std::vector<count> read_degree_sequence(std::size_t n, int id);
void write_degree_sequence(const std::vector<count>& degree_sequence, std::size_t n, int id);
void write_graph(const AdjacencyList & graph, std::size_t n, int seq_id, int graph_id);
AdjacencyList read_graph(std::size_t n, int seq_id, int graph_id);

/*
 * To run these tests, the working directory needs to contain a directory called "regression-testing".
 */

/*
 * Test if the current implementation produces the same graphs as a previous one for the same seed.
 * Run this after non-fuctional changes/refactoring.
 */
TEST_F(TestRegression, CompareGraphs) {
    std::mt19937_64 gen(0);
    for (auto [n, seqs, graphs] : N_SEQS_GRAPHS) {
        for (int i = 0; i < seqs; ++i) {
            auto degree_sequence = read_degree_sequence(n, i);
            IncPowerlawGraphSampler sampler(degree_sequence, GAMMA);
            for (int j = 0; j < graphs; ++j) {
                auto graph = sampler.sample(gen);
                auto expected_graph = read_graph(n, i, j);
                ASSERT_EQ(n, graph.size());
                ASSERT_EQ(n, expected_graph.size());
                for (int v = 0; v < n; ++v) {
                    ASSERT_TRUE(graph[v] == expected_graph[v]);
                }
            }
        }
    }
}

/*
 * Overwrite expected results with graphs output by the current implementation.
 * Run after functional changes.
 */
TEST_F(TestRegression, DISABLED_OverwriteGraphs) { // remove DISABLED_ prefix to run
    std::mt19937_64 gen(0);
    for (auto [n, seqs, graphs] : N_SEQS_GRAPHS) {
        for (int i = 0; i < seqs; ++i) {
            auto degree_sequence = incpwl::generate_degree_sequence(gen, n, GAMMA);
            write_degree_sequence(degree_sequence, n, i);
            IncPowerlawGraphSampler sampler(degree_sequence, GAMMA);
            for (int j = 0; j < graphs; ++j) {
                auto graph = sampler.sample(gen);
                write_graph(graph, n, i, j);
            }
        }
    }
}

void write_degree_sequence(const std::vector<count>& degree_sequence, std::size_t n, int seq_id) {
    std::string path = SEQS_PATH +
                       std::to_string(n) +
                       "_" +
                       std::to_string(seq_id);
    if (std::ofstream file{path}) {
        for (count degree : degree_sequence) {
            file << degree << "\n";
        }
    }
}

std::vector<count> read_degree_sequence(std::size_t n, int seq_id) {
    std::string path = SEQS_PATH +
                       std::to_string(n) +
                       "_" +
                       std::to_string(seq_id);
    std::vector<count> degree_sequence;
    if (std::ifstream file{path}) {
        std::string line;
        while (getline(file, line)) {
            count degree = std::stoul(line);
            degree_sequence.push_back(degree);
        }
    }
    return degree_sequence;
}

void write_graph(const AdjacencyList & graph, std::size_t n, int seq_id, int graph_id) {
    std::string path = GRAPHS_PATH +
                       std::to_string(n) +
                       "_" +
                       std::to_string(seq_id) +
                       "_" +
                       std::to_string(graph_id);
    if (std::ofstream file{path}) {
        for (auto neighbors : graph) {
            for (auto neighbor : neighbors) {
                file << neighbor << " ";
            }
            file << "\n";
        }
    }
}

AdjacencyList read_graph(std::size_t n, int seq_id, int graph_id) {
    std::string path = GRAPHS_PATH +
                       std::to_string(n) +
                       "_" +
                       std::to_string(seq_id) +
                       "_" +
                       std::to_string(graph_id);
    AdjacencyList graph;
    if (std::ifstream file{path}) {
        std::string line;
        while (getline(file, line)) {
            std::unordered_set<node> neighbors;
            std::istringstream iss(line);
            std::string word;
            while (getline(iss, word, ' ')) {
                node neighbor = std::stoul(word);
                neighbors.insert(neighbor);
            }
            graph.push_back(neighbors);
        }
    }
    return graph;
}