#include <incpwl/AdjacencyVector.hpp>

#include <gtest/gtest.h>
#include <range/v3/all.hpp>
#include <random>

using namespace incpwl;

static std::pair<node, node> normalized_edge(node u, node v) {
    return {std::min(u, v), std::max(u, v)};
}

TEST(TestAdjacencyVector, FromAdjList) {
    auto edges = std::set<std::pair<node, node>>{{0, 0},
                                                 {0, 1},
                                                 {0, 2},
                                                 {0, 3},
                                                 {1, 2},
                                                 {1, 3},
                                                 {3, 4},
                                                 {4, 4}};
    auto res = AdjacencyVector::from_adjacency_list({{0, 1, 2, 3}, // 0
                                                     {0, 2, 3}, // 1
                                                     {0, 1}, // 2
                                                     {0, 1, 4}, // 3
                                                     {3, 4}} // 4
    );

    ASSERT_EQ(res.num_nodes(), 5);
    ASSERT_EQ(res.degree(0), 4);
    {
        std::vector<count> ref{4, 3, 2, 3, 2};
        ASSERT_TRUE(ranges::all_of(ranges::views::zip(res.degrees(), ref), [](auto a) { return a.first == a.second; }));
    }

    size_t num_edges = 0;
    for (auto[u, v] : ranges::views::cartesian_product(res.nodes(), res.nodes())) {
        ASSERT_EQ(res.has_edge(u, v), !!edges.count(normalized_edge(u, v))) << u << " " << v;
        num_edges += res.has_edge(u, v);
    }
    ASSERT_EQ(num_edges, 14);

    for (auto e : res.edges()) {
        ASSERT_TRUE(edges.count(e));
        edges.erase(e);
    }
    ASSERT_TRUE(edges.empty());

    ASSERT_EQ(res.neighbors(2) | ranges::to<std::vector<node>> | ranges::actions::sort, (std::vector<node>{0, 1}));
    ASSERT_EQ(res.neighborhoods()[3] | ranges::to<std::vector<node>> | ranges::actions::sort, (std::vector<node>{0, 1, 4}));
}

TEST(TestAdjacencyVector, RandomAccesses) {
    std::mt19937_64 gen(0);

    for (unsigned n = 2; n < 16; ++n) {
        std::uniform_int_distribution<count> node_distr{0, n - 1};

        std::vector<count> max_degrees(n);
        for (auto &x: max_degrees) x = node_distr(gen);
        AdjacencyVector vec(max_degrees);

        std::set<std::pair<node, node>> edges;
        std::vector<count> degrees(n);

        for (unsigned iter = 0; iter < 2000 * n; ++iter) {
            if (gen() % 5000 == 0) {
                // clear
                vec.clear();
                edges.clear();
                ranges::fill(degrees, 0);
                edges.clear();

                ASSERT_TRUE(ranges::all_of(vec.degrees(), [](auto d) { return !d; }));
            }


            auto u = node_distr(gen);
            auto v = node_distr(gen);
            std::pair<node, node> norm{std::min(u, v), std::max(u, v)};

            ASSERT_EQ(vec.degree(u), degrees[u]);
            ASSERT_EQ(vec.degree(v), degrees[v]);

            ASSERT_EQ(ranges::distance(vec.neighbors(u)), degrees[u]);
            ASSERT_EQ(ranges::distance(vec.neighbors(v)), degrees[v]);

            if (edges.count(norm)) {
                ASSERT_TRUE(vec.has_edge(u, v));
                ASSERT_TRUE(vec.has_edge(v, u));

                vec.remove_edge(u, v);
                degrees[u]--;
                degrees[v] -= (u != v);
                edges.erase(norm);

                ASSERT_FALSE(vec.has_edge(u, v));
                ASSERT_FALSE(vec.has_edge(v, u));
            } else {
                ASSERT_FALSE(vec.has_edge(u, v));
                ASSERT_FALSE(vec.has_edge(v, u));

                if (degrees[u] >= max_degrees[u] || degrees[v] >= max_degrees[v])
                    continue;

                degrees[u]++;
                degrees[v] += (u != v);
                vec.add_unique_edge(u, v);
                edges.insert(norm);

                ASSERT_TRUE(vec.has_edge(u, v));
                ASSERT_TRUE(vec.has_edge(v, u));
            }
        }
    }
}

TEST(TestAdjacencyVector, CountMultiplicity) {
    std::mt19937_64 gen(0);

    for (unsigned n = 2; n < 16; ++n) {
        std::uniform_int_distribution<count> node_distr{0, n - 1};

        std::vector<count> max_degrees(n);
        for (auto &x: max_degrees) x = node_distr(gen);
        AdjacencyVector vec(max_degrees);

        std::map<std::pair<node, node>, count> edges;
        std::vector<count> degrees(n);

        count max_m = 0;
        for (unsigned i = 0; i < n * n; ++i) {
            const auto u = node_distr(gen);
            const auto v = node_distr(gen);

            if (edges[normalized_edge(u, v)] && 0 == gen() % 4) {
                // edge exists, so we might as well delete it ;)
                count m = edges[normalized_edge(u, v)];
                bool all = (gen() % 4 == 0);

                vec.remove_edge(u, v, all);
                edges[normalized_edge(u, v)] -= all ? m : 1;
                continue;
            }

            auto m = std::uniform_int_distribution<count>{0u, std::min(max_degrees[u] - vec.degree(u), max_degrees[v] - vec.degree(v))}(
                gen);
            if (!m) continue;

            max_m = std::max(max_m, (edges[normalized_edge(u, v)] += m));

            while (m--) {
                vec.add_new_edge(u, v);
            }
        }

        for (node u = 0; u < n; ++u) {
            for (node v = 0; v < n; ++v) {
                auto ref_count = edges[normalized_edge(u, v)];
                ASSERT_EQ(vec.has_edge(u, v), !!ref_count);
                ASSERT_EQ(vec.count_edge(u, v), ref_count) << n << " " << u << " " << v;
            }
        }

        bool failed = false;
        vec.for_each([&](node u, node v, count m) {
            if (!m) {
                std::cerr << "m = 0";
                failed = true;
                return;
            }

            if (m != edges[normalized_edge(u, v)]) {
                std::cerr << "Mismatch m=" << m << " edges[normalized_edge(" << u << ", " << v << ")=" << edges[normalized_edge(u, v)]
                          << "\n";
                failed = true;
                return;
            }

            edges.erase(normalized_edge(u, v));
        });

        ASSERT_FALSE(failed);
    }
}

TEST(TestAdjacencyVector, SampleLoop) {
    std::vector<count> degrees{2, 1};

    std::mt19937_64 gen;

    for (unsigned slack = 0; slack < 3; ++slack) {
        AdjacencyVector vec{degrees, slack};
        vec.add_unique_edge(0, 0);
        vec.add_unique_edge(1, 0);

        for (bool loop_counts_twice : {false, true}) {
            count num_loop = 0;
            for (unsigned i = 0; i < 100'000; ++i) {
                auto e = vec.sample(gen, loop_counts_twice);
                if (e.first == e.second) {
                    ASSERT_EQ(e.first, 0);
                    num_loop++;
                } else {
                    ASSERT_EQ(e.first + e.second, 1);
                }
            }

            if (loop_counts_twice) {
                ASSERT_GE(num_loop, 40000);
                ASSERT_LE(num_loop, 60000);
            } else {
                ASSERT_GE(num_loop, 25000);
                ASSERT_LE(num_loop, 40000);
            }
        }
    }
}

TEST(TestAdjacencyVector, SampleMulti) {
    for (count m : {2, 4, 6, 8, 10}) {
        assert(m % 2 == 0);

        std::vector<count> degrees(2 + m, 1);
        degrees[0] = m;
        degrees[1] = m;

        AdjacencyVector vec{degrees, 0};

        for (count i = 0; i < m; ++i) {
            vec.add_new_edge(0, 1);
        }

        for (count i = 0; i < m; i += 2) {
            vec.add_new_edge(2 + i, 3 + i);
        }

        std::mt19937_64 gen;

        count num_multi = 0;
        count num_direction = 0;
        for (unsigned i = 0; i < 100'000; ++i) {
            auto e = vec.sample(gen);

            num_direction += (e.first > e.second);

            if (e.first > e.second)
                std::swap(e.first, e.second);

            if (e.first == 0 && e.second == 1) {
                ASSERT_EQ(e.first, 0);
                num_multi++;
            } else {
                ASSERT_EQ(e.first + 1, e.second);
            }
        }

        ASSERT_GE(num_direction, 40000);
        ASSERT_LE(num_direction, 60000);

        ASSERT_GE(num_multi, 60000);
        ASSERT_LE(num_multi, 70000);
    }
}
