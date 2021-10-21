#include <incpwl/AdjacencyVector.hpp>
#include <range/v3/numeric.hpp>

namespace incpwl {

[[nodiscard]] uint64_t AdjacencyVector::fingerprint() const noexcept {
    return ranges::accumulate(edges(), 0llu, [] (auto s, std::pair<node, node> e) {return s + (e.first + 1) * (e.second + 1);});
}

// convert from an to obsolete representation
[[nodiscard]] AdjacencyList AdjacencyVector::to_adjacency_list() const {
    AdjacencyList res(num_nodes());
    for (auto u : nodes()) {
        res[u] = ranges::to<std::unordered_set<node>>(neighbors(u));
        assert(res[u].size() == degree(u)); // check for multi-edges
    }

    return res;
}

AdjacencyVector AdjacencyVector::from_adjacency_list(const AdjacencyList &from) {
    auto degrees = from | ranges::views::transform([](auto &set) { return set.size(); }) | ranges::to<std::vector<count>>();

    AdjacencyVector res(degrees);
    for (auto &&[u, set] : ranges::views::enumerate(from)) {
        for (auto v : set)
            if (u <= v)
                res.add_new_edge(u, v);
    }

    return res;
}

void AdjacencyVector::write_metis(std::ostream &os) const {
    auto num_edges = ranges::accumulate(degrees(), 0);

    os << num_nodes() << " " << num_edges << "\n";
    for(auto u : nodes())
        for(auto [i, v] : ranges::views::enumerate(neighbors(u)))
            os << (i ? " " : "\n") << v;
    os << "\n";
}

}