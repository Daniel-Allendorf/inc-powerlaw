#pragma once
#ifndef UNIFORM_PLD_SAMPLING_ADJACENCYVECTOR_H
#define UNIFORM_PLD_SAMPLING_ADJACENCYVECTOR_H

#include <iosfwd>
#include <cassert>
#include <vector>
#include <algorithm>
#include <utility>
#include <unordered_set>

#include <tlx/define.hpp>
#include <range/v3/view.hpp>
#include <range/v3/algorithm.hpp>

#include "defs.hpp"

namespace incpwl {
using AdjacencyList = std::vector<std::unordered_set<node>>;

class ConfigurationModel;

class AdjacencyVector {
    using boundary_t = std::pair<count, count>;
    friend ConfigurationModel;

public:
    AdjacencyVector() : boundaries_{0} {}

    AdjacencyVector(AdjacencyVector &&) = default;

    AdjacencyVector(const std::vector<count> &degree_sequence, count slack_added_to_each_node = 0) {
        boundaries_.reserve(degree_sequence.size() + 1);
        count num_edges = 0;
        for (auto d : degree_sequence) {
            boundaries_.emplace_back(num_edges, num_edges);
            num_edges += d + std::max(d, slack_added_to_each_node);
        }
        boundaries_.emplace_back(num_edges, num_edges); // sentinel to avoid boundary checks
        adj_vec_.resize(num_edges);
    }

    AdjacencyVector &operator=(AdjacencyVector &&) = default;

private: // copy is expensive, so make using it explicit via copy()
    AdjacencyVector(const AdjacencyVector &) = default;

    AdjacencyVector &operator=(const AdjacencyVector &) = default;

public:
    //! return copy of the data structure
    AdjacencyVector copy() { return {*this}; }

    //! returns the number of nodes supported
    [[nodiscard]] count num_nodes() const noexcept {
        return boundaries_.empty() ? 0 : boundaries_.size() - 1;
    }

    //! returns maximum number of neighbors that can be assigned to this node
    [[nodiscard]] count capacity(node u) const noexcept {
        assert(u < num_nodes());
        return boundaries_[u + 1].first - boundaries_[u].first;
    }

    //! returns degree of node u, i.e. number of neighbors
    [[nodiscard]] count degree(node u) const noexcept {
        assert(u < num_nodes());
        return boundaries_[u].second - boundaries_[u].first;
    }

    //! sequence from 0 to n-1
    [[nodiscard]] auto nodes() const noexcept {
        return ranges::views::ints(node(0), static_cast<node>(num_nodes()));
    }

    //! returns a sequence of all degrees
    [[nodiscard]] auto degrees() const noexcept {
        return nodes() | ranges::views::transform([&](node u) { return degree(u); });
    }

    //! view of all neighbors of node u
    [[nodiscard]] auto neighbors(node u) const noexcept {
        assert(u < num_nodes());
        return adj_vec_ | ranges::views::slice(boundaries_[u].first, boundaries_[u].second);
    }

    //! view of all neighbors of node u
    [[nodiscard]] auto neighbors(node u) noexcept {
        assert(u < num_nodes());
        return adj_vec_ | ranges::views::slice(boundaries_[u].first, boundaries_[u].second);
    }

    //! view of all neighbors of node u
    [[nodiscard]] auto unique_neighbors(node u) const noexcept {
        assert(u < num_nodes());
        return neighbors(u) | ranges::views::unique;
    }

    //! view of views, i.e. [neighbors(0), ...]
    [[nodiscard]] auto neighborhoods() const noexcept {
        return nodes() | ranges::views::transform([&](node u) { return neighbors(u); });
    }

    //! return view of std::pair<node, node> that contains all edges in an arbitrary order
    //! all edges (u, v) hold u <= v
    [[nodiscard]] auto edges() const noexcept {
        return nodes() | ranges::views::for_each([&](node u) { //
            return neighbors(u) | ranges::views::filter([=](auto v) { return u <= v; }) |
                   ranges::views::transform([=](node v) { return std::pair<node, node>{u, v}; });
        });
    }

    //! returns true if edge (u, v) exists
    [[nodiscard]] bool has_edge(node u, node v) const noexcept {
        if (u < v) std::swap(u, v);
        auto neigh = neighbors(u);
        auto it = ranges::lower_bound(neigh, v);
        return it != neigh.end() && *it == v;
    }

    //! returns the multiplicity of an edge
    [[nodiscard]] count count_edge(node u, node v) const noexcept {
        if (u < v) std::swap(u, v);
        auto neigh = neighbors(u);

        auto it = ranges::lower_bound(neigh, v);
        auto it2 = it;
        for (; it2 != neigh.end() && *it2 == v; ++it2);

        return std::distance(it, it2);
    }

    //! only add if edge does not yet exists; returns true if there was a change
    bool add_unique_edge(node u, node v) noexcept {
        if (!has_edge(u, v)) {
            assert(!has_edge(v, u));
            add_new_edge(u, v);
            return true;
        }
        assert(has_edge(v, u));

        return false;
    }

    //! add edges (u,v) and (v,u)
    //! precondition: edges do not yet exist
    void add_new_edge(node u, node v) noexcept {
        add_half_edge(u, v);
        if (TLX_LIKELY(u != v)) {
            add_half_edge(v, u);
        }
    }

    //! remove edge (u, v) and (v, u); if all_occurrences == true all instances of an edge
    //! with possibly multiplicity > 1 are removed
    //! returns the number of instances (u, v) was removed
    size_t remove_edge(node u, node v, bool all_occurrences = false) noexcept {
        auto erase_half_edge = [&](node u, node v) { // delete v in u
            auto neigh = neighbors(u);
            auto it = ranges::find(neigh, v);

            assert(it != neigh.end());
            auto it2 = it + 1;

            if (all_occurrences) {
                while (it2 != neigh.end() && *it2 == v)
                    ++it2;
            }

            std::move(it2, neigh.end(), it);

            boundaries_[u].second -= (it2 - it);
            assert(boundaries_[u].first <= boundaries_[u].second);

            return static_cast<size_t>(it2 - it);
        };

        auto count = erase_half_edge(u, v);
        if (TLX_LIKELY(u != v))
            erase_half_edge(v, u);

        return count;
    }

    //! erase all neighbors while keeping capacities
    void clear() noexcept {
        for (auto &b : boundaries_)
            b.second = b.first;
    }


    //! iterate through all edges stored; for each edge {u, v} with multiplicity m
    //! the callback cb(u, v, m) is invoked; each edge is either called with u <= v but not for v >= u
    template<typename Callback, bool OneWay = true>
    void for_each(Callback cb) const noexcept {
        for (node u : nodes()) {
            for_each<Callback, OneWay>(u, cb);
        }
    }

    template<typename Callback>
    void for_each_twoway(Callback &&cb) const noexcept {
        for_each<Callback, false>(std::forward<Callback>(cb));
    }

    template<typename Callback, bool OneWay = true>
    void for_each(node u, Callback cb) const noexcept {
        auto neigh = neighbors(u);

        auto it = neigh.begin();
        while (it != neigh.end()) {
            auto v = *it;
            if (OneWay && v > u) break;

            auto it2 = std::find_if_not(it, neigh.end(), [v](node x) { return x == v; });
            cb(u, v, static_cast<size_t>(std::distance(it, it2)));

            it = it2;
        }
    }

    // returns a random edge in a random orientation
    [[nodiscard]] std::pair<node, node> sample(std::mt19937_64 &gen, bool loop_counts_twice = true) const noexcept {
        while (true) {
            auto rand = std::uniform_int_distribution<size_t>{0u, boundaries_.back().first - 1}(gen);
            auto it = std::lower_bound(boundaries_.begin(), boundaries_.end(), rand, [](boundary_t b, size_t r) {
                return b.second <= r;
            });

            if (it->first <= rand && rand < it->second) {
                const auto u = static_cast<node>(std::distance(boundaries_.begin(), it));
                const auto v = adj_vec_[rand];

                if (loop_counts_twice && u != v) {
                    // since loops are stored only once while every other edge is stored twice
                    // (once per direction), we reject a non-loop with probability of 1/2
                    if (gen() % 2)
                        continue;
                }

                return {u, v};
            }
        }
    }

    [[nodiscard]] uint64_t fingerprint() const noexcept;

// convert from an to obsolete representation
    [[nodiscard]] AdjacencyList to_adjacency_list() const;
    static AdjacencyVector from_adjacency_list(const AdjacencyList &from);

    void write_metis(std::ostream&) const;

private:
    std::vector<boundary_t> boundaries_;
    std::vector<node> adj_vec_;

    void add_half_edge(node u, node v) noexcept {
        assert(degree(u) < capacity(u));

        auto begin = adj_vec_.begin() + boundaries_[u].first;
        auto it = adj_vec_.begin() + boundaries_[u].second;
        boundaries_[u].second++;

        // insertion sort
        if (begin == it) {
            *begin = v;
            return;
        }

        *it = v;
        while (true) {
            auto prev = std::prev(it);
            if (*prev <= *it) break;
            std::swap(*prev, *it);
            if (prev == begin) break;
            it = prev;
        }

        assert(ranges::is_sorted(neighbors(u)));
    }

};

}

#endif // UNIFORM_PLD_SAMPLING_ADJACENCYVECTOR_H
