#include <algorithm>
//#undef NDEBUG // TODO enables assertions for release build, remove before distributing?
#include <cassert>
#include <boost/random.hpp>
#include <range/v3/algorithm.hpp>
#include <range/v3/numeric.hpp>

#include <incpwl/IncPowerlawGraphSampler.hpp>

#include <tlx/math.hpp>

namespace incpwl {
#if defined(UNIFORM_PLD_SAMPLING_LOG_ACCEPT_REJECT) || defined(LOG_EDGES)
#include <iostream>
#endif

#ifdef UNIFORM_PLD_SAMPLING_LOG_ACCEPT_REJECT
#include <incpwl/ScopedTimer.hpp>
incpwl::ScopedTimer rejection_timer;
#define ACCEPT do { \
            std::cout << "PASSED " << __LINE__ << "::" << __func__ \
                      << " after " << rejection_timer.elapsed() << "ms" << std::endl; \
            return true; \
        } while (0)
#define REJECT_IF(X) \
        if (X) do { \
            std::cout << "REJECT at " << __LINE__ << "::" << __func__ << "::" #X \
                      << " after " << rejection_timer.elapsed() << "ms" << std::endl; \
            return false; \
        } while (0); \
        else do { \
            std::cout << "ACCEPT at " << __LINE__ << "::" << __func__ << "::" #X \
                      << " after " << rejection_timer.elapsed() << "ms" << std::endl; \
        } while (0)
#define REJECT_UNLESS(X) \
        if (!(X)) do { \
            std::cout << "REJECT at " << __LINE__ << "::" << __func__ << "::!(" #X ")" \
                      << " after " << rejection_timer.elapsed() << "ms" << std::endl; \
            return false; \
        } while (0); \
        else do { \
            std::cout << "ACCEPT at " << __LINE__ << "::" << __func__ << "::" #X \
                      << " after " << rejection_timer.elapsed() << "ms" << std::endl; \
        } while (0)
#else
#define ACCEPT return true
#define REJECT_IF(X) if (X) return false
#define REJECT_UNLESS(X) if (!(X)) return false
#endif

IncPowerlawGraphSampler::IncPowerlawGraphSampler(const std::vector<count> &degree_sequence, double gamma)
    : degree_sequence_(degree_sequence),
      config_model_(degree_sequence),
      num_nodes_(degree_sequence.size()),
      log_num_nodes_(tlx::integer_log2_ceil(num_nodes_)),
      gamma_(gamma),
      M1_(0), M2_(0), M3_(0), M4_(0),
      H1_(0), H2_(0), H3_(0), H4_(0),
      L2_(0), L3_(0), L4_(0),
      A2_(0), B1_(0), B2_(0), B3_(0),
      keep_going_([](){ return true; }),
      start_new_iteration_([](){}),
      parallel_shufling_enabled_(false) {
    assert(num_nodes_ > 0);
    assert(std::is_sorted(degree_sequence_.cbegin(), degree_sequence_.cend(), std::greater<count>()));

    // set max degree Delta
    Delta_ = degree_sequence_.front();

    // find a suitable h, first we need a suitable delta
    double delta_lower_bound = 1./(2 * gamma_ - 3.);
    double delta_upper_bound = (2. - 3./(gamma_ - 1.))/(4. - gamma_);
    assert(delta_lower_bound > 0.);
    assert(delta_upper_bound > 0.);
    // allow lower values of gamma
    if (delta_lower_bound > delta_upper_bound) delta_lower_bound = delta_upper_bound;
    // we can pick any delta within the bounds
    delta_ = (delta_lower_bound + delta_upper_bound)/2.;
    // this gives us h, the number of "heavy" nodes
    h_ = std::floor(std::pow(static_cast<double>(num_nodes_), 1. - delta_*(gamma_ - 1.)));
    assert(h_ > 0);
    assert(h_ < num_nodes_);

    // find node with the first index that has degree 1
    first_degree1_node_ = ranges::lower_bound(degree_sequence_, count(1), std::greater<node>{}) -
                          degree_sequence_.begin();

    // compute upper bounds for the number of pairs, two-paths, three-stars etc.
    for (node i = 0; i < num_nodes_; ++i) {
        count degree = degree_sequence_[i];
        M1_ += ordered_choices(degree, 1);
        M2_ += ordered_choices(degree, 2);
        M3_ += ordered_choices(degree, 3);
        M4_ += ordered_choices(degree, 4);
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
    assert(M1_ % 2 == 0);

    // compute A2_, an upper bound on the number of two-paths starting at a given node where the first edge is simple
    // in the worst-case, the node is connected to the Delta highest-degree nodes
    for (count i = 0; i < Delta_; ++i) {
        count degree = degree_sequence_[i];
        A2_ += degree;
    }

    // compute B1_, an upper bound on the number of simple light one-blooms at a given node
    // in the worst case, the node is connected to the Delta highest-degree light nodes
    for (count i = h_; i < std::min(h_ + Delta_, num_nodes_); ++i) {
        count degree = degree_sequence_[i];
        B1_ += degree;
    }

    // compute B2_, an upper bound on the number of simple light two-blooms at a given node
    // in the worst case, the node is connected to the Delta highest-degree light nodes
    for (count i = h_; i < std::min(h_ + Delta_, num_nodes_); ++i) {
        count degree = degree_sequence_[i];
        B2_ += ordered_choices(degree, 2);
    }

    // compute B3_, an upper bound on the number of simple light three-blooms at a given node
    // in the worst case, the node is connected to the Delta highest-degree light nodes
    for (count i = h_; i < std::min(h_ + Delta_, num_nodes_); ++i) {
        count degree = degree_sequence_[i];
        B3_ += ordered_choices(degree, 3);
    }

    // reserve space in SamplingState
    state_.reserve(degree_sequence, num_nodes_);

    // disable debugging/testing flags by default
    explicit_bl_GV0_calculation_enabled_ = false;
}

void IncPowerlawGraphSampler::enable_parallel_sampling(std::function<bool()> keep_going,
                                                   std::function<void()> start_new_iteration) {
    keep_going_ = keep_going;
    start_new_iteration_ = start_new_iteration;
    config_model_.enable_parallel_sampling(keep_going_);
}

void IncPowerlawGraphSampler::enable_parallel_shuffling() {
    parallel_shufling_enabled_ = true;
}

AdjacencyVector IncPowerlawGraphSampler::sample_vector(std::mt19937_64& gen) {
    //TODO implement stop for degree-sequences that dont meet the conditions, better: check in the constructor?
    //TODO ^^ while it makes sense, consider an overwrite that disables this check (for tests and performance)
    while (true) {
#ifdef UNIFORM_PLD_SAMPLING_LOG_ACCEPT_REJECT
        rejection_timer.start();
#endif
        if (!keep_going_()) return {};
        start_new_iteration_();

        if (!keep_going_()) return {};
        generate_initial_pairing(gen);

        if (!keep_going_()) return {};
        initialize_stage_1();

        if (!keep_going_()) return {};
        if (!meets_stage_1_preconditions())
            continue;

        if (!keep_going_()) return {};
        if (!remove_heavy_multiple_edges_or_reject(gen))
            continue;

        if (!keep_going_()) return {};
        if (!remove_heavy_loops_or_reject(gen))
            continue;

        if (!keep_going_()) return {};
        initialize_stage_2_and_3();

        if (!keep_going_()) return {};
        if (!meets_stage_2_preconditions())
            continue;

        if (!keep_going_()) return {};
        if (!remove_light_loops_or_reject(gen))
            continue;

        if (!keep_going_()) return {};
        if (!remove_light_triples_or_reject(gen))
            continue;

        if (!keep_going_()) return {};
        if (!remove_light_doubles_or_reject(gen))
            continue;

        break;
    }
    return state_.graph.copy();
}

AdjacencyList IncPowerlawGraphSampler::sample(std::mt19937_64& gen) {
    return sample_vector(gen).to_adjacency_list();
}

void IncPowerlawGraphSampler::generate_initial_pairing(std::mt19937_64& gen) {
    // initialize state
    state_.reset();
    config_model_.generate(state_.graph, gen, parallel_shufling_enabled_);
}

void IncPowerlawGraphSampler::initialize_stage_1() {
    // update stage flag
    state_.stage = STAGE_1;

    // initialize edge lists and and edge-related quantities
    for(node u = 0; u < h_; ++u) {
        state_.graph.for_each(u, [&] (node, node v, count m) {
            if (!is_heavy(v))
                return;

            // determine type of edge
            const bool loop = u == v;
            const bool multiple_edge = !loop && m >= 2;

            // add to edge queue
            if (multiple_edge) {
                state_.heavy_multiple_edges.emplace(v, u, m);
            } else if (loop) {
                state_.heavy_loops.emplace(u, m);
            }

            // update number of points in heavy multiple edges and loops at u and v
            if (multiple_edge) {
                state_.heavy_multiple_edges_at[u] += m;
                state_.heavy_multiple_edges_at[v] += m;
            } else if (loop) {
                state_.heavy_loops_at[u] += 2 * m;
            }
        });
    }
}

bool IncPowerlawGraphSampler::meets_stage_1_preconditions() {
    if (M2_ < M1_) {
        REJECT_UNLESS(state_.heavy_multiple_edges.empty() && state_.heavy_loops.empty());
        ACCEPT;
    } else {
        integer eta_2_numerator = M2_ * M2_ * H1_;
        integer eta_2_denominator = M1_ * M1_ * M1_;
        integer sum_of_heavy_multiple_edge_multiplicities = 0;
        integer sum_of_heavy_loop_multiplicities = 0;
        for (node i = 0; i < h_; ++i) {
            // check condition m_ij I_m_ij>=2 W_ij <= eta d_i
            integer d_i = degree_sequence_[i];
            integer eta_2_numerator_d_i_2 = eta_2_numerator * d_i * d_i;
            for (node j = 0; j < h_; ++j) {
                if (i == j)
                    continue;
                integer m_ij = multiplicity_of(i, j);
                if (m_ij < 2)
                    continue;
                integer W_ij = state_.heavy_multiple_edges_at[i] + state_.heavy_loops_at[i] - m_ij;
                integer m_ij_2_W_ij_2 = m_ij * m_ij * W_ij * W_ij;
                REJECT_IF(m_ij_2_W_ij_2 > eta_2_numerator_d_i_2 / eta_2_denominator);
                sum_of_heavy_multiple_edge_multiplicities += m_ij;
            }
            // check condition m_ii W_i <= eta d_i
            integer m_ii = multiplicity_of(i, i);
            integer W_i = state_.heavy_multiple_edges_at[i];
            integer m_ii_2_W_i_2 = m_ii * m_ii * W_i * W_i;
            REJECT_IF(m_ii_2_W_i_2 >  eta_2_numerator_d_i_2 / eta_2_denominator);
            sum_of_heavy_loop_multiplicities += m_ii;
        }
        integer heavy_multiple_edge_limit = 4 * M2_ * M2_ / (M1_ * M1_);
        REJECT_IF(sum_of_heavy_multiple_edge_multiplicities > heavy_multiple_edge_limit);
        integer heavy_loop_limit = 4 * M2_ / M1_;
        REJECT_IF(sum_of_heavy_loop_multiplicities > heavy_loop_limit);
#ifdef LOG_EDGES
        std::cerr << "hl mul  : " << sum_of_heavy_loop_multiplicities << std::endl;
        std::cerr << "hl sum  : " << state_.heavy_loops.size() << std::endl;
        std::cerr << "hme mul : " << sum_of_heavy_multiple_edge_multiplicities << std::endl;
        std::cerr << "hme sum : " << state_.heavy_multiple_edges.size() << std::endl;
#endif
        ACCEPT;
    }
}

bool IncPowerlawGraphSampler::remove_heavy_multiple_edges_or_reject(std::mt19937_64 &gen) {
    while (!state_.heavy_multiple_edges.empty()) {
        if (!keep_going_()) return false;

        auto [i, j, m] = state_.heavy_multiple_edges.front();
        state_.heavy_multiple_edges.pop();

        // sample m random pairs and try to use them for an m-way switching
        for (std::size_t switched_pairs = 0; switched_pairs < m; ++switched_pairs) {
            // sample a pair
            auto [v1, v2] = sample_edge(gen);

            // f-reject if the switching is invalid
            // we can't create or remove other heavy multiple edges or loops
            REJECT_IF(is_heavy(v1) && is_heavy(v2));
            REJECT_IF(i == v1 || i == v2);
            REJECT_IF(is_heavy(v1) && has_edge(i, v1));
            REJECT_IF(j == v1 || j == v2);
            REJECT_IF(is_heavy(v2) && has_edge(j, v2));

            // switch with this pair
            remove_edge(v1, v2);
            add_edge(i, v1);
            add_edge(j, v2);
        }
        remove_heavy_multiple_edge(i, j, m);

        // b-rejection chance
        count d_i = degree_sequence_[i];
        count d_j = degree_sequence_[j];
        count W_ij = state_.heavy_multiple_edges_at[i] + state_.heavy_loops_at[i];
        count W_ji = state_.heavy_multiple_edges_at[j] + state_.heavy_loops_at[j];
        integer valid_and_invalid_switchings = ordered_choices(d_i - W_ij, m) *
                                               ordered_choices(d_j - W_ji, m);
        integer invalid_switchings_upper_bound = integer(m) * integer(h_) * integer(h_) *
                                                 ordered_choices(d_i - W_ij, m - 1) *
                                                 ordered_choices(d_j - W_ji, m - 1);
        integer b_ij_lower = std::max<integer>(valid_and_invalid_switchings - invalid_switchings_upper_bound, 1);
        integer b_ij = calculate_b_ij(i, j, m);
        assert(b_ij > 0);
        assert(b_ij_lower <= b_ij);
        boost::random::uniform_int_distribution<integer> b_rejection_dist(0, b_ij - 1);
        REJECT_IF(b_rejection_dist(gen) >= b_ij_lower);

        // with a certain probability we are done with this iteration, otherwise perform a inverse 1-way switching
        integer b_ij_1_upper = integer(d_i - W_ij) * integer(d_j - W_ji);
        integer f_ij_1_lower = std::max<integer>(M1_ - 2 * H1_, 1);
        boost::random::uniform_int_distribution<integer> iteration_done_dist(0, f_ij_1_lower + b_ij_1_upper - 1);
        if (iteration_done_dist(gen) < f_ij_1_lower)
            continue;

        // perform a random inverse heavy 1-way switching
        // start by collecting all neighbors of i and j that can be used for the switching
        std::vector<node> i_neighbor_choices;
        std::vector<node> j_neighbor_choices;
        for (node ij : {i, j}) {
            for (node neighbor : state_.graph.unique_neighbors(ij)) {
                // a switching cannot create a heavy loop so the inverse cannot use a heavy loop
                if (neighbor == ij)
                    continue;
                // same for heavy multiple edge
                if (is_heavy(neighbor) && has_multiple_edge(ij, neighbor))
                    continue;
                for (count c = 0; c < multiplicity_of(ij, neighbor); ++c) {
                    if (ij == i) {
                        i_neighbor_choices.push_back(neighbor);
                    } else if (ij == j) {
                        j_neighbor_choices.push_back(neighbor);
                    }
                }
            }
        }
        assert(i_neighbor_choices.size() == d_i - W_ij);
        assert(j_neighbor_choices.size() == d_j - W_ji);
        // choose two neighbors to switch with
        std::uniform_int_distribution<node> i_neighbor_dist(0, i_neighbor_choices.size() - 1);
        std::uniform_int_distribution<node> j_neighbor_dist(0, j_neighbor_choices.size() - 1);
        node i_neighbor = i_neighbor_choices[i_neighbor_dist(gen)];
        node j_neighbor = j_neighbor_choices[j_neighbor_dist(gen)];
        // f-reject if the switching is not valid
        REJECT_IF(is_heavy(i_neighbor) && is_heavy(j_neighbor));
        // perform the switching
        add_edge(i, j);
        add_edge(i_neighbor, j_neighbor);
        remove_edge(i, i_neighbor);
        remove_edge(j, j_neighbor);

        // b-rejection chance
        // calculate f_ij_1, the number of applicable 1-way switchings
        // a valid 1-way switching either uses a pair with both incident nodes light or
        // one light and one heavy where the heavy node is not adjacent to i or j
        // first calculate Z, the number of nice pairs where both incident nodes are light (case i)
        integer Z = M1_;
        for (node v = 0; v < h_; ++v) {
            for (auto neighbor : state_.graph.unique_neighbors(v)) {
                if (neighbor != v && is_heavy(neighbor)) { // we will see this pair twice
                    Z -= 1 * multiplicity_of(v, neighbor);
                } else {
                    Z -= 2 * multiplicity_of(v, neighbor);
                }
            }
        }
        // now calculate X and Y, the number of nice pairs where one node is heavy but the other light (case ii)
        integer X = 0;
        integer Y = 0;
        for (node v = 0; v < h_; ++v) {
            if (v == i || v == j)
                continue;
            for (auto neighbor : state_.graph.unique_neighbors(v)) {
                if (is_heavy(neighbor))
                    continue;
                if (!has_edge(v, i)) {
                    X += multiplicity_of(v, neighbor);
                }
                if (!has_edge(v, j)) {
                    Y += multiplicity_of(v, neighbor);
                }
            }
        }
        // we can either use a pair of case (i) or (ii) so just add them up
        integer f_ij_1 = Z + X + Y;
        assert(f_ij_1 > 0);
        assert(f_ij_1_lower <= f_ij_1);
        boost::random::uniform_int_distribution<integer> b_rejection_1_dist(0, f_ij_1 - 1);
        REJECT_IF(b_rejection_1_dist(gen) >= f_ij_1_lower);
    }
    ACCEPT;
}

bool IncPowerlawGraphSampler::remove_heavy_loops_or_reject(std::mt19937_64 &gen) {
    while (!state_.heavy_loops.empty()) {
        if (!keep_going_()) return false;

        auto [i, m] = state_.heavy_loops.front();
        state_.heavy_loops.pop();

        // sample m pairs and try to use them for a m-way loop switching
        for (std::size_t switched_pairs = 0; switched_pairs < m; ++switched_pairs) {
            // sample a pair
            auto [v1, v2] = sample_edge(gen);

            // f-reject if the switching is invalid
            // we can't create or remove other heavy multiple edges or loops
            REJECT_IF(is_heavy(v1) && is_heavy(v2));
            REJECT_IF(i == v1 || i == v2);
            REJECT_IF(is_heavy(v1) && has_edge(i, v1));
            REJECT_IF(is_heavy(v2) && has_edge(i, v2));

            // switch with this pair
            remove_edge(v1, v2);
            add_edge(i, v1);
            add_edge(i, v2);
        }
        remove_heavy_loop(i, m);

        // b-rejection chance
        count d_i = degree_sequence_[i];
        count W_i = state_.heavy_multiple_edges_at[i];
        integer valid_and_invalid_switchings = ordered_choices(d_i - W_i, 2 * m);
        integer invalid_switchings_upper_bound = integer(m) * integer(h_) * integer(h_) *
                                                 ordered_choices(d_i - W_i, 2 * m - 2);
        integer mb_i_lower = std::max<integer>(valid_and_invalid_switchings - invalid_switchings_upper_bound, 1);
        integer b_i = calculate_b_i(i, m);
        assert(b_i > 0);
        assert(mb_i_lower <= b_i);
        boost::random::uniform_int_distribution<integer> b_rejection_dist(0, b_i - 1);
        REJECT_IF(b_rejection_dist(gen) >= mb_i_lower);
    }
    ACCEPT;
}

void IncPowerlawGraphSampler::initialize_stage_2_and_3() {
    // update stage flag
    state_.stage = STAGE_2_OR_3;

    // initialize edge lists and and edge-related quantities
    state_.graph.for_each_twoway([&] (node u, node v, std::size_t m) {
        // determine type of edge
        const bool loop = u == v;
        const bool multiple_edge = !loop && m >= 2;
        const bool simple_edge = !multiple_edge && !loop;

        // update number of simple edges at u and v
        if (simple_edge) {
            state_.simple_edges_at[u]++;
        } else if (u <= v) {
            const bool multi_loop = loop && m >= 2;
            const bool double_edge = multiple_edge && m == 2;
            const bool triple_edge = multiple_edge && m == 3;

            // add to edge list
            if (multiple_edge) {
                if (double_edge) {
                    state_.light_doubles.emplace_back(v, u);
                } else if (triple_edge) {
                    state_.light_triples.emplace_back(v, u);
                } else {
                    state_.light_high_multiplicity_edges++;
                }
            } else if (loop) {
                if (multi_loop) {
                    state_.light_high_multiplicity_loops++;
                } else {
                    state_.light_loops.emplace_back(u);
                }
            }
        }
    });

    // calculate initial values for the number of simple structures at nodes
    constexpr size_t kHistSize = 10;
    std::array<count, kHistSize> hist;
    ranges::fill(hist, 0u);

    for (node v = first_degree1_node_; v--;) {
        auto m = state_.simple_edges_at[v];
        if (is_heavy(v) || m >= kHistSize) {
            add_number_of_simple_structures_at(v);
        } else {
            ++hist[m];
        }
    }

    for(count deg = 2; deg < kHistSize; ++deg) {
        if (!hist[deg]) continue;

        integer two_paths = deg * (deg - 1);
        two_paths *= hist[deg];
        integer three_stars = two_paths * (deg - 2);
        integer four_stars = three_stars * (deg - 3);

        state_.simple_two_paths += two_paths;
        state_.light_simple_two_paths += two_paths;
        state_.simple_three_stars += three_stars;
        state_.light_simple_three_stars += three_stars;
        state_.simple_four_stars += four_stars;
        state_.light_simple_four_stars += four_stars;
    }
}

bool IncPowerlawGraphSampler::meets_stage_2_preconditions() {
#ifdef LOG_EDGES
    std::cerr << "loops   : " << state_.light_loops.size() << std::endl;
    std::cerr << "BL      : " << (rational(4 * L2_) / M1_).convert_to<double>() << std::endl;
    std::cerr << "doubles : " << state_.light_doubles.size() << std::endl;
    std::cerr << "BD      : " << (rational(4 * L2_ * M2_) / (M1_ * M1_)).convert_to<double>() << std::endl;
    std::cerr << "triples : " << state_.light_triples.size() << std::endl;
    std::cerr << "BT      : " << (rational(2 * L3_ * M3_) / (M1_ * M1_ * M1_)).convert_to<double>() << std::endl;
#endif
    std::size_t loops = state_.light_loops.size();
    integer BL = 4 * L2_ / M1_;
    REJECT_IF(loops > BL);
    std::size_t doubles = state_.light_doubles.size();
    integer BD = 4 * L2_ * M2_ / (M1_ * M1_);
    REJECT_IF(doubles > BD);
    std::size_t triples = state_.light_triples.size();
    integer BT = 2 * L3_ * M3_ / (M1_ * M1_ * M1_);
    REJECT_IF(triples > BT);
    REJECT_IF(state_.light_high_multiplicity_edges > 0 || state_.light_high_multiplicity_loops > 0);

    // reject if lower bounds don't meet assumptions for incremental relaxation or type distributions
    if (loops > 0 && !explicit_bl_GV0_calculation_enabled_) {
        integer bl_m1_min = bl_m1_lower();
        REJECT_IF(bl_m1_min < 1);
    }
    if (triples > 0) {
        integer bt_m1_min = b_triplet_1_lower();
        REJECT_IF(bt_m1_min < 1);
    }
    if (doubles > 0) {
        integer bd_m1_min = b_doublet_1_lower();
        REJECT_IF(bd_m1_min < 1);
    }
    ACCEPT;
}

bool IncPowerlawGraphSampler::remove_light_loops_or_reject(std::mt19937_64 &gen) {
    while (!state_.light_loops.empty()) {
        if (!keep_going_()) return false;

        // choose a random loop
        std::uniform_int_distribution<std::size_t> light_loop_dist(0, state_.light_loops.size() - 1);
        std::size_t light_loop_index = light_loop_dist(gen);
        node v1 = state_.light_loops[light_loop_index];
        state_.light_loops[light_loop_index] = state_.light_loops.back();
        state_.light_loops.pop_back();

        // choose two random pairs
        auto [v2, v4] = sample_edge(gen);
        auto [v3, v5] = sample_edge(gen);

        // f-reject if there are any vertex collisions
        REJECT_UNLESS(are_unique({v1, v2, v3, v4, v5}));
        // f-reject if there are forbidden edges
        REJECT_IF(has_multiple_edge(v2, v4) ||
                  has_multiple_edge(v3, v5) ||
                  has_edge(v1, v2) ||
                  has_edge(v1, v3) ||
                  has_edge(v4, v5));

        // perform the switching
        add_simple_edge(v1, v2);
        add_simple_edge(v1, v3);
        add_simple_edge(v4, v5);
        remove_light_loop(v1);
        remove_simple_edge(v2, v4);
        remove_simple_edge(v3, v5);

        // b-rejection chance
        integer dh = degree_sequence_[h_];
        integer loops = state_.light_loops.size();
        integer doubles = state_.light_doubles.size();
        integer triples = state_.light_triples.size();
        // first calculate the lower bounds for the number of ways to choose v1, v2, v3, v4 and v5
        integer bl_m0 = std::max<integer>(L2_ - // number of light two-paths
                                          12 * triples * dh - // two-paths that contain a triple
                                          8 * doubles * dh - // two-paths that contain a double
                                          loops * dh * dh, // two-paths that have a loop on v1
                                          1);
        integer bl_m1 = bl_m1_lower();
        // now calculate the number of choices possible in the current state
        integer bl_GV0;
        // if enabled, count explicitly to remove an assumption that prevents testing with small degree sequences
        if (explicit_bl_GV0_calculation_enabled_) {
            bl_m0 = 1;
            bl_GV0 = count_light_two_paths_with_matching_simple_pairs();
        } else {
            // otherwise the number of choices for v1v2v3 is the number of light simple two-paths without a loop at v1
            bl_GV0 = state_.light_simple_two_paths;
            for (auto v : state_.light_loops) {
                bl_GV0 -= calculate_simple_two_paths_at(v);
            }
        }
        // for v4v5, start with the total number of simple pairs then subtract all invalid pairs
        integer bl_GV1 = M1_ - 2 * loops
                             - 4 * doubles
                             - 6 * triples;
        // find all nodes in the two-neighborhood
        // semantics: each node in the two-neighborhoods gets an entry in invalid_v5_choices
        std::unordered_map<node, count> invalid_v5_choices;
        invalid_v5_choices.reserve(3*degree_sequence_.front() * degree_sequence_.front());
        invalid_v5_choices[v1] = 0;
        invalid_v5_choices[v2] = 0;
        invalid_v5_choices[v3] = 0;

        for (node v : {v1, v2, v3}) {
            for (node u : state_.graph.unique_neighbors(v)) {
                if (u == v1 || u == v2 || u == v3)
                    continue;

                invalid_v5_choices.try_emplace(u, 0);

                for (node w : state_.graph.unique_neighbors(u)) {
                    if (w == u || w == v1 || w == v2 || w == v3)
                        continue;
                    if (has_multiple_edge(u, w))
                        continue;

                    invalid_v5_choices[w] += (v == v3);
                }
            }
        }
        // now subtract all pairs that have vertex collisions or forbidden edges
        // we do this by treating each node as a potential start node (choice for v4)
        // and then calculate the number of possible choices for v5
        for (auto [neighbor, inv_ch] : invalid_v5_choices) {
            count simple_edges = state_.simple_edges_at[neighbor];
            // remove the old number of pairs so we can add the number of valid pairs later
            bl_GV1 -= simple_edges;
            // if there is a vertex collision we cannot add any pairs so the iteration is done
            if (neighbor == v1 || neighbor == v2 || neighbor == v3)
                continue;
            // same if this node cannot be used as v4
            if (has_edge(v2, neighbor))
                continue;
            // now re-add the number of valid pairs which is just the number of possible choices for v5
            count valid_v5_choices = simple_edges - inv_ch; // invalid_v5_choices[neighbor];
            if (multiplicity_of(neighbor, v1) == 1)
                valid_v5_choices--;
            if (multiplicity_of(neighbor, v2) == 1)
                valid_v5_choices--;
            if (multiplicity_of(neighbor, v3) == 1)
                valid_v5_choices--;
            bl_GV1 += valid_v5_choices;
        }
        assert(bl_m0 <= bl_GV0);
        assert(bl_m1 <= bl_GV1);
        assert(bl_GV0 > 0);
        assert(bl_GV1 > 0);
        boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (bl_GV0 * bl_GV1) - 1);
        REJECT_IF(b_rejection_dist(gen) >= bl_m0 * bl_m1);
    }
    ACCEPT;
}

bool IncPowerlawGraphSampler::remove_light_triples_or_reject(std::mt19937_64 &gen) {
    // parameters of the type distribution
    const std::size_t i_1 = state_.light_triples.size();
    std::size_t i = i_1;
    std::vector<rational> x(i_1 + 1, 0);
    std::vector<rational> p_t(i_1 + 1, 0);
    rational p_ta, p_tb, p_tc;
    rational epsilon;
    if (i_1 > 0) {
        // initialize type distribution
        epsilon = rational(28 * M2_ * M2_) / (M1_ * M1_ * M1_);
        REJECT_IF(epsilon >= 1); // t-rejection
        x[i_1] = 1;
        p_t[i_1] = rational(1) - epsilon;
        p_ta = 0; p_tb = 0; p_tc = 0;
    }
    while (!state_.light_triples.empty()) {
        if (!keep_going_()) return false;

        // update the type distribution if the number of triples changed
        if (i != state_.light_triples.size()) {
            i = state_.light_triples.size();
            // calculate x_i
            rational b_triplet_lower = b_triplet_0_lower() * b_triplet_1_lower();
            integer f_t_upper = 12 * (i + 1) * M1_ * M1_ * M1_;
            x[i] = x[i + 1] * p_t[i + 1] * (b_triplet_lower / f_t_upper) + 1;
            // calculate type probabilities
            if (i < i_1) { // boosters that leave the number of double-edges the same have non-zero probability
                rational f_ta_upper = 3 * M3_ * L3_ * M2_ * M2_;
                rational b_ta_lower = b_ta_tb_tc_lower(3);
                REJECT_IF(b_ta_lower < 1); // neccessary asumption for incremental relaxation doesnt hold
                p_ta = p_t[i + 1] * (x[i + 1] / x[i]) * ((f_ta_upper / b_ta_lower) / f_t_upper);
                rational f_tb_upper = 3 * M3_ * L3_ * M2_ * M2_ * M2_ * M2_;
                rational b_tb_lower = b_ta_tb_tc_lower(6);
                REJECT_IF(b_tb_lower < 1); // neccessary asumption for incremental relaxation doesnt hold
                p_tb = p_t[i + 1] * (x[i + 1] / x[i]) * ((f_tb_upper / b_tb_lower) / f_t_upper);
                rational f_tc_upper = M3_ * L3_ * M2_ * M2_ * M2_ * M2_ * M2_ * M2_;
                rational b_tc_lower = b_ta_tb_tc_lower(9);
                REJECT_IF(b_tc_lower < 1); // neccessary asumption for incremental relaxation doesnt hold
                p_tc = p_t[i + 1] * (x[i + 1] / x[i]) * ((f_tc_upper / b_tc_lower) / f_t_upper);
            } else {
                p_ta = 0; p_tb = 0; p_tc = 0;
            }
            REJECT_IF(p_ta + p_tb + p_tc > epsilon);
        }
        // sample switching type and perform it
        integer d_common = 1;
        for (const rational& p_tau : {p_t[i], p_ta, p_tb, p_tc}) {
            integer d_tau = boost::multiprecision::denominator(p_tau);
            d_common = boost::multiprecision::lcm(d_common, d_tau);
        }
        integer c_t = integer(p_t[i] * d_common);
        integer c_ta = c_t + integer(p_ta * d_common);
        integer c_tb = c_ta + integer(p_tb * d_common);
        integer c_tc = c_tb + integer(p_tc * d_common);
        boost::random::uniform_int_distribution<integer> type_dist(0, d_common - 1);
        integer type = type_dist(gen);
        if (type < c_t) {
            if (!perform_type_t_switching_or_reject(gen))
                return false;
        } else if (type < c_ta) {
            if (!perform_type_ta_switching_or_reject(gen))
                return false;
        } else if (type < c_tb) {
            if (!perform_type_tb_switching_or_reject(gen))
                return false;
        } else if (type < c_tc) {
            if (!perform_type_tc_switching_or_reject(gen))
                return false;
        } else {
            REJECT_IF(type >= c_tc);
        }
    }
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_t_switching_or_reject(std::mt19937_64& gen) {
    // choose a random triple edge
    std::uniform_int_distribution<std::size_t> light_triple_dist(0, state_.light_triples.size() - 1);
    std::size_t light_triple_index = light_triple_dist(gen);
    auto [v1, v2] = state_.light_triples[light_triple_index];
    state_.light_triples[light_triple_index] = state_.light_triples.back();
    state_.light_triples.pop_back();
    // choose three random pairs
    auto [v3, v4] = sample_edge(gen);
    auto [v5, v6] = sample_edge(gen);
    auto [v7, v8] = sample_edge(gen);

    // f-reject if there are any vertex collisions
    REJECT_UNLESS(are_unique({v1, v2, v3, v4, v5, v6, v7, v8}));
    // f-reject if there are forbidden edges
    REJECT_IF(has_multiple_edge(v3, v4) ||
              has_multiple_edge(v5, v6) ||
              has_multiple_edge(v7, v8) ||
              has_edge(v1, v3) ||
              has_edge(v1, v5) ||
              has_edge(v1, v7) ||
              has_edge(v2, v4) ||
              has_edge(v2, v6) ||
              has_edge(v2, v8));

    // perform the switching
    add_simple_edge(v1, v3);
    add_simple_edge(v1, v5);
    add_simple_edge(v1, v7);
    add_simple_edge(v2, v4);
    add_simple_edge(v2, v6);
    add_simple_edge(v2, v8);
    remove_light_triple(v1, v2);
    remove_simple_edge(v3, v4);
    remove_simple_edge(v5, v6);
    remove_simple_edge(v7, v8);

    // b-rejection chance
    integer bt_m0 = b_triplet_0_lower();
    integer bt_m1 = b_triplet_1_lower();
    // now calculate the number of choices possible for the current state (pairing)
    if (is_heavy(v2)) {
        // swap the three-stars so that we can relax the light three-star second
        std::swap(v1, v2);
        std::swap(v3, v4);
        std::swap(v5, v6);
        std::swap(v7, v8);
    }
    integer bt_GV0 = state_.simple_three_stars;
    integer bt_GV1 = calculate_bt_GV1(v1, v3, v5, v7);
    assert(bt_m0 <= bt_GV0);
    assert(bt_m1 <= bt_GV1);
    assert(bt_GV0 > 0);
    assert(bt_GV1 > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (bt_GV0 * bt_GV1) - 1);
    REJECT_IF(b_rejection_dist(gen) >= bt_m0 * bt_m1);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_ta_switching_or_reject(std::mt19937_64& gen) {
    // f-reject if there are no three-stars
    REJECT_UNLESS(state_.light_simple_three_stars > 0);
    // pick the structures
    auto [v1, v5, v7, v9] = sample_three_star(gen, false);
    auto [v3, v11, v13] = sample_two_path(gen, false);
    auto [v2, v6, v8, v10] = sample_three_star(gen, true);
    auto [v4, v12, v14] = sample_two_path(gen, false);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14}));
    REJECT_IF(has_edge(v1, v2) || // bad edge 1
              has_edge(v3, v4) || // bad edge 2
              has_edge(v5, v6) || // bad edge 3
              has_edge(v7, v8) || // bad edge 4
              has_edge(v9, v10) ||
              has_edge(v11, v12) ||
              has_edge(v13, v14));

    // perform the switching
    add_simple_edge(v1, v3);
    add_simple_edge(v2, v4);
    add_simple_edge(v3, v4); // add bad edge 2 to create the needed triplet
    add_simple_edge(v9, v10);
    add_simple_edge(v11, v12);
    add_simple_edge(v13, v14);
    remove_simple_edge(v1, v9);
    remove_simple_edge(v2, v10);
    remove_simple_edge(v3, v11);
    remove_simple_edge(v3, v13);
    remove_simple_edge(v4, v12);
    remove_simple_edge(v4, v14);

    // b-rejection
    // lower bounds
    integer b_t_0_lower = b_triplet_0_lower();
    integer b_t_1_lower = b_triplet_1_lower();
    integer b_ta_lower = b_ta_tb_tc_lower(3);
    // incremental relaxation
    integer b_ta_GV0 = state_.simple_two_paths;
    integer b_ta_GV1 = calculate_bt_GV1(v1, v3, v5, v7);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(v9, v10, v1, v2);
    additional_pairs.emplace_back(v11, v12, v3, v4);
    additional_pairs.emplace_back(v13, v14, v3, v4);
    std::size_t triples = state_.light_triples.size();
    std::size_t doubles = state_.light_doubles.size();
    integer simple_pairs = M1_ - 6 * triples - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, v2, v3, v4, v5, v6, v7, v8};
    integer b_ta_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_ta_GV_pair = simple_pairs;
        b_ta_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_ta_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_ta_GV_pairs *= b_ta_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_t_0_lower <= b_ta_GV0);
    assert(b_t_1_lower <= b_ta_GV1);
    assert(b_ta_lower <= b_ta_GV_pairs);
    assert(b_ta_GV0 > 0);
    assert(b_ta_GV1 > 0);
    assert(b_ta_GV_pairs > 0);
    integer lower_bound = b_t_0_lower * b_t_1_lower * b_ta_lower;
    integer inverse_switchings = b_ta_GV0 * b_ta_GV1 * b_ta_GV_pairs;
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, inverse_switchings - 1);
    REJECT_IF(b_rejection_dist(gen) >= lower_bound);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_tb_switching_or_reject(std::mt19937_64& gen) {
    // f-reject if there are no three-stars
    REJECT_UNLESS(state_.light_simple_three_stars > 0);
    // pick the structures
    auto [v1, v7, v9, v11] = sample_three_star(gen, false);
    auto [v3, v13, v15] = sample_two_path(gen, false);
    auto [v5, v17, v19] = sample_two_path(gen, false);
    auto [v2, v8, v10, v12] = sample_three_star(gen, true);
    auto [v4, v14, v16] = sample_two_path(gen, false);
    auto [v6, v18, v20] = sample_two_path(gen, false);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                                 v11, v12, v13, v14, v15, v16, v17, v18, v19, v20}));
    REJECT_IF(has_edge(v1, v2) || // bad edge 1
              has_edge(v3, v4) || // bad edge 2
              has_edge(v5, v6) || // bad edge 3
              has_edge(v7, v8) || // bad edge 4
              has_edge(v9, v10) ||
              has_edge(v11, v12) ||
              has_edge(v13, v14) ||
              has_edge(v15, v16) ||
              has_edge(v17, v18) ||
              has_edge(v19, v20));

    // perform the switching
    add_simple_edge(v1, v3);
    add_simple_edge(v1, v5);
    add_simple_edge(v2, v4);
    add_simple_edge(v2, v6);
    add_simple_edge(v3, v4); // add bad edge 2 to create the needed triplet
    add_simple_edge(v5, v6); // add bad edge 3
    add_simple_edge(v9, v10);
    add_simple_edge(v11, v12);
    add_simple_edge(v13, v14);
    add_simple_edge(v15, v16);
    add_simple_edge(v17, v18);
    add_simple_edge(v19, v20);
    remove_simple_edge(v1, v9);
    remove_simple_edge(v1, v11);
    remove_simple_edge(v2, v10);
    remove_simple_edge(v2, v12);
    remove_simple_edge(v3, v13);
    remove_simple_edge(v3, v15);
    remove_simple_edge(v4, v14);
    remove_simple_edge(v4, v16);
    remove_simple_edge(v5, v17);
    remove_simple_edge(v5, v19);
    remove_simple_edge(v6, v18);
    remove_simple_edge(v6, v20);

    // b-rejection
    // lower bounds
    integer b_t_0_lower = b_triplet_0_lower();
    integer b_t_1_lower = b_triplet_1_lower();
    integer b_tb_lower = b_ta_tb_tc_lower(6);
    // incremental relaxation
    integer b_tb_GV0 = state_.simple_two_paths;
    integer b_tb_GV1 = calculate_bt_GV1(v1, v3, v5, v7);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(v9, v10, v1, v2);
    additional_pairs.emplace_back(v11, v12, v1, v2);
    additional_pairs.emplace_back(v13, v14, v3, v4);
    additional_pairs.emplace_back(v15, v16, v3, v4);
    additional_pairs.emplace_back(v17, v18, v5, v6);
    additional_pairs.emplace_back(v19, v20, v5, v6);
    std::size_t triples = state_.light_triples.size();
    std::size_t doubles = state_.light_doubles.size();
    integer simple_pairs = M1_ - 6 * triples - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, v2, v3, v4, v5, v6, v7, v8};
    integer b_tb_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_tb_GV_pair = simple_pairs;
        b_tb_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_tb_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_tb_GV_pairs *= b_tb_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_t_0_lower <= b_tb_GV0);
    assert(b_t_1_lower <= b_tb_GV1);
    assert(b_tb_lower <= b_tb_GV_pairs);
    assert(b_tb_GV0 > 0);
    assert(b_tb_GV1 > 0);
    assert(b_tb_GV_pairs > 0);
    integer lower_bound = b_t_0_lower * b_t_1_lower * b_tb_lower;
    integer inverse_switchings = b_tb_GV0 * b_tb_GV1 * b_tb_GV_pairs;
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, inverse_switchings - 1);
    REJECT_IF(b_rejection_dist(gen) >= lower_bound);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_tc_switching_or_reject(std::mt19937_64& gen) {
    // f-reject if there are no three-stars
    REJECT_UNLESS(state_.light_simple_three_stars > 0);
    // pick the structures
    auto [v1, v9, v11, v13] = sample_three_star(gen, false);
    auto [v3, v15, v17] = sample_two_path(gen, false);
    auto [v5, v19, v21] = sample_two_path(gen, false);
    auto [v7, v23, v25] = sample_two_path(gen, false);
    auto [v2, v10, v12, v14] = sample_three_star(gen, true);
    auto [v4, v16, v18] = sample_two_path(gen, false);
    auto [v6, v20, v22] = sample_two_path(gen, false);
    auto [v8, v24, v26] = sample_two_path(gen, false);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                                 v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                                 v21, v22, v23, v24, v25, v26}));
    REJECT_IF(has_edge(v1, v2) || // bad edge 1
              has_edge(v3, v4) || // bad edge 2
              has_edge(v5, v6) || // bad edge 3
              has_edge(v7, v8) || // bad edge 4
              has_edge(v9, v10) ||
              has_edge(v11, v12) ||
              has_edge(v13, v14) ||
              has_edge(v15, v16) ||
              has_edge(v17, v18) ||
              has_edge(v19, v20) ||
              has_edge(v21, v22) ||
              has_edge(v23, v24) ||
              has_edge(v25, v26));

    // perform the switching
    add_simple_edge(v1, v3);
    add_simple_edge(v1, v5);
    add_simple_edge(v1, v7);
    add_simple_edge(v2, v4);
    add_simple_edge(v2, v6);
    add_simple_edge(v2, v8);
    add_simple_edge(v3, v4); // add bad edge 2 to create the needed triplet
    add_simple_edge(v5, v6); // add bad edge 3
    add_simple_edge(v7, v8); // add bad edge 4
    add_simple_edge(v9, v10);
    add_simple_edge(v11, v12);
    add_simple_edge(v13, v14);
    add_simple_edge(v15, v16);
    add_simple_edge(v17, v18);
    add_simple_edge(v19, v20);
    add_simple_edge(v21, v22);
    add_simple_edge(v23, v24);
    add_simple_edge(v25, v26);
    remove_simple_edge(v1, v9);
    remove_simple_edge(v1, v11);
    remove_simple_edge(v1, v13);
    remove_simple_edge(v2, v10);
    remove_simple_edge(v2, v12);
    remove_simple_edge(v2, v14);
    remove_simple_edge(v3, v15);
    remove_simple_edge(v3, v17);
    remove_simple_edge(v4, v16);
    remove_simple_edge(v4, v18);
    remove_simple_edge(v5, v19);
    remove_simple_edge(v5, v21);
    remove_simple_edge(v6, v20);
    remove_simple_edge(v6, v22);
    remove_simple_edge(v7, v23);
    remove_simple_edge(v7, v25);
    remove_simple_edge(v8, v24);
    remove_simple_edge(v8, v26);

    // b-rejection
    // lower bounds
    integer b_t_0_lower = b_triplet_0_lower();
    integer b_t_1_lower = b_triplet_1_lower();
    integer b_tc_lower = b_ta_tb_tc_lower(9);
    // incremental relaxation
    integer b_tc_GV0 = state_.simple_two_paths;
    integer b_tc_GV1 = calculate_bt_GV1(v1, v3, v5, v7);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(v9, v10, v1, v2);
    additional_pairs.emplace_back(v11, v12, v1, v2);
    additional_pairs.emplace_back(v13, v14, v1, v2);
    additional_pairs.emplace_back(v15, v16, v3, v4);
    additional_pairs.emplace_back(v17, v18, v3, v4);
    additional_pairs.emplace_back(v19, v20, v5, v6);
    additional_pairs.emplace_back(v21, v22, v5, v6);
    additional_pairs.emplace_back(v23, v24, v7, v8);
    additional_pairs.emplace_back(v25, v26, v7, v8);
    std::size_t triples = state_.light_triples.size();
    std::size_t doubles = state_.light_doubles.size();
    integer simple_pairs = M1_ - 6 * triples - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, v2, v3, v4, v5, v6, v7, v8};
    integer b_tc_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_tc_GV_pair = simple_pairs;
        b_tc_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_tc_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_tc_GV_pairs *= b_tc_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_t_0_lower <= b_tc_GV0);
    assert(b_t_1_lower <= b_tc_GV1);
    assert(b_tc_lower <= b_tc_GV_pairs);
    assert(b_tc_GV0 > 0);
    assert(b_tc_GV1 > 0);
    assert(b_tc_GV_pairs > 0);
    integer lower_bound = b_t_0_lower * b_t_1_lower * b_tc_lower;
    integer inverse_switchings = b_tc_GV0 * b_tc_GV1 * b_tc_GV_pairs;
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, inverse_switchings - 1);
    REJECT_IF(b_rejection_dist(gen) >= lower_bound);
    ACCEPT;
}

bool IncPowerlawGraphSampler::remove_light_doubles_or_reject(std::mt19937_64 &gen) {
    // parameters of the type distribution
    const std::size_t i_1 = state_.light_doubles.size();
    std::size_t i = i_1;
    std::vector<rational> x(i_1 + 1, 0);
    std::vector<rational> p_I(i_1 + 1, 0);
    rational p_III, p_IIIa, p_IV, p_IVa, p_V, p_Va, p_Vb, p_Vc,
             p_VI, p_VIa, p_VIb, p_VII, p_VIIa, p_VIIb, p_VIIc, p_VIId, p_VIIe;
    rational Xi;
    if (i_1 > 0) {
        // initialize type distribution
        integer M1_2 = M1_ * M1_;
        integer M1_3 = M1_2 * M1_;
        integer M1_4 = M1_3 * M1_;
        Xi = rational(32 * M2_ * M2_) / M1_3 +
             rational(36 * M4_ * L4_) / (M2_ * L2_ * M1_2) +
             rational(32 * M3_ * M3_) / M1_4;
        REJECT_IF(Xi >= 1);
        x[i_1] = 1;
        p_I[i_1] = rational(1) - Xi;
        p_III = 0; p_IIIa = 0;
        p_IV = 0; p_IVa = 0;
        p_V = 0; p_Va = 0; p_Vb = 0; p_Vc = 0;
        p_VI = 0; p_VIa = 0; p_VIb = 0;
        p_VII = 0; p_VIIa = 0; p_VIIb = 0; p_VIIc = 0; p_VIId = 0; p_VIIe = 0;
    }
    while (!state_.light_doubles.empty()) {
        if (!keep_going_()) return false;

        // first, perform a type I switching with probability p_I
        bool update_type_distribution = i != state_.light_doubles.size();
        i = state_.light_doubles.size();
        if (update_type_distribution) { // number of double edges has changed, so update p_I
            assert(i < i_1);
            // calculate x_i
            rational b_doublet_lower = b_doublet_0_lower() * b_doublet_1_lower();
            integer f_I_upper = 4 * (i + 1) * M1_ * M1_;
            x[i] = x[i + 1] * p_I[i + 1] * (b_doublet_lower / f_I_upper) + 1;
            // calculate p_III and p_I
            rational f_III_upper = M3_ * L3_;
            rational b_III_lower = b_I_to_VII_pairs_lower(1, i);
            REJECT_IF(b_III_lower < 1);  // neccessary asumption for incremental relaxation doesnt hold
            p_III = p_I[i + 1] * (x[i + 1] / x[i]) * ((f_III_upper / b_III_lower) / f_I_upper);
            p_I[i] = rational(1) - p_III - Xi;
            REJECT_IF(p_I[i] < 0);
        }
        // sample whether we perform type I
        integer d_I = boost::multiprecision::denominator(p_I[i]);
        integer c_I = boost::multiprecision::numerator(p_I[i]);
        boost::random::uniform_int_distribution<integer> p_I_dist(0, d_I - 1);
        if (p_I_dist(gen) < c_I) {
            if (!perform_type_I_switching_or_reject(gen))
                return false;
        } else {
            // otherwise, perform booster switching tau with probability p_tau
            if (update_type_distribution) { // number of double edges has changed, so update type distribution
                assert(i < i_1);
                // calculate the type probabilities
                integer f_I_upper = 4 * (i + 1) * M1_ * M1_;
                rational IV_to_VII_common = p_I[i + 1] * x[i + 1] / (x[i] * f_I_upper);
                integer f_IV_upper = 2 * M2_ * M2_ * M2_ * L2_;
                integer b_IV_lower = b_I_to_VII_pairs_lower(3, i);
                REJECT_IF(b_IV_lower < 1);
                p_IV = IV_to_VII_common * f_IV_upper / b_IV_lower;
                integer f_V_upper = 2 * M2_ * M2_ * M3_ * L3_;
                integer b_V_lower = b_I_to_VII_pairs_lower(4, i);
                REJECT_IF(b_V_lower < 1);
                p_V = IV_to_VII_common * f_V_upper / b_V_lower;
                integer f_VI_upper = M2_ * M2_ * M2_ * M2_ * M2_ * L2_;
                integer b_VI_lower = b_I_to_VII_pairs_lower(6, i);
                REJECT_IF(b_VI_lower < 1);
                p_VI = IV_to_VII_common * f_VI_upper / b_VI_lower;
                integer b_VII_lower = b_I_to_VII_pairs_lower(7, i);
                integer f_VII_upper = M2_ * M2_ * M2_ * M2_ * M3_ * L3_;
                REJECT_IF(b_VII_lower < 1);
                p_VII = IV_to_VII_common * f_VII_upper / b_VII_lower;
                if (i + 1 < i_1) { // boosters that add a double-edge have non-zero probability
                    integer f_I_upper = 4 * (i + 2) * M1_ * M1_;
                    rational IIIa_to_VIIb_common =  p_I[i + 2] * x[i + 2] / (x[i] * f_I_upper);
                    integer f_IIIa_upper = M4_ * L4_;
                    integer b_IIIa_lower = b_I_to_VII_pairs_lower(2, i + 1);
                    REJECT_IF(b_IIIa_lower < 1);
                    p_IIIa = IIIa_to_VIIb_common * f_IIIa_upper / b_IIIa_lower;
                    integer f_IVa_upper = 2 * M3_ * M3_ * M2_ * L2_;
                    integer b_IVa_lower = b_I_to_VII_pairs_lower(4, i + 1);
                    REJECT_IF(b_IVa_lower < 1);
                    p_IVa = IIIa_to_VIIb_common * f_IVa_upper / b_IVa_lower;
                    integer f_Va_upper = 2 * M2_ * M2_ * M4_ * L4_;
                    integer b_Va_lower = b_I_to_VII_pairs_lower(5, i + 1);
                    REJECT_IF(b_Va_lower < 1);
                    p_Va = IIIa_to_VIIb_common * f_Va_upper / b_Va_lower;
                    integer f_Vb_upper = 2 * M3_ * M3_ * M3_ * L3_;
                    integer b_Vb_lower = b_I_to_VII_pairs_lower(5, i + 1);
                    REJECT_IF(b_Vb_lower < 1);
                    p_Vb = IIIa_to_VIIb_common * f_Vb_upper / b_Vb_lower;
                    integer f_VIa_upper = 2 * M3_ * M3_ * M2_ * M2_ * M2_ * L2_;
                    integer b_VIa_lower = b_I_to_VII_pairs_lower(7, i + 1);
                    REJECT_IF(b_VIa_lower < 1);
                    p_VIa = IIIa_to_VIIb_common * f_VIa_upper / b_VIa_lower;
                    integer f_VIIa_upper = 2 * M2_ * M2_ * M3_ * M3_ * M3_ * L3_;
                    integer b_VIIa_lower = b_I_to_VII_pairs_lower(8, i + 1);
                    REJECT_IF(b_VIIa_lower < 1);
                    p_VIIa = IIIa_to_VIIb_common * f_VIIa_upper / b_VIIa_lower;
                    integer f_VIIb_upper = M2_ * M2_ * M2_ * M2_ * M4_ * L4_;
                    integer b_VIIb_lower = b_I_to_VII_pairs_lower(8, i + 1);
                    REJECT_IF(b_VIIb_lower < 1);
                    p_VIIb = IIIa_to_VIIb_common * f_VIIb_upper / b_VIIb_lower;
                } else {
                    p_IIIa = 0; p_IVa = 0; p_Va = 0; p_Vb = 0; p_VIa = 0; p_VIIa = 0; p_VIIb = 0;
                }
                if (i + 2 < i_1) { // boosters that add two double-edges have non-zero probability
                    integer f_I_upper = 4 * (i + 3) * M1_ * M1_;
                    rational Vc_to_VIId_common = p_I[i + 3] * x[i + 3] / (x[i] * f_I_upper);
                    integer f_Vc_upper = 2 * M3_ * M3_ * M4_ * L4_;
                    integer b_Vc_lower = b_I_to_VII_pairs_lower(6, i + 2);
                    REJECT_IF(b_Vc_lower < 1);
                    p_Vc = Vc_to_VIId_common * f_Vc_upper / b_Vc_lower;
                    integer f_VIb_upper = M3_ * M3_ * M3_ * M3_ * M2_ * L2_;
                    integer b_VIb_lower = b_I_to_VII_pairs_lower(8, i + 2);
                    REJECT_IF(b_VIb_lower < 1);
                    p_VIb = Vc_to_VIId_common * f_VIb_upper / b_VIb_lower;
                    integer f_VIIc_upper = 2 * M2_ * M2_ * M3_ * M3_ * M4_ * L4_;
                    integer b_VIIc_lower = b_I_to_VII_pairs_lower(9, i + 2);
                    REJECT_IF(b_VIIc_lower < 1);
                    p_VIIc = Vc_to_VIId_common * f_VIIc_upper / b_VIIc_lower;
                    integer f_VIId_upper = M3_ * M3_ * M3_ * M3_ * M3_ * L3_;
                    integer b_VIId_lower = b_I_to_VII_pairs_lower(9, i + 2);
                    REJECT_IF(b_VIId_lower < 1);
                    p_VIId = Vc_to_VIId_common * f_VIId_upper / b_VIId_lower;
                } else {
                    p_Vc = 0; p_VIb = 0; p_VIIc = 0; p_VIId = 0;
                }
                if (i + 3 < i_1) { // boosters that add three double-edges have non-zero probability
                    integer f_I_upper = 4 * (i + 4) * M1_ * M1_;
                    rational f_VIIe_upper = M3_ * M3_ * M3_ * M3_ * M4_ * L4_;
                    rational b_VIIe_lower = b_I_to_VII_pairs_lower(10, i + 3);
                    REJECT_IF(b_VIIe_lower < 1);
                    p_VIIe = p_I[i + 4] * (x[i + 4] / x[i]) * ((f_VIIe_upper / b_VIIe_lower) / f_I_upper);
                } else {
                    p_VIIe = 0;
                }
                REJECT_IF(p_IIIa + p_IV + p_IVa + p_V + p_Va + p_Vb + p_Vc + p_VI + p_VIa + p_VIb +
                          p_VII + p_VIIa + p_VIIb + p_VIIc + p_VIId + p_VIIe > Xi);
            }
            // we first sampled just whether to perform p_I, so rescale the other probabilities
            rational p_I_compl = rational(1) - p_I[i];
            p_III /= p_I_compl; p_IIIa /= p_I_compl;
            p_IV /= p_I_compl; p_IVa /= p_I_compl;
            p_V /= p_I_compl; p_Va /= p_I_compl; p_Vb /= p_I_compl; p_Vc /= p_I_compl;
            p_VI /= p_I_compl; p_VIa /= p_I_compl; p_VIb /= p_I_compl;
            p_VII /= p_I_compl; p_VIIa /= p_I_compl; p_VIIb /= p_I_compl;
            p_VIIc /= p_I_compl; p_VIId /= p_I_compl; p_VIIe /= p_I_compl;
            integer d_III_to_VII = 1;
            for (const rational& p_tau : {p_III, p_IIIa, p_IV, p_IVa, p_V, p_Va, p_Vb, p_Vc, p_VI, p_VIa, p_VIb,
                                          p_VII, p_VIIa, p_VIIb, p_VIIc, p_VIId, p_VIIe}) {
                integer d_tau = boost::multiprecision::denominator(p_tau);
                d_III_to_VII = boost::multiprecision::lcm(d_III_to_VII, d_tau);
            }
            integer c_III = integer(p_III * d_III_to_VII);
            integer c_IIIa = c_III + integer(p_IIIa * d_III_to_VII);
            integer c_IV = c_IIIa + integer(p_IV * d_III_to_VII);
            integer c_IVa = c_IV + integer(p_IVa * d_III_to_VII);
            integer c_V = c_IVa + integer(p_V * d_III_to_VII);
            integer c_Va = c_V + integer(p_Va * d_III_to_VII);
            integer c_Vb = c_Va + integer(p_Vb * d_III_to_VII);
            integer c_Vc = c_Vb + integer(p_Vc * d_III_to_VII);
            integer c_VI = c_Vc + integer(p_VI * d_III_to_VII);
            integer c_VIa = c_VI + integer(p_VIa * d_III_to_VII);
            integer c_VIb = c_VIa + integer(p_VIb * d_III_to_VII);
            integer c_VII = c_VIb + integer(p_VII * d_III_to_VII);
            integer c_VIIa = c_VII + integer(p_VIIa * d_III_to_VII);
            integer c_VIIb = c_VIIa + integer(p_VIIb * d_III_to_VII);
            integer c_VIIc = c_VIIb + integer(p_VIIc * d_III_to_VII);
            integer c_VIId = c_VIIc + integer(p_VIId * d_III_to_VII);
            integer c_VIIe = c_VIId + integer(p_VIIe * d_III_to_VII);
            // choose a type and perform a switching of that type
            boost::random::uniform_int_distribution<integer> type_dist(0, d_III_to_VII - 1);
            integer type = type_dist(gen);
            if (type < c_III) {
                if (!perform_type_III_switching_or_reject(gen))
                    return false;
            } else if (type < c_IIIa) {
                if (!perform_type_IIIa_switching_or_reject(gen))
                    return false;
            } else if (type < c_IV) {
                if (!perform_type_IV_switching_or_reject(gen))
                    return false;
            } else if (type < c_IVa) {
                if (!perform_type_IVa_switching_or_reject(gen))
                    return false;
            } else if (type < c_V) {
                if (!perform_type_V_switching_or_reject(gen))
                    return false;
            } else if (type < c_Va) {
                if (!perform_type_Va_switching_or_reject(gen))
                    return false;
            } else if (type < c_Vb) {
                if (!perform_type_Vb_switching_or_reject(gen))
                    return false;
            } else if (type < c_Vc) {
                if (!perform_type_Vc_switching_or_reject(gen))
                    return false;
            } else if (type < c_VI) {
                if (!perform_type_VI_switching_or_reject(gen))
                    return false;
            } else if (type < c_VIa) {
                if (!perform_type_VIa_switching_or_reject(gen))
                    return false;
            } else if (type < c_VIb) {
                if (!perform_type_VIb_switching_or_reject(gen))
                    return false;
            } else if (type < c_VII) {
                if (!perform_type_VII_switching_or_reject(gen))
                    return false;
            } else if (type < c_VIIa) {
                if (!perform_type_VIIa_switching_or_reject(gen))
                    return false;
            } else if (type < c_VIIb) {
                if (!perform_type_VIIb_switching_or_reject(gen))
                    return false;
            } else if (type < c_VIIc) {
                if (!perform_type_VIIc_switching_or_reject(gen))
                    return false;
            } else if (type < c_VIId) {
                if (!perform_type_VIId_switching_or_reject(gen))
                    return false;
            } else if (type < c_VIIe) {
                if (!perform_type_VIIe_switching_or_reject(gen))
                    return false;
            } else { // t-rejection
                // NOTE we just include the condition again here so that it can be logged with the macro
                REJECT_UNLESS(type < c_VIIe);
                assert(false);
            }
        }
    }
    return true;
}

bool IncPowerlawGraphSampler::perform_type_I_switching_or_reject(std::mt19937_64 &gen) {
    // choose a random double edge
    std::uniform_int_distribution<std::size_t> light_double_dist(0, state_.light_doubles.size() - 1);
    std::size_t light_double_index = light_double_dist(gen);
    auto [v1, v2] = state_.light_doubles[light_double_index];
    state_.light_doubles[light_double_index] = state_.light_doubles.back();
    state_.light_doubles.pop_back();

    // choose two random pairs
    auto [v3, v4] = sample_edge(gen);
    auto [v5, v6] = sample_edge(gen);

    // f-reject if there are any vertex collisions
    REJECT_UNLESS(are_unique({v1, v2, v3, v4, v5, v6}));

    // f-reject if there are forbidden edges
    REJECT_IF(has_multiple_edge(v3, v4) ||
              has_multiple_edge(v5, v6) ||
              has_edge(v1, v3) ||
              has_edge(v1, v5) ||
              has_edge(v2, v4) ||
              has_edge(v2, v6));

    // perform the switching
    add_simple_edge(v1, v3);
    add_simple_edge(v1, v5);
    add_simple_edge(v2, v4);
    add_simple_edge(v2, v6);
    remove_light_double(v1, v2);
    remove_simple_edge(v3, v4);
    remove_simple_edge(v5, v6);

    // b-rejection
    // compute the lower bounds
    integer b_I_doublet_0_lower = b_doublet_0_lower();
    integer b_I_doublet_1_lower = b_doublet_1_lower();
    // compute the actual number of inverse switchings for the current state
    if (is_heavy(v2)) {
        // swap the two-paths so that we can relax the light two-path second
        std::swap(v1, v2);
        std::swap(v3, v4);
        std::swap(v5, v6);
    }
    // incremental relaxation
    integer bd_GV0 = state_.simple_two_paths;
    integer bd_GV1 = calculate_bd_GV1(v1, v3, v5);
    assert(b_I_doublet_0_lower <= bd_GV0);
    assert(b_I_doublet_1_lower <= bd_GV1);
    assert(bd_GV0 > 0);
    assert(bd_GV1 > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (bd_GV0 * bd_GV1) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_I_doublet_0_lower * b_I_doublet_1_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_III_switching_or_reject(std::mt19937_64 &gen) {
    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_three_stars > 0);
    // pick the structures
    auto [u1, u2, u3, u4] = sample_three_star(gen, false);
    auto [v1, v2, v3, v4] = sample_three_star(gen, true);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, u4, v1, v2, v3, v4}));
    REJECT_IF(has_edge(v1, u1) || // bad edge 1
              has_edge(v2, u2) || // bad edge 2
              has_edge(v3, u3) || // bad edge 3
              has_edge(v4, u4));

    // perform the switching
    add_simple_edge(v1, u1);
    add_simple_edge(v4, u4);
    remove_simple_edge(v1, v4);
    remove_simple_edge(u1, u4);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_III_doublet_0_lower = b_doublet_0_lower();
    integer b_III_doublet_1_lower = b_doublet_1_lower();
    integer b_III_pairs_lower = b_I_to_VII_pairs_lower(1, doubles);
    // incremental relaxation
    integer b_III_GV0 = state_.simple_two_paths;
    integer b_III_GV1 = calculate_bd_GV1(u1, u2, u3);
    // relax additional pairs
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {u1, u2, u3, v1, v2, v3};
    integer b_III_GV_pairs = simple_pairs;
    b_III_GV_pairs -= count_simple_pairs_with_collisions(forbidden_nodes);
    b_III_GV_pairs -= count_simple_pairs_with_forbidden_edges(u1, v1, forbidden_nodes);
    assert(b_III_doublet_0_lower <= b_III_GV0);
    assert(b_III_doublet_1_lower <= b_III_GV1);
    assert(b_III_pairs_lower <= b_III_GV_pairs);
    assert(b_III_GV0 > 0);
    assert(b_III_GV1 > 0);
    assert(b_III_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_III_GV0 * b_III_GV1 * b_III_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_III_doublet_0_lower * b_III_doublet_1_lower * b_III_pairs_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_IIIa_switching_or_reject(std::mt19937_64 &gen) {
    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_four_stars > 0);
    // pick the structures
    auto [u1, u2, u3, u4, u5] = sample_four_star(gen, false);
    auto [v1, v2, v3, v4, v5] = sample_four_star(gen, true);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, u4, u5, v1, v2, v3, v4, v5}));
    REJECT_IF(has_edge(v1, u1) || // bad edge 1
              has_edge(v2, u2) || // bad edge 2
              has_edge(v3, u3) || // bad edge 3
              has_edge(v4, u4) ||
              has_edge(v5, u5));

    // perform the switching
    add_light_double(v1, u1);
    add_simple_edge(v4, u4);
    add_simple_edge(v5, u5);
    remove_simple_edge(v1, v4);
    remove_simple_edge(v1, v5);
    remove_simple_edge(u1, u4);
    remove_simple_edge(u1, u5);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_IIIa_doublet_0_lower = b_doublet_0_lower();
    integer b_IIIa_doublet_1_lower = b_doublet_1_lower();
    integer b_IIIa_pairs_lower = b_I_to_VII_pairs_lower(2, doubles);
    // incremental relaxation
    integer b_IIIa_GV0 = state_.simple_two_paths;
    integer b_IIIa_GV1 = calculate_bd_GV1(u1, u2, u3);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(u4, v4, u1, v1);
    additional_pairs.emplace_back(u5, v5, u1, v1);
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {u1, u2, u3, v1, v2, v3};
    integer b_IIIa_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_IIIa_GV_pair = simple_pairs;
        b_IIIa_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_IIIa_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_IIIa_GV_pairs *= b_IIIa_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_IIIa_doublet_0_lower <= b_IIIa_GV0);
    assert(b_IIIa_doublet_1_lower <= b_IIIa_GV1);
    assert(b_IIIa_pairs_lower <= b_IIIa_GV_pairs);
    assert(b_IIIa_GV0 > 0);
    assert(b_IIIa_GV1 > 0);
    assert(b_IIIa_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_IIIa_GV0 * b_IIIa_GV1 * b_IIIa_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_IIIa_doublet_0_lower * b_IIIa_doublet_1_lower * b_IIIa_pairs_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_IV_switching_or_reject(std::mt19937_64 &gen) {
    // in the paper we draw four two-paths with the centers called u1, u2, v1 and v2
    // to make the variables easier to understand we define u1 := u1, v1 := u2, w1 := v2 and x1 := v1 instead
    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_two_paths > 0);
    // pick the structures
    auto [u1, u2, u3] = sample_two_path(gen, false);
    auto [v1, v2, v3] = sample_two_path(gen, false);
    auto [w1, w2, w3] = sample_two_path(gen, false);
    auto [x1, x2, x3] = sample_two_path(gen, true);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, v1, v2, v3, w1, w2, w3, x1, x2, x3}));
    REJECT_IF(has_edge(u1, v1) || // two-path would contain double-edge
              has_edge(w1, x1) || // two-path would contain double-edge
              has_edge(u1, x1) || // bad edge 1
              has_edge(v1, w1) || // bad edge 2
              has_edge(u3, x3) || // bad edge 3
              has_edge(u2, v2) ||
              has_edge(v3, w3) ||
              has_edge(w2, x2));

    // perform the switching
    add_simple_edge(u1, v1);
    add_simple_edge(v1, w1);
    add_simple_edge(w1, x1);
    add_simple_edge(u2, v2);
    add_simple_edge(v3, w3);
    add_simple_edge(w2, x2);
    remove_simple_edge(u1, u2);
    remove_simple_edge(v1, v2);
    remove_simple_edge(v1, v3);
    remove_simple_edge(w1, w2);
    remove_simple_edge(w1, w3);
    remove_simple_edge(x1, x2);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_IV_doublet_lower = b_doublet_0_lower() * b_doublet_1_lower();
    integer b_IV_pairs_lower = b_I_to_VII_pairs_lower(3, doubles);
    // incremental relaxation
    integer b_IV_GV0 = state_.simple_two_paths;
    integer b_IV_GV1 = calculate_bd_GV1(u1, v1, u3);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(u2, v2, u1, v1);
    additional_pairs.emplace_back(v3, w3, v1, w1);
    additional_pairs.emplace_back(w2, x2, w1, x1);
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, u1, u3, w1, x1, x3};
    integer b_IV_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_IV_GV_pair = simple_pairs;
        b_IV_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_IV_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_IV_GV_pairs *= b_IV_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_IV_doublet_lower <= b_IV_GV0 * b_IV_GV1);
    assert(b_IV_pairs_lower <= b_IV_GV_pairs);
    assert(b_IV_GV0 > 0);
    assert(b_IV_GV1 > 0);
    assert(b_IV_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_IV_GV0 * b_IV_GV1 * b_IV_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_IV_doublet_lower * b_IV_pairs_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_IVa_switching_or_reject(std::mt19937_64 &gen) {
    // analogous to IV the nodes are renamed so that each node of the two-path uses a different letter

    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_two_paths > 0 && state_.simple_three_stars > 0);
    // pick the structures
    auto [u1, u2, u3] = sample_two_path(gen, false);
    auto [v1, v2, v3, v4] = sample_three_star(gen, false);
    auto [w1, w2, w3, w4] = sample_three_star(gen, false);
    auto [x1, x2, x3] = sample_two_path(gen, true);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, v1, v2, v3, v4, w1, w2, w3, w4, x1, x2, x3}));
    REJECT_IF(has_edge(u1, v1) || // two-path would contain double-edge
              has_edge(w1, x1) || // two-path would contain double-edge
              has_edge(u1, x1) || // bad edge 1
              has_edge(v1, w1) || // bad edge 2
              has_edge(u3, x3) || // bad edge 3
              has_edge(u2, v2) ||
              has_edge(v3, w3) ||
              has_edge(v4, w4) ||
              has_edge(w2, x2));

    // perform the switching
    add_simple_edge(u1, v1);
    add_light_double(v1, w1);
    add_simple_edge(w1, x1);
    add_simple_edge(u2, v2);
    add_simple_edge(v3, w3);
    add_simple_edge(v4, w4);
    add_simple_edge(w2, x2);
    remove_simple_edge(u1, u2);
    remove_simple_edge(v1, v2);
    remove_simple_edge(v1, v3);
    remove_simple_edge(v1, v4);
    remove_simple_edge(w1, w2);
    remove_simple_edge(w1, w3);
    remove_simple_edge(w1, w4);
    remove_simple_edge(x1, x2);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_IVa_doublet_lower = b_doublet_0_lower() * b_doublet_1_lower();
    integer b_IVa_pairs_lower = b_I_to_VII_pairs_lower(4, doubles);
    // incremental relaxation
    integer b_IVa_GV0 = state_.simple_two_paths;
    integer b_IVa_GV1 = calculate_bd_GV1(u1, v1, u3);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(u2, v2, u1, v1);
    additional_pairs.emplace_back(v3, w3, v1, w1);
    additional_pairs.emplace_back(v4, w4, v1, w1);
    additional_pairs.emplace_back(w2, x2, w1, x1);
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, u1, u3, w1, x1, x3};
    integer b_IVa_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_IVa_GV_pair = simple_pairs;
        b_IVa_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_IVa_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_IVa_GV_pairs *= b_IVa_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_IVa_doublet_lower <= b_IVa_GV0 * b_IVa_GV1);
    assert(b_IVa_pairs_lower <= b_IVa_GV_pairs);
    assert(b_IVa_GV0 > 0);
    assert(b_IVa_GV1 > 0);
    assert(b_IVa_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_IVa_GV0 * b_IVa_GV1 * b_IVa_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_IVa_doublet_lower * b_IVa_pairs_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_V_switching_or_reject(std::mt19937_64 &gen) {
    // in the paper we draw four structures with the centers called u1, u2, v1 and v2
    // to make the variables easier to understand we define u1 := u1, v1 := u2, w1 := v2 and x1 := v1 instead

    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_three_stars > 0 && state_.simple_two_paths > 0);
    // pick the structures
    auto [u1, u2, u3, u4] = sample_three_star(gen, false);
    auto [v1, v2, v3] = sample_two_path(gen, false);
    auto [w1, w2, w3] = sample_two_path(gen, false);
    auto [x1, x2, x3, x4] = sample_three_star(gen, true);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, u4, v1, v2, v3, w1, w2, w3, x1, x2, x3, x4}));
    REJECT_IF(has_edge(u1, v1) || // two-path would contain double-edge
              has_edge(w1, x1) || // two-path would contain double-edge
              has_edge(u1, x1) || // bad edge 1
              has_edge(v1, w1) || // bad edge 2
              has_edge(u3, x3) || // bad edge 3
              has_edge(u2, v2) ||
              has_edge(u4, v3) ||
              has_edge(w3, x4) ||
              has_edge(w2, x2));

    // perform the switching
    add_simple_edge(u1, v1);
    add_simple_edge(v1, w1);
    add_simple_edge(w1, x1);
    add_simple_edge(x1, u1);
    add_simple_edge(u2, v2);
    add_simple_edge(u4, v3);
    add_simple_edge(w3, x4);
    add_simple_edge(w2, x2);
    remove_simple_edge(u1, u2);
    remove_simple_edge(u1, u4);
    remove_simple_edge(v1, v2);
    remove_simple_edge(v1, v3);
    remove_simple_edge(w1, w2);
    remove_simple_edge(w1, w3);
    remove_simple_edge(x1, x2);
    remove_simple_edge(x1, x4);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_V_doublet_lower = b_doublet_0_lower() * b_doublet_1_lower();
    integer b_V_pairs_lower = b_I_to_VII_pairs_lower(4, doubles);
    // incremental relaxation
    integer b_V_GV0 = state_.simple_two_paths;
    integer b_V_GV1 = calculate_bd_GV1(u1, v1, u3);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(u2, v2, u1, v1);
    additional_pairs.emplace_back(u4, v3, u1, v1);
    additional_pairs.emplace_back(w2, x2, w1, x1);
    additional_pairs.emplace_back(w3, x4, w1, x1);
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, u1, u3, w1, x1, x3};
    integer b_V_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_V_GV_pair = simple_pairs;
        b_V_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_V_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_V_GV_pairs *= b_V_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_V_doublet_lower <= b_V_GV0 * b_V_GV1);
    assert(b_V_pairs_lower <= b_V_GV_pairs);
    assert(b_V_GV0 > 0);
    assert(b_V_GV1 > 0);
    assert(b_V_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_V_GV0 * b_V_GV1 * b_V_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_V_doublet_lower * b_V_pairs_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_Va_switching_or_reject(std::mt19937_64 &gen) {
    // analogous to V the nodes are renamed so that each node of the two-path uses a different letter

    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_four_stars > 0 && state_.simple_two_paths > 0);
    // pick the structures
    auto [u1, u2, u3, u4, u5] = sample_four_star(gen, false);
    auto [v1, v2, v3] = sample_two_path(gen, false);
    auto [w1, w2, w3] = sample_two_path(gen, false);
    auto [x1, x2, x3, x4, x5] = sample_four_star(gen, true);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, u4, u5, v1, v2, v3, w1, w2, w3, x1, x2, x3, x4, x5}));
    REJECT_IF(has_edge(u1, v1) || // two-path would contain double-edge
              has_edge(w1, x1) || // two-path would contain double-edge
              has_edge(u1, x1) || // bad edge 1
              has_edge(v1, w1) || // bad edge 2
              has_edge(u3, x3) || // bad edge 3
              has_edge(u2, v2) ||
              has_edge(u4, v3) ||
              has_edge(w3, x4) ||
              has_edge(w2, x2) ||
              has_edge(u5, x5));

    // perform the switching
    add_simple_edge(u1, v1);
    add_simple_edge(v1, w1);
    add_simple_edge(w1, x1);
    add_light_double(u1, x1);
    add_simple_edge(u2, v2);
    add_simple_edge(u4, v3);
    add_simple_edge(w3, x4);
    add_simple_edge(w2, x2);
    add_simple_edge(u5, x5);
    remove_simple_edge(u1, u2);
    remove_simple_edge(u1, u4);
    remove_simple_edge(u1, u5);
    remove_simple_edge(v1, v2);
    remove_simple_edge(v1, v3);
    remove_simple_edge(w1, w2);
    remove_simple_edge(w1, w3);
    remove_simple_edge(x1, x2);
    remove_simple_edge(x1, x4);
    remove_simple_edge(x1, x5);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_Va_doublet_lower = b_doublet_0_lower() * b_doublet_1_lower();
    integer b_Va_pairs_lower = b_I_to_VII_pairs_lower(5, doubles);
    // incremental relaxation
    integer b_Va_GV0 = state_.simple_two_paths;
    integer b_Va_GV1 = calculate_bd_GV1(u1, v1, u3);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(u2, v2, u1, v1);
    additional_pairs.emplace_back(u4, v3, u1, v1);
    additional_pairs.emplace_back(w2, x2, w1, x1);
    additional_pairs.emplace_back(w3, x4, w1, x1);
    additional_pairs.emplace_back(u5, x5, u1, x1);
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, u1, u3, w1, x1, x3};
    integer b_Va_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_Va_GV_pair = simple_pairs;
        b_Va_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_Va_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_Va_GV_pairs *= b_Va_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_Va_doublet_lower <= b_Va_GV0 * b_Va_GV1);
    assert(b_Va_pairs_lower <= b_Va_GV_pairs);
    assert(b_Va_GV0 > 0);
    assert(b_Va_GV1 > 0);
    assert(b_Va_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_Va_GV0 * b_Va_GV1 * b_Va_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_Va_doublet_lower * b_Va_pairs_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_Vb_switching_or_reject(std::mt19937_64 &gen) {
    // analogous to V the nodes are renamed so that each node of the two-path uses a different letter

    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_three_stars > 0);
    // pick the structures
    auto [u1, u2, u3, u4] = sample_three_star(gen, false);
    auto [v1, v2, v3, v4] = sample_three_star(gen, false);
    auto [w1, w2, w3, w4] = sample_three_star(gen, false);
    auto [x1, x2, x3, x4] = sample_three_star(gen, true);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, u4, v1, v2, v3, v4, w1, w2, w3, w4, x1, x2, x3, x4}));
    REJECT_IF(has_edge(u1, v1) || // two-path would contain double-edge
              has_edge(w1, x1) || // two-path would contain double-edge
              has_edge(u1, x1) || // bad edge 1
              has_edge(v1, w1) || // bad edge 2
              has_edge(u3, x3) || // bad edge 3
              has_edge(u2, v2) ||
              has_edge(u4, v3) ||
              has_edge(w3, x4) ||
              has_edge(w2, x2) ||
              has_edge(v4, w4));

    // perform the switching
    add_simple_edge(u1, v1);
    add_light_double(v1, w1);
    add_simple_edge(w1, x1);
    add_simple_edge(u1, x1);
    add_simple_edge(u2, v2);
    add_simple_edge(u4, v3);
    add_simple_edge(w3, x4);
    add_simple_edge(w2, x2);
    add_simple_edge(v4, w4);
    remove_simple_edge(u1, u2);
    remove_simple_edge(u1, u4);
    remove_simple_edge(v1, v2);
    remove_simple_edge(v1, v3);
    remove_simple_edge(v1, v4);
    remove_simple_edge(w1, w2);
    remove_simple_edge(w1, w3);
    remove_simple_edge(w1, w4);
    remove_simple_edge(x1, x2);
    remove_simple_edge(x1, x4);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_Vb_doublet_lower = b_doublet_0_lower() * b_doublet_1_lower();
    integer b_Vb_pairs_lower = b_I_to_VII_pairs_lower(5, doubles);
    // incremental relaxation
    integer b_Vb_GV0 = state_.simple_two_paths;
    integer b_Vb_GV1 = calculate_bd_GV1(u1, v1, u3);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(u2, v2, u1, v1);
    additional_pairs.emplace_back(u4, v3, u1, v1);
    additional_pairs.emplace_back(w2, x2, w1, x1);
    additional_pairs.emplace_back(w3, x4, w1, x1);
    additional_pairs.emplace_back(v4, w4, v1, w1);
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, u1, u3, w1, x1, x3};
    integer b_Vb_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_Vb_GV_pair = simple_pairs;
        b_Vb_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_Vb_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_Vb_GV_pairs *= b_Vb_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_Vb_doublet_lower <= b_Vb_GV0 * b_Vb_GV1);
    assert(b_Vb_pairs_lower <= b_Vb_GV_pairs);
    assert(b_Vb_GV0 > 0);
    assert(b_Vb_GV1 > 0);
    assert(b_Vb_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_Vb_GV0 * b_Vb_GV1 * b_Vb_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_Vb_doublet_lower * b_Vb_pairs_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_Vc_switching_or_reject(std::mt19937_64 &gen) {
    // analogous to V the nodes are renamed so that each node of the two-path uses a different letter

    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_four_stars > 0 && state_.simple_three_stars > 0);
    // pick the structures
    auto [u1, u2, u3, u4, u5] = sample_four_star(gen, false);
    auto [v1, v2, v3, v4] = sample_three_star(gen, false);
    auto [w1, w2, w3, w4] = sample_three_star(gen, false);
    auto [x1, x2, x3, x4, x5] = sample_four_star(gen, true);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, u4, u5, v1, v2, v3, v4, w1, w2, w3, w4, x1, x2, x3, x4, x5}));
    REJECT_IF(has_edge(u1, v1) || // two-path would contain double-edge
              has_edge(w1, x1) || // two-path would contain double-edge
              has_edge(u1, x1) || // bad edge 1
              has_edge(v1, w1) || // bad edge 2
              has_edge(u3, x3) || // bad edge 3
              has_edge(u2, v2) ||
              has_edge(u4, v3) ||
              has_edge(w3, x4) ||
              has_edge(w2, x2) ||
              has_edge(v4, w4) ||
              has_edge(u5, x5));

    // perform the switching
    add_simple_edge(u1, v1);
    add_light_double(v1, w1);
    add_simple_edge(w1, x1);
    add_light_double(u1, x1);
    add_simple_edge(u2, v2);
    add_simple_edge(u4, v3);
    add_simple_edge(w3, x4);
    add_simple_edge(w2, x2);
    add_simple_edge(u5, x5);
    remove_simple_edge(u1, u2);
    remove_simple_edge(u1, u4);
    remove_simple_edge(u1, u5);
    remove_simple_edge(v1, v2);
    remove_simple_edge(v1, v3);
    remove_simple_edge(v1, v4);
    remove_simple_edge(w1, w2);
    remove_simple_edge(w1, w3);
    remove_simple_edge(w1, w4);
    remove_simple_edge(x1, x2);
    remove_simple_edge(x1, x4);
    remove_simple_edge(x1, x5);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_Vc_doublet_lower = b_doublet_0_lower() * b_doublet_1_lower();
    integer b_Vc_pairs_lower = b_I_to_VII_pairs_lower(6, doubles);
    // incremental relaxation
    integer b_Vc_GV0 = state_.simple_two_paths;
    integer b_Vc_GV1 = calculate_bd_GV1(u1, v1, u3);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(u2, v2, u1, v1);
    additional_pairs.emplace_back(u4, v3, u1, v1);
    additional_pairs.emplace_back(w2, x2, w1, x1);
    additional_pairs.emplace_back(w3, x4, w1, x1);
    additional_pairs.emplace_back(v4, w4, v1, w1);
    additional_pairs.emplace_back(u5, x5, u1, x1);
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, u1, u3, w1, x1, x3};
    integer b_Vc_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_Vc_GV_pair = simple_pairs;
        b_Vc_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_Vc_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_Vc_GV_pairs *= b_Vc_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_Vc_doublet_lower <= b_Vc_GV0 * b_Vc_GV1);
    assert(b_Vc_pairs_lower <= b_Vc_GV_pairs);
    assert(b_Vc_GV0 > 0);
    assert(b_Vc_GV1 > 0);
    assert(b_Vc_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_Vc_GV0 * b_Vc_GV1 * b_Vc_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_Vc_doublet_lower * b_Vc_pairs_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_VI_switching_or_reject(std::mt19937_64 &gen) {
    // in the paper we draw six two-paths with the centers called u1, u2, u3, v1, v2 and v3
    // to simplify we define u1 := u1, v1 := u2, w1 := v2, x1 := v1, y1 := v3 and z1 := u3
    // the left node on the path centered at u1 will be u2, the right node u3 etc.

    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_two_paths > 0);
    // pick the structures
    auto [u1, u2, u3] = sample_two_path(gen, false);
    auto [v1, v2, v3] = sample_two_path(gen, false);
    auto [w1, w2, w3] = sample_two_path(gen, false);
    auto [x1, x2, x3] = sample_two_path(gen, true);
    auto [y1, y2, y3] = sample_two_path(gen, false);
    auto [z1, z2, z3] = sample_two_path(gen, false);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, v1, v2, v3, w1, w2, w3, x1, x2, x3, y1, y2, y3, z1, z2, z3}));
    REJECT_IF(has_edge(u1, v1) || // two-path would contain double-edge
              has_edge(w1, x1) || // two-path would contain double-edge
              has_edge(x1, y1) || // two-path would contain double-edge
              has_edge(z1, u1) || // two-path would contain double-edge
              has_edge(u1, x1) || // bad edge 1
              has_edge(v1, w1) || // bad edge 2
              has_edge(y1, z1) || // bad edge 3
              has_edge(u2, v3) ||
              has_edge(v2, w2) ||
              has_edge(w3, x2) ||
              has_edge(x3, y2) ||
              has_edge(y3, z3) ||
              has_edge(z2, u3));

    // perform the switching
    add_simple_edge(u1, v1);
    add_simple_edge(v1, w1);
    add_simple_edge(w1, x1);
    add_simple_edge(x1, y1);
    add_simple_edge(y1, z1);
    add_simple_edge(z1, u1);
    add_simple_edge(u2, v3);
    add_simple_edge(v2, w2);
    add_simple_edge(w3, x2);
    add_simple_edge(x3, y2);
    add_simple_edge(y3, z3);
    add_simple_edge(z2, u3);
    remove_simple_edge(u1, u2);
    remove_simple_edge(u1, u3);
    remove_simple_edge(v1, v2);
    remove_simple_edge(v1, v3);
    remove_simple_edge(w1, w2);
    remove_simple_edge(w1, w3);
    remove_simple_edge(x1, x2);
    remove_simple_edge(x1, x3);
    remove_simple_edge(y1, y2);
    remove_simple_edge(y1, y3);
    remove_simple_edge(z1, z2);
    remove_simple_edge(z1, z3);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_VI_doublet_lower = b_doublet_0_lower() * b_doublet_1_lower();
    integer b_VI_pairs_lower = b_I_to_VII_pairs_lower(6, doubles);
    // incremental relaxation
    integer b_VI_GV0 = state_.simple_two_paths;
    integer b_VI_GV1 = calculate_bd_GV1(u1, v1, z1);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(u2, v3, u1, v1);
    additional_pairs.emplace_back(v2, w2, v1, w1);
    additional_pairs.emplace_back(w3, x2, w1, x1);
    additional_pairs.emplace_back(x3, y2, x1, y1);
    additional_pairs.emplace_back(y3, z3, y1, z1);
    additional_pairs.emplace_back(z2, u3, z1, u1);
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, u1, z1, w1, x1, y1};
    integer b_VI_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_VI_GV_pair = simple_pairs;
        b_VI_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_VI_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_VI_GV_pairs *= b_VI_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_VI_doublet_lower <= b_VI_GV0 * b_VI_GV1);
    assert(b_VI_pairs_lower <= b_VI_GV_pairs);
    assert(b_VI_GV0 > 0);
    assert(b_VI_GV1 > 0);
    assert(b_VI_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_VI_GV0 * b_VI_GV1 * b_VI_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_VI_doublet_lower * b_VI_pairs_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_VIa_switching_or_reject(std::mt19937_64 &gen) {
    // analogous to VI the nodes are renamed so that each node of the two-path uses a different letter

    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_two_paths > 0 && state_.simple_three_stars > 0);
    // pick the structures
    auto [u1, u2, u3] = sample_two_path(gen, false);
    auto [v1, v2, v3, v4] = sample_three_star(gen, false);
    auto [w1, w2, w3, w4] = sample_three_star(gen, false);
    auto [x1, x2, x3] = sample_two_path(gen, true);
    auto [y1, y2, y3] = sample_two_path(gen, false);
    auto [z1, z2, z3] = sample_two_path(gen, false);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, v1, v2, v3, v4, w1, w2, w3, w4, x1, x2, x3, y1, y2, y3, z1, z2, z3}));
    REJECT_IF(has_edge(u1, v1) || // two-path would contain double-edge
              has_edge(w1, x1) || // two-path would contain double-edge
              has_edge(x1, y1) || // two-path would contain double-edge
              has_edge(z1, u1) || // two-path would contain double-edge
              has_edge(u1, x1) || // bad edge 1
              has_edge(v1, w1) || // bad edge 2
              has_edge(y1, z1) || // bad edge 3
              has_edge(u2, v3) ||
              has_edge(v2, w2) ||
              has_edge(w3, x2) ||
              has_edge(x3, y2) ||
              has_edge(y3, z3) ||
              has_edge(z2, u3) ||
              has_edge(v4, w4));

    // perform the switching
    add_simple_edge(u1, v1);
    add_light_double(v1, w1);
    add_simple_edge(w1, x1);
    add_simple_edge(x1, y1);
    add_simple_edge(y1, z1);
    add_simple_edge(z1, u1);
    add_simple_edge(u2, v3);
    add_simple_edge(v2, w2);
    add_simple_edge(w3, x2);
    add_simple_edge(x3, y2);
    add_simple_edge(y3, z3);
    add_simple_edge(z2, u3);
    add_simple_edge(v4, w4);
    remove_simple_edge(u1, u2);
    remove_simple_edge(u1, u3);
    remove_simple_edge(v1, v2);
    remove_simple_edge(v1, v3);
    remove_simple_edge(v1, v4);
    remove_simple_edge(w1, w2);
    remove_simple_edge(w1, w3);
    remove_simple_edge(w1, w4);
    remove_simple_edge(x1, x2);
    remove_simple_edge(x1, x3);
    remove_simple_edge(y1, y2);
    remove_simple_edge(y1, y3);
    remove_simple_edge(z1, z2);
    remove_simple_edge(z1, z3);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_VIa_doublet_lower = b_doublet_0_lower() * b_doublet_1_lower();
    integer b_VIa_pairs_lower = b_I_to_VII_pairs_lower(7, doubles);
    // incremental relaxation
    integer b_VIa_GV0 = state_.simple_two_paths;
    integer b_VIa_GV1 = calculate_bd_GV1(u1, v1, z1);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(u2, v3, u1, v1);
    additional_pairs.emplace_back(v2, w2, v1, w1);
    additional_pairs.emplace_back(w3, x2, w1, x1);
    additional_pairs.emplace_back(x3, y2, x1, y1);
    additional_pairs.emplace_back(y3, z3, y1, z1);
    additional_pairs.emplace_back(z2, u3, z1, u1);
    additional_pairs.emplace_back(v4, w4, v1, w1);
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, u1, z1, w1, x1, y1};
    integer b_VIa_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_VIa_GV_pair = simple_pairs;
        b_VIa_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_VIa_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_VIa_GV_pairs *= b_VIa_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_VIa_doublet_lower <= b_VIa_GV0 * b_VIa_GV1);
    assert(b_VIa_pairs_lower <= b_VIa_GV_pairs);
    assert(b_VIa_GV0 > 0);
    assert(b_VIa_GV1 > 0);
    assert(b_VIa_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_VIa_GV0 * b_VIa_GV1 * b_VIa_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_VIa_doublet_lower * b_VIa_pairs_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_VIb_switching_or_reject(std::mt19937_64 &gen) {
    // analogous to VI the nodes are renamed so that each node of the two-path uses a different letter

    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_two_paths > 0 && state_.simple_three_stars > 0);
    // pick the structures
    auto [u1, u2, u3] = sample_two_path(gen, false);
    auto [v1, v2, v3, v4] = sample_three_star(gen, false);
    auto [w1, w2, w3, w4] = sample_three_star(gen, false);
    auto [x1, x2, x3] = sample_two_path(gen, true);
    auto [y1, y2, y3, y4] = sample_three_star(gen, false);
    auto [z1, z2, z3, z4] = sample_three_star(gen, false);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, v1, v2, v3, v4, w1, w2, w3, w4, x1, x2, x3, y1, y2, y3, y4, z1, z2, z3, z4}));
    REJECT_IF(has_edge(u1, v1) || // two-path would contain double-edge
              has_edge(w1, x1) || // two-path would contain double-edge
              has_edge(x1, y1) || // two-path would contain double-edge
              has_edge(z1, u1) || // two-path would contain double-edge
              has_edge(u1, x1) || // bad edge 1
              has_edge(v1, w1) || // bad edge 2
              has_edge(y1, z1) || // bad edge 3
              has_edge(u2, v3) ||
              has_edge(v2, w2) ||
              has_edge(w3, x2) ||
              has_edge(x3, y2) ||
              has_edge(y3, z3) ||
              has_edge(z2, u3) ||
              has_edge(v4, w4) ||
              has_edge(y4, z4));

    // perform the switching
    add_simple_edge(u1, v1);
    add_light_double(v1, w1);
    add_simple_edge(w1, x1);
    add_simple_edge(x1, y1);
    add_light_double(y1, z1);
    add_simple_edge(z1, u1);
    add_simple_edge(u2, v3);
    add_simple_edge(v2, w2);
    add_simple_edge(w3, x2);
    add_simple_edge(x3, y2);
    add_simple_edge(y3, z3);
    add_simple_edge(z2, u3);
    add_simple_edge(v4, w4);
    add_simple_edge(y4, z4);
    remove_simple_edge(u1, u2);
    remove_simple_edge(u1, u3);
    remove_simple_edge(v1, v2);
    remove_simple_edge(v1, v3);
    remove_simple_edge(v1, v4);
    remove_simple_edge(w1, w2);
    remove_simple_edge(w1, w3);
    remove_simple_edge(w1, w4);
    remove_simple_edge(x1, x2);
    remove_simple_edge(x1, x3);
    remove_simple_edge(y1, y2);
    remove_simple_edge(y1, y3);
    remove_simple_edge(y1, y4);
    remove_simple_edge(z1, z2);
    remove_simple_edge(z1, z3);
    remove_simple_edge(z1, z4);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_VIb_doublet_lower = b_doublet_0_lower() * b_doublet_1_lower();
    integer b_VIb_pairs_lower = b_I_to_VII_pairs_lower(8, doubles);
    // incremental relaxation
    integer b_VIb_GV0 = state_.simple_two_paths;
    integer b_VIb_GV1 = calculate_bd_GV1(u1, v1, z1);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(u2, v3, u1, v1);
    additional_pairs.emplace_back(v2, w2, v1, w1);
    additional_pairs.emplace_back(w3, x2, w1, x1);
    additional_pairs.emplace_back(x3, y2, x1, y1);
    additional_pairs.emplace_back(y3, z3, y1, z1);
    additional_pairs.emplace_back(z2, u3, z1, u1);
    additional_pairs.emplace_back(v4, w4, v1, w1);
    additional_pairs.emplace_back(y4, z4, y1, z1);
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, u1, z1, w1, x1, y1};
    integer b_VIb_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_VIb_GV_pair = simple_pairs;
        b_VIb_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_VIb_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_VIb_GV_pairs *= b_VIb_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_VIb_doublet_lower <= b_VIb_GV0 * b_VIb_GV1);
    assert(b_VIb_pairs_lower <= b_VIb_GV_pairs);
    assert(b_VIb_GV0 > 0);
    assert(b_VIb_GV1 > 0);
    assert(b_VIb_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_VIb_GV0 * b_VIb_GV1 * b_VIb_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_VIb_doublet_lower * b_VIb_pairs_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_VII_switching_or_reject(std::mt19937_64 &gen) {
    // in the paper we draw six structures with the centers called u1, u2, u3, v1, v2 and v3
    // to simplify we define u1 := u1, v1 := u2, w1 := v2, x1 := v1, y1 := v3 and z1 := u3
    // the other nodes attached to each center node will be called u2, u3...etc

    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_three_stars > 0 && state_.simple_two_paths > 0);
    // pick the structures
    auto [u1, u2, u3, u4] = sample_three_star(gen, false);
    auto [v1, v2, v3] = sample_two_path(gen, false);
    auto [w1, w2, w3] = sample_two_path(gen, false);
    auto [x1, x2, x3, x4] = sample_three_star(gen, true);
    auto [y1, y2, y3] = sample_two_path(gen, false);
    auto [z1, z2, z3] = sample_two_path(gen, false);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, u4, v1, v2, v3, w1, w2, w3, x1, x2, x3, x4, y1, y2, y3, z1, z2, z3}));
    REJECT_IF(has_edge(u1, v1) || // two-path would contain double-edge
              has_edge(w1, x1) || // two-path would contain double-edge
              has_edge(x1, y1) || // two-path would contain double-edge
              has_edge(z1, u1) || // two-path would contain double-edge
              has_edge(u1, x1) || // bad edge 1
              has_edge(v1, w1) || // bad edge 2
              has_edge(y1, z1) || // bad edge 3
              has_edge(u2, v3) ||
              has_edge(v2, w2) ||
              has_edge(w3, x2) ||
              has_edge(x3, y3) ||
              has_edge(x4, y2) ||
              has_edge(z3, u3) ||
              has_edge(z2, u4));

    // perform the switching
    add_simple_edge(u1, v1);
    add_simple_edge(v1, w1);
    add_simple_edge(w1, x1);
    add_simple_edge(x1, y1);
    add_simple_edge(y1, z1);
    add_simple_edge(z1, u1);
    add_simple_edge(u1, x1);
    add_simple_edge(u2, v3);
    add_simple_edge(v2, w2);
    add_simple_edge(w3, x2);
    add_simple_edge(x3, y3);
    add_simple_edge(x4, y2);
    add_simple_edge(z3, u3);
    add_simple_edge(z2, u4);
    remove_simple_edge(u1, u2);
    remove_simple_edge(u1, u3);
    remove_simple_edge(u1, u4);
    remove_simple_edge(v1, v2);
    remove_simple_edge(v1, v3);
    remove_simple_edge(w1, w2);
    remove_simple_edge(w1, w3);
    remove_simple_edge(x1, x2);
    remove_simple_edge(x1, x3);
    remove_simple_edge(x1, x4);
    remove_simple_edge(y1, y2);
    remove_simple_edge(y1, y3);
    remove_simple_edge(z1, z2);
    remove_simple_edge(z1, z3);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_VII_doublet_lower = b_doublet_0_lower() * b_doublet_1_lower();
    integer b_VII_pairs_lower = b_I_to_VII_pairs_lower(7, doubles);
    // incremental relaxation
    integer b_VII_GV0 = state_.simple_two_paths;
    integer b_VII_GV1 = calculate_bd_GV1(u1, v1, z1);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(u2, v3, u1, v1);
    additional_pairs.emplace_back(v2, w2, v1, w1);
    additional_pairs.emplace_back(w3, x2, w1, x1);
    additional_pairs.emplace_back(x3, y3, x1, y1);
    additional_pairs.emplace_back(x4, y2, x1, y1);
    additional_pairs.emplace_back(z3, u3, z1, u1);
    additional_pairs.emplace_back(z2, u4, z1, u1);
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, u1, z1, w1, x1, y1};
    integer b_VII_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_VII_GV_pair = simple_pairs;
        b_VII_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_VII_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_VII_GV_pairs *= b_VII_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_VII_doublet_lower <= b_VII_GV0 * b_VII_GV1);
    assert(b_VII_pairs_lower <= b_VII_GV_pairs);
    assert(b_VII_GV0 > 0);
    assert(b_VII_GV1 > 0);
    assert(b_VII_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_VII_GV0 * b_VII_GV1 * b_VII_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_VII_doublet_lower * b_VII_pairs_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_VIIa_switching_or_reject(std::mt19937_64 &gen) {
    // analogous to VII the nodes are renamed so that each node of the two-path uses a different letter

    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_three_stars > 0 && state_.simple_two_paths > 0);
    // pick the structures
    auto [u1, u2, u3, u4] = sample_three_star(gen, false);
    auto [v1, v2, v3, v4] = sample_three_star(gen, false);
    auto [w1, w2, w3, w4] = sample_three_star(gen, false);
    auto [x1, x2, x3, x4] = sample_three_star(gen, true);

    // pick another two light or heavy two-paths
    auto [y1, y2, y3] = sample_two_path(gen, false);
    auto [z1, z2, z3] = sample_two_path(gen, false);
    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, u4, v1, v2, v3, v4, w1, w2, w3, w4, x1, x2, x3, x4, y1, y2, y3, z1, z2, z3}));
    REJECT_IF(has_edge(u1, v1) || // two-path would contain double-edge
              has_edge(w1, x1) || // two-path would contain double-edge
              has_edge(x1, y1) || // two-path would contain double-edge
              has_edge(z1, u1) || // two-path would contain double-edge
              has_edge(u1, x1) || // bad edge 1
              has_edge(v1, w1) || // bad edge 2
              has_edge(y1, z1) || // bad edge 3
              has_edge(u2, v3) ||
              has_edge(v2, w2) ||
              has_edge(w3, x2) ||
              has_edge(x3, y3) ||
              has_edge(x4, y2) ||
              has_edge(z3, u3) ||
              has_edge(z2, u4) ||
              has_edge(v4, w4));

    // perform the switching
    add_simple_edge(u1, v1);
    add_light_double(v1, w1);
    add_simple_edge(w1, x1);
    add_simple_edge(x1, y1);
    add_simple_edge(y1, z1);
    add_simple_edge(z1, u1);
    add_simple_edge(u1, x1);
    add_simple_edge(u2, v3);
    add_simple_edge(v2, w2);
    add_simple_edge(w3, x2);
    add_simple_edge(x3, y3);
    add_simple_edge(x4, y2);
    add_simple_edge(z3, u3);
    add_simple_edge(z2, u4);
    add_simple_edge(v4, w4);
    remove_simple_edge(u1, u2);
    remove_simple_edge(u1, u3);
    remove_simple_edge(u1, u4);
    remove_simple_edge(v1, v2);
    remove_simple_edge(v1, v3);
    remove_simple_edge(v1, v4);
    remove_simple_edge(w1, w2);
    remove_simple_edge(w1, w3);
    remove_simple_edge(w1, w4);
    remove_simple_edge(x1, x2);
    remove_simple_edge(x1, x3);
    remove_simple_edge(x1, x4);
    remove_simple_edge(y1, y2);
    remove_simple_edge(y1, y3);
    remove_simple_edge(z1, z2);
    remove_simple_edge(z1, z3);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_VIIa_doublet_lower = b_doublet_0_lower() * b_doublet_1_lower();
    integer b_VIIa_pairs_lower = b_I_to_VII_pairs_lower(8, doubles);
    // incremental relaxation
    integer b_VIIa_GV0 = state_.simple_two_paths;
    integer b_VIIa_GV1 = calculate_bd_GV1(u1, v1, z1);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(u2, v3, u1, v1);
    additional_pairs.emplace_back(v2, w2, v1, w1);
    additional_pairs.emplace_back(w3, x2, w1, x1);
    additional_pairs.emplace_back(x3, y3, x1, y1);
    additional_pairs.emplace_back(x4, y2, x1, y1);
    additional_pairs.emplace_back(z3, u3, z1, u1);
    additional_pairs.emplace_back(z2, u4, z1, u1);
    additional_pairs.emplace_back(v4, w4, v1, w1);
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, u1, z1, w1, x1, y1};
    integer b_VIIa_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_VIIa_GV_pair = simple_pairs;
        b_VIIa_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_VIIa_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_VIIa_GV_pairs *= b_VIIa_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_VIIa_doublet_lower <= b_VIIa_GV0 * b_VIIa_GV1);
    assert(b_VIIa_pairs_lower <= b_VIIa_GV_pairs);
    assert(b_VIIa_GV0 > 0);
    assert(b_VIIa_GV1 > 0);
    assert(b_VIIa_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_VIIa_GV0 * b_VIIa_GV1 * b_VIIa_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_VIIa_doublet_lower * b_VIIa_pairs_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_VIIb_switching_or_reject(std::mt19937_64 &gen) {
    // analogous to VII the nodes are renamed so that each node of the two-path uses a different letter

    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_four_stars > 0 && state_.simple_two_paths > 0);
    // pick the structures
    auto [u1, u2, u3, u4, u5] = sample_four_star(gen, false);
    auto [v1, v2, v3] = sample_two_path(gen, false);
    auto [w1, w2, w3] = sample_two_path(gen, false);
    auto [x1, x2, x3, x4, x5] = sample_four_star(gen, true);
    auto [y1, y2, y3] = sample_two_path(gen, false);
    auto [z1, z2, z3] = sample_two_path(gen, false);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, u4, u5, v1, v2, v3, w1, w2, w3, x1, x2, x3, x4, x5, y1, y2, y3, z1, z2, z3}));
    REJECT_IF(has_edge(u1, v1) || // two-path would contain double-edge
              has_edge(w1, x1) || // two-path would contain double-edge
              has_edge(x1, y1) || // two-path would contain double-edge
              has_edge(z1, u1) || // two-path would contain double-edge
              has_edge(u1, x1) || // bad edge 1
              has_edge(v1, w1) || // bad edge 2
              has_edge(y1, z1) || // bad edge 3
              has_edge(u2, v3) ||
              has_edge(v2, w2) ||
              has_edge(w3, x2) ||
              has_edge(x3, y3) ||
              has_edge(x4, y2) ||
              has_edge(z3, u3) ||
              has_edge(z2, u4) ||
              has_edge(u5, x5));

    // perform the switching
    add_simple_edge(u1, v1);
    add_simple_edge(v1, w1);
    add_simple_edge(w1, x1);
    add_simple_edge(x1, y1);
    add_simple_edge(y1, z1);
    add_simple_edge(z1, u1);
    add_light_double(u1, x1);
    add_simple_edge(u2, v3);
    add_simple_edge(v2, w2);
    add_simple_edge(w3, x2);
    add_simple_edge(x3, y3);
    add_simple_edge(x4, y2);
    add_simple_edge(z3, u3);
    add_simple_edge(z2, u4);
    add_simple_edge(u5, x5);
    remove_simple_edge(u1, u2);
    remove_simple_edge(u1, u3);
    remove_simple_edge(u1, u4);
    remove_simple_edge(u1, u5);
    remove_simple_edge(v1, v2);
    remove_simple_edge(v1, v3);
    remove_simple_edge(w1, w2);
    remove_simple_edge(w1, w3);
    remove_simple_edge(x1, x2);
    remove_simple_edge(x1, x3);
    remove_simple_edge(x1, x4);
    remove_simple_edge(x1, x5);
    remove_simple_edge(y1, y2);
    remove_simple_edge(y1, y3);
    remove_simple_edge(z1, z2);
    remove_simple_edge(z1, z3);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_VIIb_doublet_lower = b_doublet_0_lower() * b_doublet_1_lower();
    integer b_VIIb_pairs_lower = b_I_to_VII_pairs_lower(8, doubles);
    // incremental relaxation
    integer b_VIIb_GV0 = state_.simple_two_paths;
    integer b_VIIb_GV1 = calculate_bd_GV1(u1, v1, z1);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(u2, v3, u1, v1);
    additional_pairs.emplace_back(v2, w2, v1, w1);
    additional_pairs.emplace_back(w3, x2, w1, x1);
    additional_pairs.emplace_back(x3, y3, x1, y1);
    additional_pairs.emplace_back(x4, y2, x1, y1);
    additional_pairs.emplace_back(z3, u3, z1, u1);
    additional_pairs.emplace_back(z2, u4, z1, u1);
    additional_pairs.emplace_back(u5, x5, u1, x1);
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, u1, z1, w1, x1, y1};
    integer b_VIIb_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_VIIb_GV_pair = simple_pairs;
        b_VIIb_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_VIIb_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_VIIb_GV_pairs *= b_VIIb_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_VIIb_doublet_lower <= b_VIIb_GV0 * b_VIIb_GV1);
    assert(b_VIIb_pairs_lower <= b_VIIb_GV_pairs);
    assert(b_VIIb_GV0 > 0);
    assert(b_VIIb_GV1 > 0);
    assert(b_VIIb_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_VIIb_GV0 * b_VIIb_GV1 * b_VIIb_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_VIIb_doublet_lower * b_VIIb_pairs_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_VIIc_switching_or_reject(std::mt19937_64 &gen) {
    // analogous to VII the nodes are renamed so that each node of the two-path uses a different letter

    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_four_stars > 0 && state_.simple_three_stars > 0);
    // pick the structures
    auto [u1, u2, u3, u4, u5] = sample_four_star(gen, false);
    auto [v1, v2, v3, v4] = sample_three_star(gen, false);
    auto [w1, w2, w3, w4] = sample_three_star(gen, false);
    auto [x1, x2, x3, x4, x5] = sample_four_star(gen, true);
    auto [y1, y2, y3] = sample_two_path(gen, false);
    auto [z1, z2, z3] = sample_two_path(gen, false);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, u4, u5, v1, v2, v3, v4, w1, w2, w3, w4,
                              x1, x2, x3, x4, x5, y1, y2, y3, z1, z2, z3}));
    REJECT_IF(has_edge(u1, v1) || // two-path would contain double-edge
              has_edge(w1, x1) || // two-path would contain double-edge
              has_edge(x1, y1) || // two-path would contain double-edge
              has_edge(z1, u1) || // two-path would contain double-edge
              has_edge(u1, x1) || // bad edge 1
              has_edge(v1, w1) || // bad edge 2
              has_edge(y1, z1) || // bad edge 3
              has_edge(u2, v3) ||
              has_edge(v2, w2) ||
              has_edge(w3, x2) ||
              has_edge(x3, y3) ||
              has_edge(x4, y2) ||
              has_edge(z3, u3) ||
              has_edge(z2, u4) ||
              has_edge(v4, w4) ||
              has_edge(u5, x5));

    // perform the switching
    add_simple_edge(u1, v1);
    add_light_double(v1, w1);
    add_simple_edge(w1, x1);
    add_simple_edge(x1, y1);
    add_simple_edge(y1, z1);
    add_simple_edge(z1, u1);
    add_light_double(u1, x1);
    add_simple_edge(u2, v3);
    add_simple_edge(v2, w2);
    add_simple_edge(w3, x2);
    add_simple_edge(x3, y3);
    add_simple_edge(x4, y2);
    add_simple_edge(z3, u3);
    add_simple_edge(z2, u4);
    add_simple_edge(v4, w4);
    add_simple_edge(u5, x5);
    remove_simple_edge(u1, u2);
    remove_simple_edge(u1, u3);
    remove_simple_edge(u1, u4);
    remove_simple_edge(u1, u5);
    remove_simple_edge(v1, v2);
    remove_simple_edge(v1, v3);
    remove_simple_edge(v1, v4);
    remove_simple_edge(w1, w2);
    remove_simple_edge(w1, w3);
    remove_simple_edge(w1, w4);
    remove_simple_edge(x1, x2);
    remove_simple_edge(x1, x3);
    remove_simple_edge(x1, x4);
    remove_simple_edge(x1, x5);
    remove_simple_edge(y1, y2);
    remove_simple_edge(y1, y3);
    remove_simple_edge(z1, z2);
    remove_simple_edge(z1, z3);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_VIIc_doublet_lower = b_doublet_0_lower() * b_doublet_1_lower();
    integer b_VIIc_pairs_lower = b_I_to_VII_pairs_lower(9, doubles);
    // incremental relaxation
    integer b_VIIc_GV0 = state_.simple_two_paths;
    integer b_VIIc_GV1 = calculate_bd_GV1(u1, v1, z1);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(u2, v3, u1, v1);
    additional_pairs.emplace_back(v2, w2, v1, w1);
    additional_pairs.emplace_back(w3, x2, w1, x1);
    additional_pairs.emplace_back(x3, y3, x1, y1);
    additional_pairs.emplace_back(x4, y2, x1, y1);
    additional_pairs.emplace_back(z3, u3, z1, u1);
    additional_pairs.emplace_back(z2, u4, z1, u1);
    additional_pairs.emplace_back(v4, w4, v1, w1);
    additional_pairs.emplace_back(u5, x5, u1, x1);
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, u1, z1, w1, x1, y1};
    integer b_VIIc_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_VIIc_GV_pair = simple_pairs;
        b_VIIc_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_VIIc_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_VIIc_GV_pairs *= b_VIIc_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_VIIc_doublet_lower <= b_VIIc_GV0 * b_VIIc_GV1);
    assert(b_VIIc_pairs_lower <= b_VIIc_GV_pairs);
    assert(b_VIIc_GV0 > 0);
    assert(b_VIIc_GV1 > 0);
    assert(b_VIIc_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_VIIc_GV0 * b_VIIc_GV1 * b_VIIc_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_VIIc_doublet_lower * b_VIIc_pairs_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_VIId_switching_or_reject(std::mt19937_64 &gen) {
    // analogous to VII the nodes are renamed so that each node of the two-path uses a different letter

    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_three_stars > 0 && state_.simple_two_paths > 0);
    // pick the structures
    auto [u1, u2, u3, u4] = sample_three_star(gen, false);
    auto [v1, v2, v3, v4] = sample_three_star(gen, false);
    auto [w1, w2, w3, w4] = sample_three_star(gen, false);
    auto [x1, x2, x3, x4] = sample_three_star(gen, true);
    auto [y1, y2, y3, y4] = sample_three_star(gen, false);
    auto [z1, z2, z3, z4] = sample_three_star(gen, false);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, u4, v1, v2, v3, v4, w1, w2, w3, w4,
                              x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4}));
    REJECT_IF(has_edge(u1, v1) || // two-path would contain double-edge
              has_edge(w1, x1) || // two-path would contain double-edge
              has_edge(x1, y1) || // two-path would contain double-edge
              has_edge(z1, u1) || // two-path would contain double-edge
              has_edge(u1, x1) || // bad edge 1
              has_edge(v1, w1) || // bad edge 2
              has_edge(y1, z1) || // bad edge 3
              has_edge(u2, v3) ||
              has_edge(v2, w2) ||
              has_edge(w3, x2) ||
              has_edge(x3, y3) ||
              has_edge(x4, y2) ||
              has_edge(z3, u3) ||
              has_edge(z2, u4) ||
              has_edge(v4, w4) ||
              has_edge(y4, z4));

    // perform the switching
    add_simple_edge(u1, v1);
    add_light_double(v1, w1);
    add_simple_edge(w1, x1);
    add_simple_edge(x1, y1);
    add_light_double(y1, z1);
    add_simple_edge(z1, u1);
    add_simple_edge(u1, x1);
    add_simple_edge(u2, v3);
    add_simple_edge(v2, w2);
    add_simple_edge(w3, x2);
    add_simple_edge(x3, y3);
    add_simple_edge(x4, y2);
    add_simple_edge(z3, u3);
    add_simple_edge(z2, u4);
    add_simple_edge(v4, w4);
    add_simple_edge(y4, z4);
    remove_simple_edge(u1, u2);
    remove_simple_edge(u1, u3);
    remove_simple_edge(u1, u4);
    remove_simple_edge(v1, v2);
    remove_simple_edge(v1, v3);
    remove_simple_edge(v1, v4);
    remove_simple_edge(w1, w2);
    remove_simple_edge(w1, w3);
    remove_simple_edge(w1, w4);
    remove_simple_edge(x1, x2);
    remove_simple_edge(x1, x3);
    remove_simple_edge(x1, x4);
    remove_simple_edge(y1, y2);
    remove_simple_edge(y1, y3);
    remove_simple_edge(y1, y4);
    remove_simple_edge(z1, z2);
    remove_simple_edge(z1, z3);
    remove_simple_edge(z1, z4);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_VIId_doublet_lower = b_doublet_0_lower() * b_doublet_1_lower();
    integer b_VIId_pairs_lower = b_I_to_VII_pairs_lower(9, doubles);
    // incremental relaxation
    integer b_VIId_GV0 = state_.simple_two_paths;
    integer b_VIId_GV1 = calculate_bd_GV1(u1, v1, z1);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(u2, v3, u1, v1);
    additional_pairs.emplace_back(v2, w2, v1, w1);
    additional_pairs.emplace_back(w3, x2, w1, x1);
    additional_pairs.emplace_back(x3, y3, x1, y1);
    additional_pairs.emplace_back(x4, y2, x1, y1);
    additional_pairs.emplace_back(z3, u3, z1, u1);
    additional_pairs.emplace_back(z2, u4, z1, u1);
    additional_pairs.emplace_back(v4, w4, v1, w1);
    additional_pairs.emplace_back(y4, z4, y1, z1);
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, u1, z1, w1, x1, y1};
    integer b_VIId_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_VIId_GV_pair = simple_pairs;
        b_VIId_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_VIId_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_VIId_GV_pairs *= b_VIId_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_VIId_doublet_lower <= b_VIId_GV0 * b_VIId_GV1);
    assert(b_VIId_pairs_lower <= b_VIId_GV_pairs);
    assert(b_VIId_GV0 > 0);
    assert(b_VIId_GV1 > 0);
    assert(b_VIId_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_VIId_GV0 * b_VIId_GV1 * b_VIId_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_VIId_doublet_lower * b_VIId_pairs_lower);
    ACCEPT;
}

bool IncPowerlawGraphSampler::perform_type_VIIe_switching_or_reject(std::mt19937_64 &gen) {
    // analogous to VII the nodes are renamed so that each node of the two-path uses a different letter

    // f-reject if we don't have the structures needed for the switching
    REJECT_UNLESS(state_.light_simple_four_stars > 0 && state_.simple_three_stars > 0);
    // pick the structures
    auto [u1, u2, u3, u4, u5] = sample_four_star(gen, false);
    auto [v1, v2, v3, v4] = sample_three_star(gen, false);
    auto [w1, w2, w3, w4] = sample_three_star(gen, false);
    auto [x1, x2, x3, x4, x5] = sample_four_star(gen, true);
    auto [y1, y2, y3, y4] = sample_three_star(gen, false);
    auto [z1, z2, z3, z4] = sample_three_star(gen, false);

    // f-reject if the switching is invalid
    REJECT_UNLESS(are_unique({u1, u2, u3, u4, u5, v1, v2, v3, v4, w1, w2, w3, w4,
                              x1, x2, x3, x4, x5, y1, y2, y3, y4, z1, z2, z3, z4}));
    REJECT_IF(has_edge(u1, v1) || // two-path would contain double-edge
              has_edge(w1, x1) || // two-path would contain double-edge
              has_edge(x1, y1) || // two-path would contain double-edge
              has_edge(z1, u1) || // two-path would contain double-edge
              has_edge(u1, x1) || // bad edge 1
              has_edge(v1, w1) || // bad edge 2
              has_edge(y1, z1) || // bad edge 3
              has_edge(u2, v3) ||
              has_edge(v2, w2) ||
              has_edge(w3, x2) ||
              has_edge(x3, y3) ||
              has_edge(x4, y2) ||
              has_edge(z3, u3) ||
              has_edge(z2, u4) ||
              has_edge(v4, w4) ||
              has_edge(y4, z4) ||
              has_edge(u5, x5));

    // perform the switching
    add_simple_edge(u1, v1);
    add_light_double(v1, w1);
    add_simple_edge(w1, x1);
    add_simple_edge(x1, y1);
    add_light_double(y1, z1);
    add_simple_edge(z1, u1);
    add_light_double(u1, x1);
    add_simple_edge(u2, v3);
    add_simple_edge(v2, w2);
    add_simple_edge(w3, x2);
    add_simple_edge(x3, y3);
    add_simple_edge(x4, y2);
    add_simple_edge(z3, u3);
    add_simple_edge(z2, u4);
    add_simple_edge(v4, w4);
    add_simple_edge(u5, x5);
    add_simple_edge(y4, z4);
    remove_simple_edge(u1, u2);
    remove_simple_edge(u1, u3);
    remove_simple_edge(u1, u4);
    remove_simple_edge(u1, u5);
    remove_simple_edge(v1, v2);
    remove_simple_edge(v1, v3);
    remove_simple_edge(v1, v4);
    remove_simple_edge(w1, w2);
    remove_simple_edge(w1, w3);
    remove_simple_edge(w1, w4);
    remove_simple_edge(x1, x2);
    remove_simple_edge(x1, x3);
    remove_simple_edge(x1, x4);
    remove_simple_edge(x1, x5);
    remove_simple_edge(y1, y2);
    remove_simple_edge(y1, y3);
    remove_simple_edge(y1, y4);
    remove_simple_edge(z1, z2);
    remove_simple_edge(z1, z3);
    remove_simple_edge(z1, z4);

    // b-rejection
    // compute the lower bounds
    std::size_t doubles = state_.light_doubles.size();
    integer b_VIIe_doublet_lower = b_doublet_0_lower() * b_doublet_1_lower();
    integer b_VIIe_pairs_lower = b_I_to_VII_pairs_lower(10, doubles);
    // incremental relaxation
    integer b_VIIe_GV0 = state_.simple_two_paths;
    integer b_VIIe_GV1 = calculate_bd_GV1(u1, v1, z1);
    // relax additional pairs
    std::vector<std::tuple<node, node, node, node>> additional_pairs;
    additional_pairs.emplace_back(u2, v3, u1, v1);
    additional_pairs.emplace_back(v2, w2, v1, w1);
    additional_pairs.emplace_back(w3, x2, w1, x1);
    additional_pairs.emplace_back(x3, y3, x1, y1);
    additional_pairs.emplace_back(x4, y2, x1, y1);
    additional_pairs.emplace_back(z3, u3, z1, u1);
    additional_pairs.emplace_back(z2, u4, z1, u1);
    additional_pairs.emplace_back(v4, w4, v1, w1);
    additional_pairs.emplace_back(u5, x5, u1, x1);
    additional_pairs.emplace_back(y4, z4, y1, z1);
    integer simple_pairs = M1_ - 4 * doubles;
    std::vector<node> forbidden_nodes = {v1, u1, z1, w1, x1, y1};
    integer b_VIIe_GV_pairs = 1;
    for (auto [va, vb, fa, fb] : additional_pairs) {
        integer b_VIIe_GV_pair = simple_pairs;
        b_VIIe_GV_pair -= count_simple_pairs_with_collisions(forbidden_nodes);
        b_VIIe_GV_pair -= count_simple_pairs_with_forbidden_edges(fa, fb, forbidden_nodes);
        b_VIIe_GV_pairs *= b_VIIe_GV_pair;
        forbidden_nodes.push_back(va);
        forbidden_nodes.push_back(vb);
    }
    assert(b_VIIe_doublet_lower <= b_VIIe_GV0 * b_VIIe_GV1);
    assert(b_VIIe_pairs_lower <= b_VIIe_GV_pairs);
    assert(b_VIIe_GV0 > 0);
    assert(b_VIIe_GV1 > 0);
    assert(b_VIIe_GV_pairs > 0);
    boost::random::uniform_int_distribution<integer> b_rejection_dist(0, (b_VIIe_GV0 * b_VIIe_GV1 * b_VIIe_GV_pairs) - 1);
    REJECT_IF(b_rejection_dist(gen) >= b_VIIe_doublet_lower * b_VIIe_pairs_lower);
    ACCEPT;
}

integer IncPowerlawGraphSampler::bl_m1_lower() const {
    std::size_t triples = state_.light_triples.size();
    std::size_t doubles = state_.light_doubles.size();
    std::size_t loops = state_.light_loops.size();
    integer bl_m1 = M1_; // number of pairs
    bl_m1 -= 6 * triples; // pairs in triples
    bl_m1 -= 4 * doubles; // pairs in doubles
    bl_m1 -= 2 * loops; // pairs in loops
    bl_m1 -= 2 * A2_; // forbidden edge v2v4 or v3v5
    bl_m1 -= 4 * Delta_; // node collision with v2 or v3
    bl_m1 -= 2 * degree_sequence_[h_]; // node collision with v1
    if (bl_m1 < 1) return 0;
    return bl_m1;
}

integer IncPowerlawGraphSampler::b_triplet_0_lower() const {
    integer d1_2 = Delta_ * Delta_;
    integer triples = state_.light_triples.size();
    integer doubles = state_.light_doubles.size();
    integer bt_m0 = M3_; // number of three-stars
    bt_m0 -= 18 * triples * d1_2; // three-stars in triples
    bt_m0 -= 12 * doubles * d1_2; // three-stars in doubles
    if (bt_m0 < 1) return 1;
    return bt_m0;
}

integer IncPowerlawGraphSampler::b_triplet_1_lower() const {
    count dh = degree_sequence_[h_];
    std::size_t triples = state_.light_triples.size();
    std::size_t doubles = state_.light_doubles.size();
    integer bt_m1 = L3_; // number of light three-stars
    bt_m1 -= 18 * triples * dh * dh; // light three-stars in triples
    bt_m1 -= 12 * doubles * dh * dh; // light three-stars in doubles
    bt_m1 -= B3_; // forbidden edge v1v2
    bt_m1 -= 3 * (triples + doubles) * B2_; // forbidden multi-edge v3v4, v5v6 or v7v8
    bt_m1 -= dh * dh * dh; // node collision v2 = v1
    bt_m1 -= 9 * B2_; // node collision v4, v6 or v8 in {v3, v5, v7}
    if (bt_m1 < 1) return 0;
    return bt_m1;
}

integer IncPowerlawGraphSampler::b_ta_tb_tc_lower(count pairs) const {
    count d1 = Delta_;
    std::size_t triples = state_.light_triples.size();
    std::size_t doubles = state_.light_doubles.size();
    integer b_tau_pairs_lower = 1;
    for (count p = 0; p < pairs; ++p) {
        integer b_tau_pair_lower = M1_;
        b_tau_pair_lower -= 6 * triples; // pair is in a triple
        b_tau_pair_lower -= 4 * doubles; // pair is in a double
        b_tau_pair_lower -= 2 * A2_; // forbidden edge
        b_tau_pair_lower -= 16 * d1; // vertex collision with the triplet
        b_tau_pair_lower -= 4 * p * d1; // vertex collision with the other pairs
        if (b_tau_pair_lower < 1) return 0;
        b_tau_pairs_lower *= b_tau_pair_lower;
    }
    return b_tau_pairs_lower;
}

integer IncPowerlawGraphSampler::b_doublet_0_lower() const {
    std::size_t doubles = state_.light_doubles.size();
    count d1 = Delta_;
    integer bd_m0 = M2_; // number of two-paths
    bd_m0 -= 8 * doubles * d1; // two-paths that contain a double edge
    if (bd_m0 < 1) return 1;
    return bd_m0;
}

integer IncPowerlawGraphSampler::b_doublet_1_lower() const {
    std::size_t doubles = state_.light_doubles.size();
    count dh = degree_sequence_[h_];
    integer bd_m1 = L2_; // number of light two-paths
    bd_m1 -= 8 * doubles * dh; // contains double edge
    bd_m1 -= 6 * B1_; // u2 or u3 in {v1, v2, v3}
    bd_m1 -= 2 * dh * dh; // u1 in {v2, v3}
    bd_m1 -= dh * dh; // u1 == v1
    if (bd_m1 < 1) return 0;
    return bd_m1;
}

integer IncPowerlawGraphSampler::b_I_to_VII_pairs_lower(count pairs, std::size_t doubles) const {
    count d1 = Delta_;
    integer b_tau_pairs_lower = 1;
    for (count p = 0; p < pairs; ++p) {
        integer b_tau_pair_lower = M1_; // number of pairs
        b_tau_pair_lower -= 4 * doubles; // pair is not simple
        b_tau_pair_lower -= 2 * A2_; // forbidden edge
        b_tau_pair_lower -= 12 * d1; // vertex collision with the doublet
        b_tau_pair_lower -= 4 * p * d1; // vertex collision with the other pairs
        if (b_tau_pair_lower < 1) return 0;
        b_tau_pairs_lower *= b_tau_pair_lower;
    }
    return b_tau_pairs_lower;
}

// TODO we only use this in one phase, should we just move it there?
integer IncPowerlawGraphSampler::calculate_b_ij(node i, node j, count m) const {
    // initialize with the number of valid + invalid inverse switchings
    count d_i = degree_sequence_[i];
    count d_j = degree_sequence_[j];
    count W_ij = state_.heavy_multiple_edges_at[i] + state_.heavy_loops_at[i];
    count W_ji = state_.heavy_multiple_edges_at[j] + state_.heavy_loops_at[j];
    integer b_ij = ordered_choices(d_i - W_ij, m) *
                   ordered_choices(d_j - W_ji, m);
    // now subtract the number of invalid inverse switchings
    // start by computing Y1 and Y2, the number of invalid neighbours of i and j respectively
    count Y1 = 0;
    for (auto neighbor : state_.graph.unique_neighbors(i)) {
        // non-simple pairs have already been subtracted
        if (neighbor == i || multiplicity_of(i, neighbor) > 1)
            continue;
        // a neighbor is invalid if it is heavy
        if (is_heavy(neighbor))
            Y1++;
    }
    count Y2 = 0;
    for (auto neighbor : state_.graph.unique_neighbors(j)) {
        if (neighbor == j || multiplicity_of(j, neighbor) > 1)
            continue;
        if (is_heavy(neighbor))
            Y2++;
    }
    integer invalid_switchings = 0;
    for (count l = 1; l <= m; ++l) { // l is the number of heavy vertices which make the switching invalid
        integer invalid_switchings_l = choices(m, l) *
                                       ordered_choices(Y1, l) *
                                       ordered_choices(Y2, l) *
                                       ordered_choices(d_i - W_ij - l, m - l) *
                                       ordered_choices(d_j - W_ji - l, m - l);
        if (l % 2 == 1) { // inclusion-exclusion formula
            invalid_switchings += invalid_switchings_l;
        } else {
            invalid_switchings -= invalid_switchings_l;
        }
    }
    assert(b_ij >= invalid_switchings);
    return b_ij - invalid_switchings;
}

// TODO we only use this in one phase, should we just move it there?
integer IncPowerlawGraphSampler::calculate_b_i(node i, count m) const {
    // initialise with number of valid + invalid switchings
    count d_i = degree_sequence_[i];
    count W_i = state_.heavy_multiple_edges_at[i];
    integer b_i = ordered_choices(d_i - W_i, 2 * m);
    // now subtract the number of invalid switchings
    // start by computing Y, the number of invalid neighbours of i
    count Y = 0;
    for (auto neighbor : state_.graph.unique_neighbors(i)) {
        // non-simple pairs have already been subtracted
        if (neighbor == i || multiplicity_of(i, neighbor) > 1)
            continue;
        // a neighbor is invalid if it heavy
        if (is_heavy(neighbor))
            Y++;
    }
    integer invalid_switchings = 0;
    for (count l = 1; l <= m; ++l) { // l is the number of heavy pairs which make the switching invalid
        integer invalid_switchings_l = choices(m, l) *
                                       ordered_choices(Y, 2 * l) *
                                       ordered_choices(d_i - W_i - 2 * l, 2 * m - 2 * l);
        if (l % 2 == 1) { // inclusion-exclusion formula
            invalid_switchings += invalid_switchings_l;
        } else {
            invalid_switchings -= invalid_switchings_l;
        }
    }
    assert(b_i >= invalid_switchings);
    return b_i - invalid_switchings;
}

integer IncPowerlawGraphSampler::calculate_bt_GV1(node v1, node v3, node v5, node v7) const {
    // start with the total number of light simple three-stars, then subtract all invalid choices
    integer bt_GV1 = state_.light_simple_three_stars;
    // all invalid three-stars have their centers in the two-neighborhood
    std::unordered_set<node> neighborhood = {v1, v3, v5, v7};
    // keep track of all bad choices of nodes
    std::unordered_set<node> invalid_v2_choices = neighborhood;
    std::unordered_map<node, std::unordered_set<node>> invalid_v4_choices;
    std::unordered_map<node, std::unordered_set<node>> invalid_v6_choices;
    std::unordered_map<node, std::unordered_set<node>> invalid_v8_choices;
    std::unordered_map<node, std::unordered_set<node>> invalid_v4v6_choices;
    std::unordered_map<node, std::unordered_set<node>> invalid_v4v8_choices;
    std::unordered_map<node, std::unordered_set<node>> invalid_v6v8_choices;
    std::unordered_map<node, std::unordered_set<node>> invalid_v4v6v8_choices;
    for (node v : {v1, v3, v5, v7}) {
        for (node u : state_.graph.unique_neighbors(v)) {
            if (u == v1 || u == v3 || u == v5 || u == v7)
                continue;
            neighborhood.insert(u);
            if (v == v1) {
                invalid_v2_choices.insert(u);
            }
            bool u_v_is_forbbiden_edge = multiplicity_of(u, v) > 1;
            for (node w : state_.graph.unique_neighbors(u)) {
                if (w == u || w == v1 || w == v3 || w == v5 || w == v7)
                    continue;
                if (has_multiple_edge(u, w))
                    continue;
                neighborhood.insert(w);
                if (u_v_is_forbbiden_edge) {
                    if (v == v3) {
                        invalid_v4_choices[w].insert(u);
                    } else if (v == v5) {
                        invalid_v6_choices[w].insert(u);
                    } else if (v == v7) {
                        invalid_v8_choices[w].insert(u);
                    }
                    bool is_invalid_v4_choice = invalid_v4_choices[w].find(u) != invalid_v4_choices[w].end();
                    bool is_invalid_v6_choice = invalid_v6_choices[w].find(u) != invalid_v6_choices[w].end();
                    bool is_invalid_v8_choice = invalid_v8_choices[w].find(u) != invalid_v8_choices[w].end();
                    if (is_invalid_v4_choice && is_invalid_v6_choice)
                        invalid_v4v6_choices[w].insert(u);
                    if (is_invalid_v4_choice && is_invalid_v8_choice)
                        invalid_v4v8_choices[w].insert(u);
                    if (is_invalid_v6_choice && is_invalid_v8_choice)
                        invalid_v6v8_choices[w].insert(u);
                    if (is_invalid_v4_choice && is_invalid_v6_choice && is_invalid_v8_choice)
                        invalid_v4v6v8_choices[w].insert(u);
                }
            }
        }
    }
    // go through each node and correct the number of three stars that have it as their center (v2)
    for (node neighbor : neighborhood) {
        // the second three-star must be light
        if (is_heavy(neighbor))
            continue;
        // first subtract the uncorrected number of three stars so that we can re-add the correct number
        bt_GV1 -= calculate_simple_three_stars_at(neighbor);
        // if using this node as center is not valid there is nothing to re-add so we are done
        if (invalid_v2_choices.find(neighbor) != invalid_v2_choices.end())
            continue;
        // otherwise calculate the number of valid three stars and re-add it
        count simple_edges = state_.simple_edges_at[neighbor];
        for (node v : {v1, v3, v5, v7}) {
            if (multiplicity_of(neighbor, v) == 1)
                simple_edges--;
        }
        count valid_v4v6v8_choices = simple_edges -
                                     invalid_v4_choices[neighbor].size() -
                                     invalid_v6_choices[neighbor].size() -
                                     invalid_v8_choices[neighbor].size() +
                                     invalid_v4v6_choices[neighbor].size() +
                                     invalid_v4v8_choices[neighbor].size() +
                                     invalid_v6v8_choices[neighbor].size() -
                                     invalid_v4v6v8_choices[neighbor].size();
        count valid_v4v6_choices = invalid_v8_choices[neighbor].size() -
                                   invalid_v4v8_choices[neighbor].size() -
                                   invalid_v6v8_choices[neighbor].size() +
                                   invalid_v4v6v8_choices[neighbor].size();
        count valid_v4v8_choices = invalid_v6_choices[neighbor].size() -
                                   invalid_v4v6_choices[neighbor].size() -
                                   invalid_v6v8_choices[neighbor].size() +
                                   invalid_v4v6v8_choices[neighbor].size();
        count valid_v6v8_choices = invalid_v4_choices[neighbor].size() -
                                   invalid_v4v8_choices[neighbor].size() -
                                   invalid_v4v6_choices[neighbor].size() +
                                   invalid_v4v6v8_choices[neighbor].size();
        count valid_v4_choices = invalid_v6v8_choices[neighbor].size() -
                                 invalid_v4v6v8_choices[neighbor].size();
        count valid_v6_choices = invalid_v4v8_choices[neighbor].size() -
                                 invalid_v4v6v8_choices[neighbor].size();
        count valid_v8_choices = invalid_v4v6_choices[neighbor].size() -
                                 invalid_v4v6v8_choices[neighbor].size();
        // re-add all possible combinations that make a valid three star
        // 1) use node as v4 that can only be used as v4
        // 1a) use node as v6 that can only be used as v6
        // 1ai) use node as v8 that can only be used as v8
        bt_GV1 += valid_v4_choices * valid_v6_choices * valid_v8_choices;
        // 1aii) use node as v8 that can be used in two ways
        bt_GV1 += valid_v4_choices * valid_v6_choices * valid_v4v8_choices;
        bt_GV1 += valid_v4_choices * valid_v6_choices * valid_v6v8_choices;
        // 1aiii) use node as v8 that can be used in three ways
        bt_GV1 += valid_v4_choices * valid_v6_choices * valid_v4v6v8_choices;
        // 1b) use node as v6 that can be used in two ways
        // 1bi) use node as v8 that can only be used as v8
        bt_GV1 += valid_v4_choices * valid_v4v6_choices * valid_v8_choices;
        bt_GV1 += valid_v4_choices * valid_v6v8_choices * valid_v8_choices;
        // 1bii) use node as v8 that can be used in two ways
        bt_GV1 += valid_v4_choices * valid_v4v6_choices * valid_v4v8_choices;
        bt_GV1 += valid_v4_choices * valid_v4v6_choices * valid_v6v8_choices;
        bt_GV1 += valid_v4_choices * valid_v6v8_choices * valid_v4v8_choices;
        bt_GV1 += valid_v4_choices * ordered_choices(valid_v6v8_choices, 2);
        // 1aiii) use node as v8 that can be used in three ways
        bt_GV1 += valid_v4_choices * valid_v4v6_choices * valid_v4v6v8_choices;
        bt_GV1 += valid_v4_choices * valid_v6v8_choices * valid_v4v6v8_choices;
        // 1c) use node as v6 that can be used in three ways
        // 1ci) use node as v8 that can only be used as v8
        bt_GV1 += valid_v4_choices * valid_v4v6v8_choices * valid_v8_choices;
        // 1cii) use node as v8 that can be used in two ways
        bt_GV1 += valid_v4_choices * valid_v4v6v8_choices * valid_v4v8_choices;
        bt_GV1 += valid_v4_choices * valid_v4v6v8_choices * valid_v6v8_choices;
        // 1ciii) use node as v8 that can be used in three ways
        bt_GV1 += valid_v4_choices * ordered_choices(valid_v4v6v8_choices, 2);
        // 2) use node as v4 that can be used in two ways
        // 2a) use node as v6 that can only be used as v6
        // 2ai) use node as v8 that can only be used as v8
        bt_GV1 += valid_v4v6_choices * valid_v6_choices * valid_v8_choices;
        bt_GV1 += valid_v4v8_choices * valid_v6_choices * valid_v8_choices;
        // 2aii) use node as v8 that can be used in two ways
        bt_GV1 += valid_v4v6_choices * valid_v6_choices * valid_v4v8_choices;
        bt_GV1 += valid_v4v6_choices * valid_v6_choices * valid_v6v8_choices;
        bt_GV1 += ordered_choices(valid_v4v8_choices, 2) * valid_v6_choices;
        bt_GV1 += valid_v4v8_choices * valid_v6_choices * valid_v6v8_choices;
        // 2aiii) use node as v8 that can be used in three ways
        bt_GV1 += valid_v4v6_choices * valid_v6_choices * valid_v4v6v8_choices;
        bt_GV1 += valid_v4v8_choices * valid_v6_choices * valid_v4v6v8_choices;
        // 2b) use node as v6 that can be used in two ways
        // 2bi) use node as v8 that can only be used as v8
        bt_GV1 += ordered_choices(valid_v4v6_choices, 2) * valid_v8_choices;
        bt_GV1 += valid_v4v6_choices * valid_v6v8_choices * valid_v8_choices;
        bt_GV1 += valid_v4v8_choices * valid_v4v6_choices * valid_v8_choices;
        bt_GV1 += valid_v4v8_choices * valid_v6v8_choices * valid_v8_choices;
        // 2bii) use node as v8 that can be used in two ways
        bt_GV1 += ordered_choices(valid_v4v6_choices, 2) * valid_v4v8_choices;
        bt_GV1 += ordered_choices(valid_v4v6_choices, 2) * valid_v6v8_choices;
        bt_GV1 += valid_v4v6_choices * valid_v6v8_choices * valid_v4v8_choices;
        bt_GV1 += valid_v4v6_choices * ordered_choices(valid_v6v8_choices, 2);
        bt_GV1 += ordered_choices(valid_v4v8_choices, 2) * valid_v4v6_choices;
        bt_GV1 += valid_v4v8_choices * valid_v4v6_choices * valid_v6v8_choices;
        bt_GV1 += ordered_choices(valid_v4v8_choices, 2) * valid_v6v8_choices;
        bt_GV1 += valid_v4v8_choices * ordered_choices(valid_v6v8_choices, 2);
        // 2biii) use node as v8 that can be used in three ways
        bt_GV1 += ordered_choices(valid_v4v6_choices, 2) * valid_v4v6v8_choices;
        bt_GV1 += valid_v4v6_choices * valid_v6v8_choices * valid_v4v6v8_choices;
        bt_GV1 += valid_v4v8_choices * valid_v4v6_choices * valid_v4v6v8_choices;
        bt_GV1 += valid_v4v8_choices * valid_v6v8_choices * valid_v4v6v8_choices;
        // 2c) use node as v6 that can be used in three ways
        // 2ci) use node as v8 that can only be used as v8
        bt_GV1 += valid_v4v6_choices * valid_v4v6v8_choices * valid_v8_choices;
        bt_GV1 += valid_v4v8_choices * valid_v4v6v8_choices * valid_v8_choices;
        // 2cii) use node as v8 that can be used in two ways
        bt_GV1 += valid_v4v6_choices * valid_v4v6v8_choices * valid_v4v8_choices;
        bt_GV1 += valid_v4v6_choices * valid_v4v6v8_choices * valid_v6v8_choices;
        bt_GV1 += ordered_choices(valid_v4v8_choices, 2) * valid_v4v6v8_choices;
        bt_GV1 += valid_v4v8_choices * valid_v4v6v8_choices * valid_v6v8_choices;
        // 2ciii) use node as v8 that can be used in three ways
        bt_GV1 += valid_v4v6_choices * ordered_choices(valid_v4v6v8_choices, 2);
        bt_GV1 += valid_v4v8_choices * ordered_choices(valid_v4v6v8_choices, 2);
        // 3) use node as v4 that can be used in three ways
        // 3a) use node as v6 that can only be used as v6
        // 3ai) use node as v8 that can only be used as v8
        bt_GV1 += valid_v4v6v8_choices * valid_v6_choices * valid_v8_choices;
        // 3aii) use node as v8 that can be used in two ways
        bt_GV1 += valid_v4v6v8_choices * valid_v6_choices * valid_v4v8_choices;
        bt_GV1 += valid_v4v6v8_choices * valid_v6_choices * valid_v6v8_choices;
        // 3aiii) use node as v8 that can be used in three ways
        bt_GV1 += ordered_choices(valid_v4v6v8_choices, 2) * valid_v6_choices;
        // 3b) use node as v6 that can be used in two ways
        // 3bi) use node as v8 that can only be used as v8
        bt_GV1 += valid_v4v6v8_choices * valid_v4v6_choices * valid_v8_choices;
        bt_GV1 += valid_v4v6v8_choices * valid_v6v8_choices * valid_v8_choices;
        // 3bii) use node as v8 that can be used in two ways
        bt_GV1 += valid_v4v6v8_choices * valid_v4v6_choices * valid_v4v8_choices;
        bt_GV1 += valid_v4v6v8_choices * valid_v4v6_choices * valid_v6v8_choices;
        bt_GV1 += valid_v4v6v8_choices * valid_v6v8_choices * valid_v4v8_choices;
        bt_GV1 += valid_v4v6v8_choices * ordered_choices(valid_v6v8_choices, 2);
        // 3biii) use node as v8 that can be used in three ways
        bt_GV1 += ordered_choices(valid_v4v6v8_choices, 2) * valid_v4v6_choices;
        bt_GV1 += ordered_choices(valid_v4v6v8_choices, 2) * valid_v6v8_choices;
        // 3c) use node as v6 that can be used in three ways
        // 3ci) use node as v8 that can only be used as v8
        bt_GV1 += ordered_choices(valid_v4v6v8_choices, 2) * valid_v8_choices;
        // 3cii) use node as v8 that can be used in two ways
        bt_GV1 += ordered_choices(valid_v4v6v8_choices, 2) * valid_v4v8_choices;
        bt_GV1 += ordered_choices(valid_v4v6v8_choices, 2) * valid_v6v8_choices;
        // 3ciii) use node as v8 that can be used in three ways
        bt_GV1 += ordered_choices(valid_v4v6v8_choices, 3);
    }
    return bt_GV1;
}

integer IncPowerlawGraphSampler::calculate_bd_GV1(node v1, node v3, node v5) const {
    integer bd_GV1 = state_.light_simple_two_paths;
    std::unordered_set<node> neighborhood = {v1, v3, v5};
    neighborhood.reserve(degree_sequence_[v1] + degree_sequence_[v3] + degree_sequence_[v5]);
    for (node v : {v1, v3, v5}) {
        for (node u : state_.graph.unique_neighbors(v)) {
            if (u == v1 || u == v3 || u == v5)
                continue;

            neighborhood.insert(u);
        }
    }
    // now count the number of invalid two-paths at all nodes in the two-neighborhood
    for (node neighbor : neighborhood) {
        // the two-path must be light
        if (is_heavy(neighbor))
            continue;

        unsigned delta = 0;
        delta += (multiplicity_of(v1, neighbor) == 1);
        delta += (multiplicity_of(v3, neighbor) == 1);
        delta += (multiplicity_of(v5, neighbor) == 1);

        if (!delta)
            continue;

        // add the old number of paths
        bd_GV1 -= calculate_simple_two_paths_at(neighbor);
        // otherwise we need to calculate the number of two-paths that are valid and subtract it
        count simple_edges = state_.simple_edges_at[neighbor];
        bd_GV1 += ordered_choices(simple_edges - delta, 2);
    }
    return bd_GV1;
}

integer IncPowerlawGraphSampler::count_simple_pairs_with_forbidden_edges(
        node u, node v,
        const std::vector<node>& forbidden_nodes) const {
    // we want to count how many choices we have for a pair (a, b) so that
    // 1) there is no collision with any node from forbidden_nodes
    // 2) the pair has a node a with an edge (u, a) or
    // 3) the pair has a node b with an edge (v, b)
    integer invalid_pairs = 0;
    // count pairs that have the forbidden edge (u, a) by going through the 1-neighborhood of u
    for (node a : state_.graph.unique_neighbors(u)) {
        // skip if there is a collision with forbidden nodes
        if (is_in(a, forbidden_nodes))
            continue;
        // now go through all possible choices for the second node b
        for (node b : state_.graph.unique_neighbors(a)) {
            // skip if there is a collision with forbidden nodes
            if (is_in(b, forbidden_nodes))
                continue;
            // skip if the pair is not simple
            if (a == b || has_multiple_edge(a, b))
                continue;
            invalid_pairs++;
        }
    }
    // now count all pairs that have the forbidden edge (v, b) by going through the 1-neighborhood of v
    for (node b : state_.graph.unique_neighbors(v)) {
        if (is_in(b, forbidden_nodes))
            continue;
        for (node a : state_.graph.unique_neighbors(b)) {
            if (is_in(a, forbidden_nodes))
                continue;
            if (a == b || has_multiple_edge(a, b))
                continue;
            // make sure that we count the same pair at most once
            if (has_edge(a, u))
                continue;
            invalid_pairs++;
        }
    }
    return invalid_pairs;
}

integer IncPowerlawGraphSampler::count_simple_pairs_with_collisions(const std::vector<node>& forbidden_nodes) const {
    // count the number of simple pairs that have node collisions with forbidden nodes
    integer invalid_pairs = 0;
    for (node fb : forbidden_nodes) {
        for (node v : state_.graph.unique_neighbors(fb)) {
            // we only count simple pairs
            if (v == fb || has_multiple_edge(v, fb))
                continue;
            // check if there are two collisions
            // if the pair has one collision with forbidden nodes we will only see it once,
            // but since both ways to orient the pair are invalid, we must subtract it twice
            bool two_collisions = false;
            for (node fb2 : forbidden_nodes) {
                if (v == fb2) {
                    two_collisions = true;
                    break;
                }
            }
            invalid_pairs += two_collisions ? 1 : 2;
        }
    }
    return invalid_pairs;
}

void IncPowerlawGraphSampler::add_edge(node u, node v) {
    state_.graph.add_new_edge(u, v);
}

void IncPowerlawGraphSampler::add_simple_edge(node u, node v) {
    assert(!has_edge(u, v));
    state_.graph.add_new_edge(u, v);
    // update number of simple structures
    subtract_number_of_simple_structures_at(u);
    subtract_number_of_simple_structures_at(v);
    state_.simple_edges_at[u]++;
    state_.simple_edges_at[v]++;
    add_number_of_simple_structures_at(u);
    add_number_of_simple_structures_at(v);
}

void IncPowerlawGraphSampler::add_light_double(node u, node v) {
    assert(!has_edge(u, v));
    state_.graph.add_new_edge(u, v);
    state_.graph.add_new_edge(u, v);
    state_.light_doubles.emplace_back(u, v);
}

void IncPowerlawGraphSampler::remove_edge(node u, node v) {
    state_.graph.remove_edge(u, v);
}

void IncPowerlawGraphSampler::remove_heavy_multiple_edge(node i, node j, count m) {
    state_.graph.remove_edge(i, j, true);
    state_.heavy_multiple_edges_at[i] -= m;
    state_.heavy_multiple_edges_at[j] -= m;
}

void IncPowerlawGraphSampler::remove_heavy_loop(node i, count m) {
    state_.graph.remove_edge(i, i, true);
    state_.heavy_loops_at[i] -= 2 * m;
}

void IncPowerlawGraphSampler::remove_simple_edge(node u, node v) {
    assert(multiplicity_of(u, v) == 1);
    state_.graph.remove_edge(u, v);
    // update number of simple structures
    subtract_number_of_simple_structures_at(u);
    subtract_number_of_simple_structures_at(v);
    state_.simple_edges_at[u]--;
    state_.simple_edges_at[v]--;
    add_number_of_simple_structures_at(u);
    add_number_of_simple_structures_at(v);
}

void IncPowerlawGraphSampler::remove_light_loop(node v1) {
    state_.graph.remove_edge(v1, v1);
}

void IncPowerlawGraphSampler::remove_light_triple(node v1, node v2) {
    state_.graph.remove_edge(v1, v2, true);
}

void IncPowerlawGraphSampler::remove_light_double(node v1, node v2) {
    state_.graph.remove_edge(v1, v2, true);
}

void IncPowerlawGraphSampler::add_number_of_simple_structures_at(node v) {
    uint64_t deg = static_cast<uint64_t>(state_.simple_edges_at[v]);
    if (deg < 2)
        return;

    auto update = [&] (auto two_paths, auto three_stars, auto four_stars) {
        state_.simple_two_paths += two_paths;
        state_.simple_three_stars += three_stars;
        state_.simple_four_stars += four_stars;
        if (!is_heavy(v)) {
            state_.light_simple_two_paths += two_paths;
            state_.light_simple_three_stars += three_stars;
            state_.light_simple_four_stars += four_stars;
        }
    };

    if (deg < 0xffff) {
        uint64_t two_paths = deg * (deg - 1);
        uint64_t three_stars = two_paths * (deg - 2);
        uint64_t four_stars = three_stars * (deg - 3);
        update(two_paths, three_stars, four_stars);
        assert(four_stars == calculate_simple_four_stars_at(v));
    } else {
        integer two_paths = calculate_simple_two_paths_at(v);
        integer three_stars = calculate_simple_three_stars_at(v);
        integer four_stars = calculate_simple_four_stars_at(v);
        update(two_paths, three_stars, four_stars);
    }
}

void IncPowerlawGraphSampler::subtract_number_of_simple_structures_at(node v) {
    uint64_t deg = static_cast<uint64_t>(state_.simple_edges_at[v]);
    if (deg < 2)
        return;

    auto update = [&] (auto two_paths, auto three_stars, auto four_stars) {
        state_.simple_two_paths -= two_paths;
        state_.simple_three_stars -= three_stars;
        state_.simple_four_stars -= four_stars;
        if (!is_heavy(v)) {
            state_.light_simple_two_paths -= two_paths;
            state_.light_simple_three_stars -= three_stars;
            state_.light_simple_four_stars -= four_stars;
        }
    };

    if (deg < 0xffff) {
        uint64_t two_paths = deg * (deg - 1);
        uint64_t three_stars = two_paths * (deg - 2);
        uint64_t four_stars = three_stars * (deg - 3);
        update(two_paths, three_stars, four_stars);
        assert(four_stars == calculate_simple_four_stars_at(v));
    } else {
        integer two_paths = calculate_simple_two_paths_at(v);
        integer three_stars = calculate_simple_three_stars_at(v);
        integer four_stars = calculate_simple_four_stars_at(v);
        update(two_paths, three_stars, four_stars);
    }
}

integer IncPowerlawGraphSampler::calculate_simple_two_paths_at(node v) const {
    return ordered_choices(state_.simple_edges_at[v], 2);
}

integer IncPowerlawGraphSampler::calculate_simple_three_stars_at(node v) const {
    return ordered_choices(state_.simple_edges_at[v], 3);
}

integer IncPowerlawGraphSampler::calculate_simple_four_stars_at(node v) const {
    return ordered_choices(state_.simple_edges_at[v], 4);
}

node IncPowerlawGraphSampler::sample_simple_neighbor_of(node u, std::mt19937_64& gen, std::initializer_list<node> different_from) const {
    assert(state_.simple_edges_at[u] > 0);

    auto neighbors = state_.graph.neighbors(u);
    count degree = state_.graph.degree(u);
    std::uniform_int_distribution<count> unif(0, degree - 1);

    while (true) {
        count i = unif(gen);
        node v = neighbors[i];

        if (u == v) continue; // loop
        if (i > 0 && neighbors[i - 1] == v) continue; // multi-edge
        if (i + 1 < degree && neighbors[i + 1] == v) continue; // multi-edge
        if (ranges::find(different_from, v) != different_from.end()) continue; // node collision

        return v;
    }
}

std::tuple<node, node, node> IncPowerlawGraphSampler::sample_two_path(std::mt19937_64 &gen, bool light) const {
    // sample a two-path uniform at random
    // first sample a center node weighted by its number of two-paths
    assert(light ? state_.light_simple_two_paths : state_.simple_two_paths > 0);
    boost::random::uniform_int_distribution<integer> two_path_dist(0, (light ? state_.light_simple_two_paths
                                                                             : state_.simple_two_paths) - 1);
    integer two_path_index = two_path_dist(gen);
    node u = light ? h_ : 0;
    node v, w;
    while (u < num_nodes_) {
        integer two_paths_at_u = calculate_simple_two_paths_at(u);
        if (two_path_index < two_paths_at_u) {
            break;
        } else {
            two_path_index -= two_paths_at_u;
            u++;
        }
    }
    assert(u < num_nodes_);

    v = sample_simple_neighbor_of(u, gen);
    w = sample_simple_neighbor_of(u, gen, {v});

    return std::tuple<node, node, node>(u, v, w);
}

std::tuple<node, node, node, node> IncPowerlawGraphSampler::sample_three_star(std::mt19937_64 &gen, bool light) const {
    // sample a three-star uniform at random
    // first sample a center node weighted by its number of three-stars
    assert(light ? state_.light_simple_three_stars : state_.simple_three_stars > 0);
    boost::random::uniform_int_distribution<integer> three_star_dist(0, (light ? state_.light_simple_three_stars
                                                                               : state_.simple_three_stars) - 1);
    integer three_star_index = three_star_dist(gen);
    node u = light ? h_ : 0;
    node v, w, x;
    while (u < num_nodes_) {
        integer three_stars_at_u = calculate_simple_three_stars_at(u);
        if (three_star_index < three_stars_at_u) {
            break;
        } else {
            three_star_index -= three_stars_at_u;
            u++;
        }
    }
    assert(u < num_nodes_);

    v = sample_simple_neighbor_of(u, gen);
    w = sample_simple_neighbor_of(u, gen, {v});
    x = sample_simple_neighbor_of(u, gen, {v, w});

    return std::tuple<node, node, node, node>(u, v, w, x);
}

std::tuple<node, node, node, node, node> IncPowerlawGraphSampler::sample_four_star(std::mt19937_64 &gen, bool light) const {
    // sample a four-star uniform at random
    // first sample a center node weighted by its number of four-stars
    assert(light ? state_.light_simple_four_stars : state_.simple_four_stars > 0);
    boost::random::uniform_int_distribution<integer> four_star_dist(0, (light ? state_.light_simple_four_stars
                                                                              : state_.simple_four_stars) - 1);
    integer four_star_index = four_star_dist(gen);
    node u = light ? h_ : 0;
    node v, w, x, y;
    while (u < num_nodes_) {
        integer four_stars_at_u = calculate_simple_four_stars_at(u);
        if (four_star_index < four_stars_at_u) {
            break;
        } else {
            four_star_index -= four_stars_at_u;
            u++;
        }
    }
    assert(u < num_nodes_);

    v = sample_simple_neighbor_of(u, gen);
    w = sample_simple_neighbor_of(u, gen, {v});
    x = sample_simple_neighbor_of(u, gen, {v, w});
    y = sample_simple_neighbor_of(u, gen, {v, w, x});

    return std::tuple<node, node, node, node, node>(u, v, w, x, y);
}

count IncPowerlawGraphSampler::multiplicity_of(node u, node v) const {
    return state_.graph.count_edge(u, v);
}

bool IncPowerlawGraphSampler::is_heavy(node v) const {
    return v < h_;
}

bool IncPowerlawGraphSampler::has_edge(node u, node v) const {
    return state_.graph.has_edge(u, v);
}

bool IncPowerlawGraphSampler::has_multiple_edge(node u, node v) const {
    return multiplicity_of(u, v) >= 2;
}

bool IncPowerlawGraphSampler::has_loop_at(node v) const {
    if (v >= first_degree1_node_)
        return false;
    return state_.graph.has_edge(v, v);
}

std::tuple<node, node> IncPowerlawGraphSampler::sample_edge(std::mt19937_64 &gen) const {
    auto [u, v] = state_.graph.sample(gen);
    assert(has_edge(u, v));
    return {u, v};
}

integer IncPowerlawGraphSampler::choices(std::size_t n, std::size_t k) const {
    assert(k <= n);
    integer r = ordered_choices(n, k);
    for (std::size_t i = 1; i <= k; ++i) {
        r /= i;
    }
    return r;
}

// TODO: Many invocations use constexpr k; consider adding a variant with templated k
integer IncPowerlawGraphSampler::ordered_choices(std::size_t n, std::size_t k) const {
    switch(k) {
        case 0: return 1;
        case 1: return n;
        case 2: if (n < (1llu << 32)) return n * (n - 1); else break;
        case 3: if (n < (1llu << 21)) return n * (n - 1) * (n - 2); else break;
        case 4: if (n < (1llu << 16)) return n * (n - 1) * (n - 2) * (n - 3); else break;
    }

    if (k > n)
        return 0;

    integer r = n;
    for (std::size_t i = 1; i < k; ++i) {
        r *= (n - i);
    }

    return r;
}

bool IncPowerlawGraphSampler::is_in(node v, const std::vector<node>& list) const {
    for (node x : list) {
        if (x == v)
            return true;
    }
    return false;
}

bool IncPowerlawGraphSampler::are_unique(const std::vector<node>& list, std::size_t p) const {
    if (p + 1 >= list.size())
        return true;
    for (std::size_t i = p + 1; i < list.size(); ++i) {
        if (list[p] == list[i])
            return false;
    }
    return are_unique(list, p + 1);
}

integer IncPowerlawGraphSampler::count_light_two_paths_with_matching_simple_pairs() const {
    integer two_paths = 0;
    for (node v1 = h_; v1 < first_degree1_node_; ++v1) {
        if (has_loop_at(v1))
            continue;
        for (node v2 : state_.graph.unique_neighbors(v1)) {
            if (multiplicity_of(v1, v2) > 1)
                continue;
            for (node v3 : state_.graph.unique_neighbors(v1)) {
                if (multiplicity_of(v1, v3) > 1)
                    continue;
                if (v2 == v3)
                    continue;
                for (auto [v4, v5] : state_.graph.edges()) {
                    if (v4 == v5)
                        continue;
                    if (multiplicity_of(v4, v5) > 1)
                        continue;
                    if (is_in(v4, {v1, v2, v3}))
                        continue;
                    if (is_in(v5, {v1, v2, v3}))
                        continue;
                    bool valid_v4v5_choice = !has_edge(v2, v4) && !has_edge(v3, v5);
                    bool valid_v5v4_choice = !has_edge(v2, v5) && !has_edge(v3, v4);
                    if (valid_v4v5_choice || valid_v5v4_choice) {
                        two_paths++;
                        break;
                    }
                }
            }
        }
    }
    return two_paths;
}

void IncPowerlawGraphSampler::SamplingState::reserve(const std::vector<count>& degree_sequence,
                                                     node _num_nodes) {
    num_nodes = _num_nodes;
    graph = AdjacencyVector(degree_sequence, 2);
    heavy_multiple_edges_at = std::vector<count>(num_nodes, 0);
    heavy_loops_at = std::vector<count>(num_nodes, 0);
    simple_edges_at = std::vector<count>(num_nodes, 0);
}

void IncPowerlawGraphSampler::SamplingState::reset() {
    // reset containers
    graph.clear();
    heavy_multiple_edges = std::queue<std::tuple<node, node, count>>();
    heavy_loops = std::queue<std::tuple<node, count>>();
    light_loops.clear();
    light_doubles.clear();
    light_triples.clear();
    ranges::fill(heavy_multiple_edges_at, 0);
    ranges::fill(heavy_loops_at, 0);
    ranges::fill(simple_edges_at, 0);

    // reset scalars
    light_high_multiplicity_edges = 0;
    light_high_multiplicity_loops = 0;
    simple_two_paths = 0;
    simple_three_stars = 0;
    simple_four_stars = 0;
    light_simple_two_paths = 0;
    light_simple_three_stars = 0;
    light_simple_four_stars = 0;
    stage = INITIALIZATION;
}

}
