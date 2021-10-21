#ifndef UNIFORM_PLD_SAMPLING_INCPOWERLAWGRAPHSAMPLER_HPP
#define UNIFORM_PLD_SAMPLING_INCPOWERLAWGRAPHSAMPLER_HPP

#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>
#include <initializer_list>

#include <boost/multiprecision/cpp_int.hpp>

#include <incpwl/defs.hpp>
#include <incpwl/AdjacencyVector.hpp>
#include <incpwl/ConfigurationModel.hpp>

namespace incpwl {

using integer = boost::multiprecision::checked_cpp_int;
using rational = boost::multiprecision::checked_cpp_rational;

class IncPowerlawGraphSampler {
public:
    IncPowerlawGraphSampler(const std::vector<count>& degree_sequence, double gamma);

    void enable_parallel_sampling(std::function<bool()> keep_going, std::function<void()> start_new_iteration);
    void enable_parallel_shuffling();

    AdjacencyList sample(std::mt19937_64& gen);
    AdjacencyVector sample_vector(std::mt19937_64& gen);

protected:
    // algorithm phases
    void generate_initial_pairing(std::mt19937_64& gen);
    void initialize_stage_1();
    bool meets_stage_1_preconditions();
    bool remove_heavy_multiple_edges_or_reject(std::mt19937_64& gen); // stage 1 phase 1
    bool remove_heavy_loops_or_reject(std::mt19937_64& gen); // stage 1 phase 2
    void initialize_stage_2_and_3();
    bool meets_stage_2_preconditions();
    bool remove_light_loops_or_reject(std::mt19937_64& gen); // stage 2 phase 3
    bool remove_light_triples_or_reject(std::mt19937_64& gen); // stage 2 phase 4
    bool remove_light_doubles_or_reject(std::mt19937_64& gen); // stage 3 phase 5
    // phase 4 switchings
    bool perform_type_t_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_ta_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_tb_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_tc_switching_or_reject(std::mt19937_64& gen);
    // phase 5 switchings
    bool perform_type_I_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_III_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_IIIa_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_IV_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_IVa_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_V_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_Va_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_Vb_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_Vc_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_VI_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_VIa_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_VIb_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_VII_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_VIIa_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_VIIb_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_VIIc_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_VIId_switching_or_reject(std::mt19937_64& gen);
    bool perform_type_VIIe_switching_or_reject(std::mt19937_64& gen);
    // calculation of bounds
    integer bl_m1_lower() const;
    integer b_triplet_0_lower() const;
    integer b_triplet_1_lower() const;
    integer b_ta_tb_tc_lower(count pairs) const;
    integer b_doublet_0_lower() const;
    integer b_doublet_1_lower() const;
    integer b_I_to_VII_pairs_lower(count pairs, std::size_t doubles) const;
    // calculation of rejection probabilities
    integer calculate_b_ij(node i, node j, count m) const; // number of possible inverse m-way-switchings
    integer calculate_b_i(node i, count m) const; // number of possible inverse m-way-loop-switchings
    integer calculate_bt_GV1(node v1, node v3, node v5, node v7) const;
    integer calculate_bd_GV1(node v1, node v3, node v5) const;
    integer count_simple_pairs_with_forbidden_edges(node u, node v, const std::vector<node>& forbidden_nodes) const;
    integer count_simple_pairs_with_collisions(const std::vector<node>& forbidden_nodes) const;

    // adding/removing edges
    void add_edge(node u, node v);
    void add_simple_edge(node u, node v);
    void add_light_double(node u, node v);
    void remove_edge(node u, node v);
    void remove_heavy_multiple_edge(node i, node j, count m);
    void remove_heavy_loop(node i, count m);
    void remove_simple_edge(node u, node v);
    void remove_light_loop(node v1);
    void remove_light_triple(node v1, node v2);
    void remove_light_double(node v1, node v2);

    // for updating the number of simple structures
    void add_number_of_simple_structures_at(node v);
    void subtract_number_of_simple_structures_at(node v);

    // calculation of other quantities
    integer calculate_simple_two_paths_at(node v) const;
    integer calculate_simple_three_stars_at(node v) const;
    integer calculate_simple_four_stars_at(node v) const;

    // sampling
    node sample_simple_neighbor_of(node u, std::mt19937_64& gen, std::initializer_list<node> different_from = {}) const;
    std::tuple<node, node, node> sample_two_path(std::mt19937_64& gen, bool light) const;
    std::tuple<node, node, node, node> sample_three_star(std::mt19937_64& gen, bool light) const;
    std::tuple<node, node, node, node, node> sample_four_star(std::mt19937_64& gen, bool light) const;

    // query functions
    count multiplicity_of(node u, node v) const;
    bool is_heavy(node v) const;
    bool has_edge(node u, node v) const;
    bool has_multiple_edge(node u, node v) const;
    bool has_loop_at(node v) const;

    // utility functions
    std::tuple<node, node> sample_edge(std::mt19937_64&) const;
    integer choices(std::size_t n, std::size_t k) const;
    integer ordered_choices(std::size_t n, std::size_t k) const;
    bool is_in(node v, const std::vector<node>& list) const;
    bool are_unique(const std::vector<node>& list, std::size_t p = 0) const;

    // fixed parameters
    std::vector<count> degree_sequence_;
    ConfigurationModel config_model_;
    node num_nodes_;
    node log_num_nodes_;
    count Delta_; // maximum degree
    double gamma_; // parameter of the powerlaw-sequence, we have Delta_ <= O(n^{1/(gamma - 1)})
    double delta_;
    node h_; // the first h nodes are "heavy"
    node first_degree1_node_;
    integer M1_;
    integer M2_;
    integer M3_;
    integer M4_;
    integer H1_;
    integer H2_;
    integer H3_;
    integer H4_;
    integer L2_;
    integer L3_;
    integer L4_;
    integer A2_;
    integer B1_;
    integer B2_;
    integer B3_;

    // for parallel sampling/shuffling
    std::function<bool()> keep_going_;
    std::function<void()> start_new_iteration_;
    bool parallel_shufling_enabled_;

    // flags for enhanced debugging/testing
    bool explicit_bl_GV0_calculation_enabled_;

    // functions to calculate some quantities for enhanced debugging/testing
    integer count_light_two_paths_with_matching_simple_pairs() const;

    enum Stage {
        INITIALIZATION = 0,
        STAGE_1 = 1,
        STAGE_2_OR_3 = 2
    };

    // state parameters for the current graph
    struct SamplingState {
        void reserve(const std::vector<count>& degree_sequence, node _num_nodes);
        void reset();
        node num_nodes;
        AdjacencyVector graph;
        std::queue<std::tuple<node, node, count>> heavy_multiple_edges;
        std::queue<std::tuple<node, count>> heavy_loops;
        std::vector<node> light_loops;
        std::vector<std::pair<node, node>> light_doubles;
        std::vector<std::pair<node, node>> light_triples;
        std::vector<count> heavy_multiple_edges_at;
        std::vector<count> heavy_loops_at;
        std::vector<count> simple_edges_at;
        std::size_t light_high_multiplicity_edges;
        std::size_t light_high_multiplicity_loops;
        integer simple_two_paths;
        integer simple_three_stars;
        integer simple_four_stars;
        integer light_simple_two_paths;
        integer light_simple_three_stars;
        integer light_simple_four_stars;
        Stage stage;
    };
    SamplingState state_;
};

}

#endif //UNIFORM_PLD_SAMPLING_INCPOWERLAWGRAPHSAMPLER_HPP
