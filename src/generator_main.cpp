#include <iostream>
#include <fstream>
#include <optional>
#include <random>
#include <thread>

#include <tlx/cmdline_parser.hpp>
#include <tlx/die.hpp>

#include <range/v3/numeric.hpp>

#include <incpwl/IncPowerlawGraphSampler.hpp>
#include <incpwl/DegreeSequenceHelper.hpp>

namespace incpwl {
static constexpr double kGamma = 2.88103;

struct Config {
// input
    enum class InputSource {
        None, Generate, File
    };

    unsigned seed{std::random_device{}()};

    InputSource input_source = InputSource::None;

    std::string input_path;
    unsigned num_nodes;
    unsigned min_degree{1};
    unsigned max_degree{0};
    double gamma{kGamma};

    std::string output_path;

    static std::optional<Config> parse(int argc, char *argv[]) {
        tlx::CmdlineParser parser;

        Config config;

        // degree sequence generator
        parser.add_unsigned('n', "nodes", config.num_nodes, "Number of nodes");
        parser.add_unsigned('a', "mindegree", config.min_degree, "Minimal degree; default 1");
        parser.add_unsigned('b', "maxdegree", config.max_degree,
                            "Maximum degree; if not provided, a value based on n and gamma is computed");
        parser.add_double('g', "gamma", config.gamma, "Power law exponent");
        parser.add_string('i', "degseq", config.input_path, "Path to degree sequence (single line per node; sorted). If empty STDIN");

        parser.add_string('o', "output", config.output_path, "Path to output; if not provided use STDERR");

        parser.process(argc, argv);

        if (config.gamma < kGamma) {
            std::cout << "!!! Only gamma values of at least " << kGamma
                      << " are support; you're in the realm of undefined behaviour !!!\n"
                         "!!! If an output is produced, it's quite likely that it's correct; but it might take some time to get there !!!\n";
        }

        if (config.num_nodes || config.min_degree != 1 || config.max_degree) {
            config.input_source = InputSource::Generate;

            if (!config.num_nodes) {
                std::cout << "Invalid number of nodes (-n)" << std::endl;
                return {};
            }

            if (config.min_degree + 1 > config.num_nodes) {
                std::cout << "Invalid min-degree (-a)" << std::endl;
                return {};
            }

            if (!config.max_degree)
                config.max_degree = 1 + std::pow(config.num_nodes, 1. / (config.gamma - 1));

            if (config.min_degree > config.max_degree) {
                std::cout << "Invalid max-degree (-b)" << std::endl;
                return {};
            }
        }

        if (!config.input_path.empty()) {
            if (config.input_source != InputSource::None) {
                std::cout << "If you specify an external input source (-i), you may not set (-n, -a, -b)" << std::endl;
                return {};
            }
        }

        return {config};
    };
};

std::mt19937_64 &get_urng(const Config &config) {
    std::hash<std::thread::id> hasher;
    static thread_local std::mt19937_64 gen(config.seed + hasher(std::this_thread::get_id()));
    return gen;
}

auto get_degree_sequence(const Config &config) {
    auto degree_sequence = [&]() -> std::vector<count> {
        switch (config.input_source) {
            case Config::InputSource::Generate: {
                auto &urng = get_urng(config);
                return incpwl::generate_degree_sequence(urng, config.num_nodes, config.gamma, config.min_degree, config.max_degree);
            }

            case Config::InputSource::File: {
                if (!config.input_path.empty()) {
                    std::ifstream file{config.input_path};
                    return incpwl::read_degree_sequence(file);
                }

                std::cout << "Expect degree sequence from STDIN" << std::endl;
                return incpwl::read_degree_sequence(std::cin);
            }

            default:
                die("Unsupported input source");
        }
    }();

    std::cout << "Degree sequence on " << degree_sequence.size() << " nodes and " << (ranges::accumulate(degree_sequence, 0u) / 2)
              << " edges\n";
    die_if(degree_sequence.empty());
    die_unless(incpwl::is_degree_sequence_graphical(degree_sequence));

    return degree_sequence;
}

auto generate_graph(const Config &config, const std::vector<count> &degree_sequence) {
    IncPowerlawGraphSampler sampler(degree_sequence, config.gamma);
    return sampler.sample_vector(get_urng(config));
}

void output_graph(const Config &config, const AdjacencyVector &graph) {
    if (config.output_path.empty())
        return graph.write_metis(std::cerr);

    std::ofstream output{config.output_path};
    graph.write_metis(output);
}

}

int main(int argc, char* argv[]) {
    auto config_opt = incpwl::Config::parse(argc, argv);
    if (!config_opt)
        return -1;

    const auto &config = config_opt.value();

    auto degree_sequence = incpwl::get_degree_sequence(config);
    auto graph = incpwl::generate_graph(config, degree_sequence);
    incpwl::output_graph(config, graph);

    return 0;
}