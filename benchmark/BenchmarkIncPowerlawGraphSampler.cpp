#include <fstream>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <omp.h>

#include <tlx/die.hpp>

#include <incpwl/ScopedTimer.hpp>
#include <incpwl/DegreeSequenceHelper.hpp>
#include <incpwl/IncPowerlawGraphSampler.hpp>

namespace incpwl {
constexpr double GAMMA = 2.88103;

std::mt19937_64 & get_urng() {
    std::random_device rd;
    static thread_local std::mt19937_64 gen{rd()};
    return gen;
}

std::vector<count> read_degree_sequence_from_db(std::size_t n, std::size_t id) {
    std::string path = "benchmarks-statistics/degree-sequences/" + std::to_string(n) + "_" + std::to_string(id);
    std::ifstream file{path};
    return incpwl::read_degree_sequence(file);
}

void write_degree_sequence(const std::vector<count> &degree_sequence, std::size_t n, std::size_t id) {
    std::string path = "benchmarks-statistics/degree-sequences/" + std::to_string(n) + "_" + std::to_string(id);
    if (std::ofstream file{path}) {
        for (count degree : degree_sequence) {
            file << degree << "\n";
        }
    }
}

void benchmark(const std::vector<count> &degree_sequence, std::size_t n, double gamma, int graphs, std::vector<double> &times, int threads_per_run = -1) {
    auto &gen = get_urng();

    IncPowerlawGraphSampler sampler(degree_sequence, gamma);
    if (threads_per_run > 0) {
        omp_set_num_threads(threads_per_run);
        sampler.enable_parallel_shuffling();
    }


    edgeid m = std::accumulate(degree_sequence.begin(), degree_sequence.end(), 0) / 2;
    double avg_d = 2 * static_cast<double>(m) / n;
    count delta = degree_sequence[0];

    std::string params = "n=" + std::to_string(n) + " g=" + std::to_string(int(100 * gamma)) + " m=" + std::to_string(m) + " delta=" +
                         std::to_string(delta) + " avg_d=" + std::to_string(avg_d);
    for (int i = 0; i < graphs; ++i) {
        ScopedTimer timer(params);
        auto G = sampler.sample_vector(gen);
        times.emplace_back(timer.elapsed());
    }
}

void benchmark_cm(const std::vector<count> &degree_sequence, std::size_t n, double gamma, int graphs, std::vector<double> &times, int threads_per_run) {
    auto &gen = get_urng();


    edgeid m = std::accumulate(degree_sequence.begin(), degree_sequence.end(), 0) / 2;
    double avg_d = 2 * static_cast<double>(m) / n;
    count delta = degree_sequence[0];

    std::string params = "n=" + std::to_string(n) + " g=" + std::to_string(int(100 * gamma)) + " m=" + std::to_string(m) + " delta=" +
                         std::to_string(delta) + " avg_d=" + std::to_string(avg_d);

    ConfigurationModel cm(degree_sequence);
    AdjacencyVector advec(degree_sequence, 10);

    if (threads_per_run > 0) {
        omp_set_num_threads(threads_per_run);
    }

    for (int i = 0; i < graphs; ++i) {
        advec.clear();
        ScopedTimer timer(params);
        cm.generate(advec, gen, threads_per_run > 0);
        times.emplace_back(timer.elapsed());
    }
}

void benchmark_parallel(const std::vector<count>& degree_sequence,
                        std::size_t n, double gamma, int num_graphs, int num_threads, int num_threads_per_run) {

    if (num_threads_per_run) {
        omp_set_nested(true);
        die_unless(omp_get_max_active_levels() > 1);
    }


    std::random_device rd;
    std::vector<std::mt19937_64> gens;
    std::vector<IncPowerlawGraphSampler> samplers;
    for (int t = 0; t < num_threads; ++t) {
        gens.emplace_back(rd() + t);
        samplers.emplace_back(degree_sequence, gamma);
        if (num_threads_per_run > 1)
            samplers.back().enable_parallel_shuffling();
    }



    edgeid m = std::accumulate(degree_sequence.begin(), degree_sequence.end(), 0) / 2;
    double avg_d = 2 * static_cast<double>(m) / n;
    count delta = degree_sequence[0];
    std::string params = "n=" + std::to_string(n) +
                         " g=" + std::to_string(int(100 * gamma)) +
                         " m=" + std::to_string(m) +
                         " delta=" + std::to_string(delta) +
                         " avg_d=" + std::to_string(avg_d);


    for (int i = 0; i < num_graphs; ++i) {
        ScopedTimer timer(params);

        AdjacencyVector global_result;

        using atomic_type = std::atomic_uint_fast32_t;
        using atomic_type_value_type = std::uint_fast32_t;
        // will be incremented with each new iteration
        atomic_type global_iterations(0);
        // eventually the lowest index of all accepted solutions
        atomic_type global_upper_bound_first_accepted(std::numeric_limits<atomic_type_value_type>::max());

        #pragma omp parallel num_threads(num_threads)
        {
            die_unequal(omp_get_num_threads(), num_threads);

            const auto tid = omp_get_thread_num();
            omp_set_num_threads(num_threads_per_run);

            auto &gen = gens[tid];
            auto &sampler = samplers[tid];
            // enable concurrent sampling
            auto local_iteration = std::numeric_limits<atomic_type_value_type>::min();
            auto start_new_iteration = [&] {
                local_iteration = global_iterations.fetch_add(1, std::memory_order_acq_rel);
            };
            auto keep_going = [&] {
                auto upper_bound = global_upper_bound_first_accepted.load(std::memory_order_acquire);
                return local_iteration < upper_bound;
            };
            sampler.enable_parallel_sampling(keep_going, start_new_iteration);
            auto local_result = sampler.sample_vector(gen);

            #pragma omp critical
            {
                // if we can improve the lower bound, do it!
                if (keep_going()) {
                    global_upper_bound_first_accepted.store(local_iteration, std::memory_order_release);
                    global_result = std::move(local_result);
                }
            }
        }
    }
}
}

int main(int argc, const char** argv) {
    using namespace incpwl;

    ScopedTimer timer("Total Runtime");
    std::string mode = argc > 1 ? argv[1] : "bench";

    auto &gen = get_urng();

    if (mode == "gen") {
        int sequences = argc > 2 ? std::stoi(argv[2]) : 5;
        int id_offset = argc > 3 ? std::stoi(argv[3]) : 1;
        std::size_t min = argc > 4 ? (1<<std::stoi(argv[4])) : (1<<4);
        std::size_t max = argc > 5 ? (1<<std::stoi(argv[5])) : (1<<16);
        std::cout << "Generating " << std::to_string(sequences) << " degree-sequences for each n" << std::endl
                  << "IDs " << std::to_string(id_offset) << " to " << std::to_string(id_offset + sequences - 1) << std::endl
                  << "n_min=" + std::to_string(min) << std::endl
                  << "n_max=" + std::to_string(max) << std::endl;
        for (std::size_t n = min; n <= max; n <<= 1) {
            std::cout << "n=" << n << ": " << std::flush;
            for (int s = 0; s < sequences; ++s) {
                write_degree_sequence(incpwl::generate_degree_sequence(gen, n, GAMMA), n, id_offset + s);
                std::cout << std::to_string(id_offset + s) << " " << std::flush;
            }
            std::cout << std::endl;
        }
    } else if (mode == "bench_seq") {
        die_unless(argc > 2);
        std::string filename{argv[2]};
        int graphs = argc > 3 ? std::stoi(argv[3]) : 5;
        double gamma = -1;
        std::vector<count> degree_sequence;
        {
            std::ifstream ifile{filename};
            std::string line;
            getline(ifile, line);
            gamma = std::stod(line);
            degree_sequence = read_degree_sequence(ifile);
        }

        die_unless(gamma > 2.8);
        die_if(degree_sequence.empty());
        die_unless(is_degree_sequence_graphical(degree_sequence));

        std::cout << "Starting experiment with parameters\n"
                  << "gamma=" << gamma << "\n"
                  << "nodes=" << degree_sequence.size() << "\n"
                  << "n_min="  << degree_sequence.back() << "\n"
                  << "n_max="  << degree_sequence.front() << "\n"
                  << "graphs=" << graphs << std::endl;

        std::vector<double> times;
        benchmark(degree_sequence, degree_sequence.size(), GAMMA, graphs, times);

        std::sort(times.begin(), times.end());
        std::cout << "Min: " << times[0]
                  << " 25%: " << (times[times.size() / 4])
                  << " 50%: " << (times[times.size() / 2])
                  << " 75%: " << (times[times.size() * 3 / 4])
                  << " Max: " << times.back() << "\n";

    } else if (mode == "bench_sequences") {
        int sequences = argc > 2 ? std::stoi(argv[2]) : 10;
        int id_offset = argc > 3 ? std::stoi(argv[3]) : 1;
        int graphs = argc > 4 ? std::stoi(argv[4]) : 2;
        std::size_t min = argc > 5 ? (1 << std::stoi(argv[5])) : (1 << 10);
        std::size_t max = argc > 6 ? (1 << std::stoi(argv[6])) : (1 << 20);
        std::cout << "Starting experiment with parameters" << std::endl
                  << "n_min=" + std::to_string(min) << std::endl
                  << "n_max=" + std::to_string(max) << std::endl
                  << "sequences=" + std::to_string(sequences) << std::endl
                  << "graphs=" + std::to_string(graphs) << std::endl;

        std::vector<double> times;
        for (std::size_t n = min; n <= max; n <<= 1) {
            for (int s = 0; s < sequences; ++s) {
                auto degree_sequence = read_degree_sequence_from_db(n, id_offset + s);
                benchmark(degree_sequence, n, GAMMA, graphs, times);
            }
        }
    } else if (mode == "bench") {
        int sequences = argc > 2 ? std::stoi(argv[2]) : 5;
        int graphs = argc > 3 ? std::stoi(argv[3]) : 1;
        std::size_t min = argc > 4 ? (1llu << std::stoi(argv[4])) : (1 << 10);
        std::size_t max = argc > 5 ? (1llu << std::stoi(argv[5])) : (1 << 20);
        count min_degree = argc > 6 ? std::stoi(argv[6]) : 1;
        double gamma = argc > 7 ? std::stod(argv[7]) : GAMMA;
        int threads_per_run = argc > 8 ? std::stoi(argv[8]) : -1;
        std::cout << "Starting experiment with parameters\n"
                  << "n_min=" << min << "\n"
                  << "n_max=" << max << "\n"
                  << "min_degree=" << min_degree << "\n"
                  << "sequences=" << sequences << "\n"
                  << "graphs=" << graphs << "\n"
                  << "threads_per_run=" << threads_per_run << std::endl;

        for (std::size_t n = min; n <= max; n <<= 1) {
            std::vector<double> times;
            times.reserve(sequences * graphs);
            for (int s = 0; s < sequences; ++s) {
                auto degree_sequence = incpwl::generate_degree_sequence(gen, n, gamma, min_degree);
                benchmark(degree_sequence, n, gamma, graphs, times, threads_per_run);
            }

            std::sort(times.begin(), times.end());
            std::cout << "Min: " << times[0]
                      << " 25%: " << (times[times.size() / 4])
                      << " 50%: " << (times[times.size() / 2])
                      << " 75%: " << (times[times.size() * 3 / 4])
                      << " Max: " << times.back() << "\n";
        }
    } else if (mode == "bench_parallel") {
        int threads = argc > 2 ? std::stoi(argv[2]) : 3;
        int sequences = argc > 3 ? std::stoi(argv[3]) : 5;
        int graphs = argc > 4 ? std::stoi(argv[4]) : 2;
        std::size_t min = argc > 5 ? (1 << std::stoi(argv[5])) : (1 << 10);
        std::size_t max = argc > 6 ? (1 << std::stoi(argv[6])) : (1 << 20);
        count min_degree = argc > 7 ? std::stoi(argv[7]) : 1;
        double gamma = argc > 8 ? std::stod(argv[8]) : GAMMA;
        int threads_per_run = argc > 9 ? std::stoi(argv[9]) : 1;
        std::cout << "Starting parallel experiment with parameters\n"
                  << "n_min=" << min << '\n'
                  << "n_max=" << max << '\n'
                  << "min_degree=" << min_degree << '\n'
                  << "threads=" << threads << '\n'
                  << "sequences=" << sequences << '\n'
                  << "graphs=" << graphs << '\n'
                  << "threads_per_run=" << threads_per_run << std::endl;
        for (std::size_t n = min; n <= max; n <<= 1) {
            for (int s = 0; s < sequences; ++s) {
                auto degree_sequence = incpwl::generate_degree_sequence(gen, n, gamma, min_degree);
                benchmark_parallel(degree_sequence, n, gamma, graphs, threads, threads_per_run);
            }
        }
    } else if (mode == "bench_cm") {
        int sequences = argc > 2 ? std::stoi(argv[2]) : 5;
        int graphs = argc > 3 ? std::stoi(argv[3]) : 1;
        std::size_t min = argc > 4 ? (1 << std::stoi(argv[4])) : (1 << 10);
        std::size_t max = argc > 5 ? (1 << std::stoi(argv[5])) : (1 << 20);
        count min_degree = argc > 6 ? std::stoi(argv[6]) : 1;
        double gamma = argc > 7 ? std::stod(argv[7]) : GAMMA;
        int threads_per_run = argc > 8 ? std::stoi(argv[8]) : 0;
        std::cout << "Starting experiment with parameters" << std::endl
                  << "n_min=" + std::to_string(min) << std::endl
                  << "n_max=" + std::to_string(max) << std::endl
                  << "min_degree=" + std::to_string(min_degree) << std::endl
                  << "sequences=" + std::to_string(sequences) << std::endl
                  << "graphs=" + std::to_string(graphs) << std::endl
                  << "threads_per_run=" << threads_per_run << std::endl;

        // uncomment the following to use a fixed seed
        for (std::size_t n = min; n <= max; n <<= 1) {
            std::vector<double> times;
            times.reserve(sequences * graphs);
            for (int s = 0; s < sequences; ++s) {
                auto degree_sequence = incpwl::generate_degree_sequence(gen, n, gamma, min_degree);
                benchmark_cm(degree_sequence, n, gamma, graphs, times, threads_per_run);
            }
        }
    }
}
