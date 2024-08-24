#include "agent.hpp"
#include "agent_io.hpp"
#include "agents/activity_agent.hpp"
#include "agents/discrete_vector_agent.hpp"
#include "agents/inertial_agent.hpp"
#include "agents/simple_agent.hpp"
#include "config_parser.hpp"
#include "model.hpp"
#include "model_factory.hpp"
#include "models/ActivityDrivenModel.hpp"
#include "models/DeGroot.hpp"
#include "models/DeffuantModel.hpp"
#include "models/InertialModel.hpp"
#include "network.hpp"
#include "network_generation.hpp"
#include "network_io.hpp"
#include "simulation.hpp"
#include "util/erfinv.hpp"
#include "util/math.hpp"
#include "util/misc.hpp"
#include "util/tomlplusplus.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

// pybind11 headers
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

// adding it here because the linker is not able to dynamically identify this template class iteration method
extern template class Seldon::ActivityDrivenModelAbstract<Seldon::ActivityAgent>;

using namespace std::string_literals;
using namespace pybind11::literals;
namespace py = pybind11;

namespace fs = std::filesystem;

void run_simulation(const std::optional<std::string> &config_file_path,
                    const std::optional<py::object> &options,
                    const std::optional<std::string> agent_file_path,
                    const std::optional<std::string> network_file_path,
                    const std::optional<std::string> output_dir_path) {
    std::string _output_dir_path = output_dir_path.value_or((fs::current_path() / fs::path("output")).string());
    fs::remove_all(_output_dir_path);
    fs::create_directories(_output_dir_path);
    Seldon::Config::SimulationOptions simulation_options;

    if (config_file_path) {
        simulation_options = Seldon::Config::parse_config_file(config_file_path.value());
    } else if (options && !options->is_none()) {
        simulation_options = py::cast<Seldon::Config::SimulationOptions>(*options);
    } else {
        throw std::runtime_error("Either config_file_path or simulation_options must be provided");
    }

    Seldon::Config::validate_settings(simulation_options);
    Seldon::Config::print_settings(simulation_options);

    if (simulation_options.model == Seldon::Config::Model::DeGroot) {
        auto simulation = Seldon::Simulation<Seldon::DeGrootModel::AgentT>(simulation_options, network_file_path, agent_file_path);
        simulation.run(_output_dir_path);
    } else if (simulation_options.model == Seldon::Config::Model::ActivityDrivenModel) {
        auto simulation = Seldon::Simulation<Seldon::ActivityDrivenModel::AgentT>(simulation_options, network_file_path, agent_file_path);

        simulation.run(_output_dir_path);

    } else if (simulation_options.model == Seldon::Config::Model::ActivityDrivenInertial) {
        auto simulation = Seldon::Simulation<Seldon::InertialModel::AgentT>(simulation_options, network_file_path, agent_file_path);

        simulation.run(_output_dir_path);

    } else if (simulation_options.model == Seldon::Config::Model::DeffuantModel) {
        auto model_settings = std::get<Seldon::Config::DeffuantSettings>(simulation_options.model_settings);
        if (model_settings.use_binary_vector) {
            auto simulation = Seldon::Simulation<Seldon::DeffuantModelVector::AgentT>(simulation_options, network_file_path, agent_file_path);

            simulation.run(_output_dir_path);

        } else {
            auto simulation = Seldon::Simulation<Seldon::DeffuantModel::AgentT>(simulation_options, network_file_path, agent_file_path);

            simulation.run(_output_dir_path);
        }
    } else {
        throw std::runtime_error("Model has not been created");
    }
}

template <typename AgentT>
void bind_Network(py::module &m, const std::string &name) {
    std::string Network_name = name + "Network";
    py::class_<Seldon::Network<AgentT>>(m, Network_name.c_str())
        .def(py::init<>())
        .def(py::init<const std::size_t>())
        .def(py::init<const std::vector<AgentT> &>())
        .def(py::init<>(
                 [](std::vector<std::vector<size_t>> &&neighbour_list, std::vector<std::vector<double>> &&weight_list, const std::string &direction) {
                     typename Seldon::Network<AgentT>::EdgeDirection edge_direction;
                     if (direction == "Incoming") {
                         edge_direction = Seldon::Network<AgentT>::EdgeDirection::Incoming;
                     } else {
                         edge_direction = Seldon::Network<AgentT>::EdgeDirection::Outgoing;
                     }
                     return Seldon::Network<AgentT>(std::move(neighbour_list), std::move(weight_list), edge_direction);
                 }),
             "neighbour_list"_a,
             "weight_list"_a,
             "direction"_a = "Incoming")
        .def("n_agents", &Seldon::Network<AgentT>::n_agents)
        .def("n_edges", &Seldon::Network<AgentT>::n_edges, "agent_idx"_a = std::nullopt)
        .def("direction",
             [](Seldon::Network<AgentT> &self) {
                 auto edge_direction = self.direction();
                 if (edge_direction == Seldon::Network<AgentT>::EdgeDirection::Incoming) {
                     return "Incoming";
                 } else {
                     return "Outgoing";
                 }
             })
        .def("strongly_connected_components",
             &Seldon::Network<AgentT>::
                 strongly_connected_components) // https://stackoverflow.com/questions/64632424/interpreting-static-cast-static-castvoid-petint-syntax
                                                // // https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods
        .def(
            "get_neighbours",
            [](Seldon::Network<AgentT> &self, std::size_t index) {
                auto span = self.get_neighbours(index);
                return std::vector<size_t>(span.begin(), span.end());
            },
            "index"_a)
        .def("get_weights",
             [](Seldon::Network<AgentT> &self, std::size_t index) {
                 auto span = self.get_weights(index);
                 return std::vector<double>(span.begin(), span.end());
             })
        .def(
            "set_weights",
            [](Seldon::Network<AgentT> &self, std::size_t agent_idx, const std::vector<double> &weights) {
                self.set_weights(agent_idx, std::span<const double>(weights));
            },
            "agent_idx"_a,
            "weights"_a)
        .def(
            "set_neighbours_and_weights",
            [](Seldon::Network<AgentT> &self,
               std::size_t agent_idx,
               const std::vector<std::size_t> &buffer_neighbours,
               const std::vector<double> &buffer_weights) {
                self.set_neighbours_and_weights(agent_idx, std::span<const std::size_t>(buffer_neighbours), std::span<const double>(buffer_weights));
            },
            "agent_idx"_a,
            "buffer_neighbours"_a,
            "buffer_weights"_a)
        .def(
            "set_neighbours_and_weights",
            [](Seldon::Network<AgentT> &self, std::size_t agent_idx, const std::vector<std::size_t> &buffer_neighbours, const double &weight) {
                self.set_neighbours_and_weights(agent_idx, std::span<const std::size_t>(buffer_neighbours), weight);
            },
            "agent_idx"_a,
            "buffer_neighbours"_a,
            "weight"_a)
        .def("push_back_neighbour_and_weight", &Seldon::Network<AgentT>::push_back_neighbour_and_weight, "agent_idx_i"_a, "agent_idx_j"_a, "w"_a)
        .def("transpose", &Seldon::Network<AgentT>::transpose)
        .def("toggle_incoming_outgoing", &Seldon::Network<AgentT>::toggle_incoming_outgoing)
        .def("switch_direction_flag", &Seldon::Network<AgentT>::switch_direction_flag)
        .def("remove_double_counting", &Seldon::Network<AgentT>::remove_double_counting)
        .def("clear", &Seldon::Network<AgentT>::clear)
        .def_readwrite("agent", &Seldon::Network<AgentT>::agents);
}

// generate bindings for generate_n_connections
template <typename AgentT>
void generate_bindings_for_gnc(std::string name, py::module &m) {
    m.def(("generate_n_connections_" + name).c_str(),
          [](std::size_t n_agents, std::size_t n_connections, bool self_interaction, std::size_t seed) {
              std::mt19937 gen(seed);
              return Seldon::NetworkGeneration::generate_n_connections<AgentT>(n_agents, n_connections, self_interaction, gen);
          },
          "n_agents"_a,
          "n_connections"_a,
          "self_interaction"_a,
          "seed"_a = std::random_device()());
}

// generate bindings for generate_from_file
template <typename AgentT>
void generate_bindings_for_gff(std::string name, py::module &m) {
    m.def(("generate_from_file_" + name).c_str(), &Seldon::NetworkGeneration::generate_from_file<AgentT>, "file"_a);
}

template <typename AgentT>
void generate_bindings_for_gsl(std::string name, py::module &m) {
    m.def(("generate_square_lattice_" + name).c_str(),
          [](std::size_t n_edge, double weight) { return Seldon::NetworkGeneration::generate_square_lattice<AgentT>(n_edge, weight); },
          "n_edge"_a,
          "weight"_a = 0.0);
}

// generate bindings for generate_fully_connected
template <typename AgentT>
void generate_bindings_for_gfc(std::string name, py::module &m) {
    m.def(("generate_fully_connected_" + name).c_str(),
          [](std::size_t n_agents, std::optional<typename Seldon::Network<AgentT>::WeightT> weight, std::optional<size_t> seed) {
              if (seed.has_value()) {
                  std::mt19937 gen(seed.value());
                  return Seldon::NetworkGeneration::generate_fully_connected<AgentT>(n_agents, gen);
              } else if (weight.has_value()) {
                  return Seldon::NetworkGeneration::generate_fully_connected<AgentT>(n_agents, weight.value());
              } else {
                  return Seldon::NetworkGeneration::generate_fully_connected<AgentT>(n_agents, 0.0);
              }
          },
          "n_agents"_a,
          "weight"_a,
          "seed"_a);
}

PYBIND11_MODULE(seldoncore, m) {
    m.doc() = "Python bindings for Seldon Cpp Engine";

    m.def("run_simulation",
          &run_simulation,
          "config_file_path"_a = std::optional<std::string>{},
          "options"_a = std::optional<py::object>{},
          "agent_file_path"_a = std::optional<std::string>{},
          "network_file_path"_a = std::optional<std::string>{},
          "output_dir_path"_a = std::optional<std::string>{});

    py::class_<Seldon::SimpleAgentData>(m, "SimpleAgentData").def(py::init<>()).def_readwrite("opinion", &Seldon::SimpleAgentData::opinion);

    py::class_<Seldon::Agent<Seldon::SimpleAgentData>>(m, "SimpleAgent")
        .def(py::init<>())
        .def(py::init<Seldon::SimpleAgentData>())
        .def_readwrite("data", &Seldon::Agent<Seldon::SimpleAgentData>::data);

    py::class_<Seldon::DiscreteVectorAgentData>(m, "DiscreteVectorAgentData")
        .def(py::init<>())
        .def_readwrite("opinion", &Seldon::DiscreteVectorAgentData::opinion);

    py::class_<Seldon::Agent<Seldon::DiscreteVectorAgentData>>(m, "DiscreteVectorAgent")
        .def(py::init<>())
        .def(py::init<Seldon::DiscreteVectorAgentData>())
        .def_readwrite("data", &Seldon::Agent<Seldon::DiscreteVectorAgentData>::data);

    py::class_<Seldon::ActivityAgentData>(m, "ActivityAgentData")
        .def(py::init<>())
        .def_readwrite("opinion", &Seldon::ActivityAgentData::opinion)
        .def_readwrite("activity", &Seldon::ActivityAgentData::activity)
        .def_readwrite("reluctance", &Seldon::ActivityAgentData::reluctance);

    py::class_<Seldon::Agent<Seldon::ActivityAgentData>>(m, "ActivityAgent")
        .def(py::init<>())
        .def(py::init<Seldon::ActivityAgentData>())
        .def_readwrite("data", &Seldon::Agent<Seldon::ActivityAgentData>::data);

    py::class_<Seldon::InertialAgentData>(m, "InertialAgentData")
        .def(py::init<>())
        .def_readwrite("opinion", &Seldon::InertialAgentData::opinion)
        .def_readwrite("activity", &Seldon::InertialAgentData::activity)
        .def_readwrite("reluctance", &Seldon::InertialAgentData::reluctance)
        .def_readwrite("velocity", &Seldon::InertialAgentData::velocity);

    py::class_<Seldon::Agent<Seldon::InertialAgentData>>(m, "InertialAgent")
        .def(py::init<>())
        .def(py::init<Seldon::InertialAgentData>())
        .def_readwrite("data", &Seldon::Agent<Seldon::InertialAgentData>::data);

    bind_Network<double>(m, "");
    bind_Network<Seldon::SimpleAgent>(m, "SimpleAgent");
    bind_Network<Seldon::DiscreteVectorAgent>(m, "DiscreteVectorAgent");
    bind_Network<Seldon::ActivityAgent>(m, "ActivityAgent");
    bind_Network<Seldon::InertialAgent>(m, "InertialAgent");

    py::class_<Seldon::Simulation<Seldon::SimpleAgent>>(m, "SimulationSimpleAgent")
        .def(py::init<const Seldon::Config::SimulationOptions &, const std::optional<std::string> &, const std::optional<std::string> &>(),
             "options"_a,
             "cli_network_file"_a = std::nullopt,
             "cli_agent_file"_a = std::nullopt)
        .def("run", &Seldon::Simulation<Seldon::SimpleAgent>::run, "output_dir_path"_a)
        .def_readwrite("network", &Seldon::Simulation<Seldon::SimpleAgent>::network);

    py::class_<Seldon::Simulation<Seldon::DiscreteVectorAgent>>(m, "SimulationDiscreteVectorAgent")
        .def(py::init<const Seldon::Config::SimulationOptions &, const std::optional<std::string> &, const std::optional<std::string> &>(),
             "options"_a,
             "cli_network_file"_a = std::nullopt,
             "cli_agent_file"_a = std::nullopt)
        .def("run", &Seldon::Simulation<Seldon::DiscreteVectorAgent>::run, "output_dir_path"_a)
        .def_readwrite("network", &Seldon::Simulation<Seldon::DiscreteVectorAgent>::network);

    py::class_<Seldon::Simulation<Seldon::ActivityAgent>>(m, "SimulationActivityAgent")
        .def(py::init<const Seldon::Config::SimulationOptions &, const std::optional<std::string> &, const std::optional<std::string> &>(),
             "options"_a,
             "cli_network_file"_a = std::nullopt,
             "cli_agent_file"_a = std::nullopt)
        .def("run", &Seldon::Simulation<Seldon::ActivityAgent>::run, "output_dir_path"_a)
        .def_readwrite("network", &Seldon::Simulation<Seldon::ActivityAgent>::network);

    py::class_<Seldon::Simulation<Seldon::InertialAgent>>(m, "SimulationInertialAgent")
        .def(py::init<const Seldon::Config::SimulationOptions &, const std::optional<std::string> &, const std::optional<std::string> &>(),
             "options"_a,
             "cli_network_file"_a = std::nullopt,
             "cli_agent_file"_a = std::nullopt)
        .def("run", &Seldon::Simulation<Seldon::InertialAgent>::run, "output_dir_path"_a)
        .def_readwrite("network", &Seldon::Simulation<Seldon::InertialAgent>::network);

    py::class_<Seldon::Config::OutputSettings>(m, "OutputSettings")
        .def(py::init<>())
        .def_readwrite("n_output_agents", &Seldon::Config::OutputSettings::n_output_agents)
        .def_readwrite("n_output_network", &Seldon::Config::OutputSettings::n_output_network)
        .def_readwrite("print_progress", &Seldon::Config::OutputSettings::print_progress)
        .def_readwrite("output_initial", &Seldon::Config::OutputSettings::output_initial)
        .def_readwrite("start_output", &Seldon::Config::OutputSettings::start_output)
        .def_readwrite("start_numbering_from", &Seldon::Config::OutputSettings::start_numbering_from);

    py::class_<Seldon::Config::DeGrootSettings>(m, "DeGrootSettings")
        .def(py::init<>())
        .def_readwrite("max_iterations", &Seldon::Config::DeGrootSettings::max_iterations)
        .def_readwrite("convergence_tol", &Seldon::Config::DeGrootSettings::convergence_tol);

    py::class_<Seldon::Config::DeffuantSettings>(m, "DeffuantSettings")
        .def(py::init<>())
        .def_readwrite("max_iterations", &Seldon::Config::DeffuantSettings::max_iterations)
        .def_readwrite("homophily_threshold", &Seldon::Config::DeffuantSettings::homophily_threshold)
        .def_readwrite("mu", &Seldon::Config::DeffuantSettings::mu)
        .def_readwrite("use_network", &Seldon::Config::DeffuantSettings::use_network)
        .def_readwrite("use_binary_vector", &Seldon::Config::DeffuantSettings::use_binary_vector)
        .def_readwrite("dim", &Seldon::Config::DeffuantSettings::dim);

    py::class_<Seldon::Config::ActivityDrivenSettings>(m, "ActivityDrivenSettings")
        .def(py::init<>())
        .def_readwrite("max_iterations", &Seldon::Config::ActivityDrivenSettings::max_iterations)
        .def_readwrite("dt", &Seldon::Config::ActivityDrivenSettings::dt)
        .def_readwrite("m", &Seldon::Config::ActivityDrivenSettings::m)
        .def_readwrite("eps", &Seldon::Config::ActivityDrivenSettings::eps)
        .def_readwrite("gamma", &Seldon::Config::ActivityDrivenSettings::gamma)
        .def_readwrite("alpha", &Seldon::Config::ActivityDrivenSettings::alpha)
        .def_readwrite("homophily", &Seldon::Config::ActivityDrivenSettings::homophily)
        .def_readwrite("reciprocity", &Seldon::Config::ActivityDrivenSettings::reciprocity)
        .def_readwrite("K", &Seldon::Config::ActivityDrivenSettings::K)
        .def_readwrite("mean_activities", &Seldon::Config::ActivityDrivenSettings::mean_activities)
        .def_readwrite("mean_weights", &Seldon::Config::ActivityDrivenSettings::mean_weights)
        .def_readwrite("n_bots", &Seldon::Config::ActivityDrivenSettings::n_bots)
        .def_readwrite("bot_m", &Seldon::Config::ActivityDrivenSettings::bot_m)
        .def_readwrite("bot_activity", &Seldon::Config::ActivityDrivenSettings::bot_activity)
        .def_readwrite("bot_opinion", &Seldon::Config::ActivityDrivenSettings::bot_opinion)
        .def_readwrite("bot_homophily", &Seldon::Config::ActivityDrivenSettings::bot_homophily)
        .def_readwrite("use_reluctances", &Seldon::Config::ActivityDrivenSettings::use_reluctances)
        .def_readwrite("reluctance_mean", &Seldon::Config::ActivityDrivenSettings::reluctance_mean)
        .def_readwrite("reluctance_sigma", &Seldon::Config::ActivityDrivenSettings::reluctance_sigma)
        .def_readwrite("reluctance_eps", &Seldon::Config::ActivityDrivenSettings::reluctance_eps)
        .def_readwrite("covariance_factor", &Seldon::Config::ActivityDrivenSettings::covariance_factor);

    py::class_<Seldon::Config::ActivityDrivenInertialSettings>(m, "ActivityDrivenInertialSettings")
        .def(py::init<>())
        .def_readwrite("max_iterations", &Seldon::Config::ActivityDrivenInertialSettings::max_iterations)
        .def_readwrite("dt", &Seldon::Config::ActivityDrivenInertialSettings::dt)
        .def_readwrite("m", &Seldon::Config::ActivityDrivenInertialSettings::m)
        .def_readwrite("eps", &Seldon::Config::ActivityDrivenInertialSettings::eps)
        .def_readwrite("gamma", &Seldon::Config::ActivityDrivenInertialSettings::gamma)
        .def_readwrite("alpha", &Seldon::Config::ActivityDrivenInertialSettings::alpha)
        .def_readwrite("homophily", &Seldon::Config::ActivityDrivenInertialSettings::homophily)
        .def_readwrite("reciprocity", &Seldon::Config::ActivityDrivenInertialSettings::reciprocity)
        .def_readwrite("K", &Seldon::Config::ActivityDrivenInertialSettings::K)
        .def_readwrite("mean_activities", &Seldon::Config::ActivityDrivenInertialSettings::mean_activities)
        .def_readwrite("mean_weights", &Seldon::Config::ActivityDrivenInertialSettings::mean_weights)
        .def_readwrite("n_bots", &Seldon::Config::ActivityDrivenInertialSettings::n_bots)
        .def_readwrite("bot_m", &Seldon::Config::ActivityDrivenInertialSettings::bot_m)
        .def_readwrite("bot_activity", &Seldon::Config::ActivityDrivenInertialSettings::bot_activity)
        .def_readwrite("bot_opinion", &Seldon::Config::ActivityDrivenInertialSettings::bot_opinion)
        .def_readwrite("bot_homophily", &Seldon::Config::ActivityDrivenInertialSettings::bot_homophily)
        .def_readwrite("use_reluctances", &Seldon::Config::ActivityDrivenInertialSettings::use_reluctances)
        .def_readwrite("reluctance_mean", &Seldon::Config::ActivityDrivenInertialSettings::reluctance_mean)
        .def_readwrite("reluctance_sigma", &Seldon::Config::ActivityDrivenInertialSettings::reluctance_sigma)
        .def_readwrite("reluctance_eps", &Seldon::Config::ActivityDrivenInertialSettings::reluctance_eps)
        .def_readwrite("covariance_factor", &Seldon::Config::ActivityDrivenInertialSettings::covariance_factor)
        .def_readwrite("friction_coefficient", &Seldon::Config::ActivityDrivenInertialSettings::friction_coefficient);

    py::class_<Seldon::Config::InitialNetworkSettings>(m, "InitialNetworkSettings")
        .def(py::init<>())
        .def_readwrite("file", &Seldon::Config::InitialNetworkSettings::file)
        .def_readwrite("number_of_agents", &Seldon::Config::InitialNetworkSettings::n_agents)
        .def_readwrite("connections_per_agent", &Seldon::Config::InitialNetworkSettings::n_connections);

    py::enum_<Seldon::Config::Model>(m, "Model")
        .value("DeGroot", Seldon::Config::Model::DeGroot)
        .value("ActivityDrivenModel", Seldon::Config::Model::ActivityDrivenModel)
        .value("DeffuantModel", Seldon::Config::Model::DeffuantModel)
        .value("ActivityDrivenInertial", Seldon::Config::Model::ActivityDrivenInertial);

    py::class_<Seldon::Config::SimulationOptions>(m, "SimulationOptions")
        .def(py::init<>())
        .def_readwrite("model_string", &Seldon::Config::SimulationOptions::model_string)
        .def_readwrite("model", &Seldon::Config::SimulationOptions::model)
        .def_readwrite("rng_seed", &Seldon::Config::SimulationOptions::rng_seed)
        .def_readwrite("output_settings", &Seldon::Config::SimulationOptions::output_settings)
        .def_readwrite("model_settings", &Seldon::Config::SimulationOptions::model_settings)
        .def_readwrite("network_settings", &Seldon::Config::SimulationOptions::network_settings);

    // gnc = generate_n_connections
    generate_bindings_for_gnc<double>("", m);
    generate_bindings_for_gnc<Seldon::SimpleAgent>("simple_agent", m);
    generate_bindings_for_gnc<Seldon::DiscreteVectorAgent>("discrete_vector_agent", m);
    generate_bindings_for_gnc<Seldon::ActivityAgent>("activity_agent", m);
    generate_bindings_for_gnc<Seldon::InertialAgent>("inertial_agent", m);

    // gfc = generate_fully_connected
    generate_bindings_for_gfc<double>("", m);
    generate_bindings_for_gfc<Seldon::SimpleAgent>("simple_agent", m);
    generate_bindings_for_gfc<Seldon::DiscreteVectorAgent>("discrete_vector_agent", m);
    generate_bindings_for_gfc<Seldon::ActivityAgent>("activity_agent", m);
    generate_bindings_for_gfc<Seldon::InertialAgent>("inertial_agent", m);

    // gff = generate_from_file
    generate_bindings_for_gff<double>("", m);
    generate_bindings_for_gff<Seldon::SimpleAgent>("simple_agent", m);
    generate_bindings_for_gff<Seldon::DiscreteVectorAgent>("discrete_vector_agent", m);
    generate_bindings_for_gff<Seldon::ActivityAgent>("activity_agent", m);
    generate_bindings_for_gff<Seldon::InertialAgent>("inertial_agent", m);

    // gsl = generate_square_lattice
    generate_bindings_for_gsl<double>("", m);
    generate_bindings_for_gsl<Seldon::SimpleAgent>("simple_agent", m);
    generate_bindings_for_gsl<Seldon::DiscreteVectorAgent>("discrete_vector_agent", m);
    generate_bindings_for_gsl<Seldon::ActivityAgent>("activity_agent", m);
    generate_bindings_for_gsl<Seldon::InertialAgent>("inertial_agent", m);

    m.def("parse_config_file", &Seldon::Config::parse_config_file, "file"_a);

    // network
    m.def("network_to_dot_file", &Seldon::network_to_dot_file<double>, "network"_a, "file_path"_a);
    m.def("network_to_dot_file_simple_agent", &Seldon::network_to_dot_file<Seldon::SimpleAgent>, "network"_a, "file_path"_a);
    m.def("network_to_dot_file_discrete_vector_agent", &Seldon::network_to_dot_file<Seldon::DiscreteVectorAgent>, "network"_a, "file_path"_a);
    m.def("network_to_dot_file_activity_agent", &Seldon::network_to_dot_file<Seldon::ActivityAgent>, "network"_a, "file_path"_a);
    m.def("network_to_dot_file_inertial_agent", &Seldon::network_to_dot_file<Seldon::InertialAgent>, "network"_a, "file_path"_a);

    m.def("network_to_file", &Seldon::network_to_file<double>, "network"_a, "file_path"_a);
    m.def("network_to_file_simple_agent", &Seldon::network_to_file<Seldon::SimpleAgent>, "network"_a, "file_path"_a);
    m.def("network_to_file_discrete_vector_agent", &Seldon::network_to_file<Seldon::DiscreteVectorAgent>, "network"_a, "file_path"_a);
    m.def("network_to_file_activity_agent", &Seldon::network_to_file<Seldon::ActivityAgent>, "network"_a, "file_path"_a);
    m.def("network_to_file_inertial_agent", &Seldon::network_to_file<Seldon::InertialAgent>, "network"_a, "file_path"_a);

    m.def("agents_from_file", &Seldon::agents_from_file<double>, "file"_a);
    m.def("agents_from_file_simple_agent", &Seldon::agents_from_file<Seldon::SimpleAgent>, "file"_a);
    m.def("agents_from_file_discrete_vector_agent", &Seldon::agents_from_file<Seldon::DiscreteVectorAgent>, "file"_a);
    m.def("agents_from_file_activity_agent", &Seldon::agents_from_file<Seldon::ActivityAgent>, "file"_a);
    m.def("agents_from_file_inertial_agent", &Seldon::agents_from_file<Seldon::InertialAgent>, "file"_a);

    m.def("agents_to_file", &Seldon::agents_to_file<double>, "network"_a, "file_path"_a);
    m.def("agents_to_file_simple_agent", &Seldon::agents_to_file<Seldon::SimpleAgent>, "network"_a, "file_path"_a);
    m.def("agents_to_file_discrete_vector_agent", &Seldon::agents_to_file<Seldon::DiscreteVectorAgent>, "network"_a, "file_path"_a);
    m.def("agents_to_file_activity_agent", &Seldon::agents_to_file<Seldon::ActivityAgent>, "network"_a, "file_path"_a);
    m.def("agents_to_file_inertial_agent", &Seldon::agents_to_file<Seldon::InertialAgent>, "network"_a, "file_path"_a);

    // Function for getting a vector of k agents (corresponding to connections)
    // drawing from n agents (without duplication)
    // ignore_idx ignores the index of the agent itself, since we will later add the agent itself ourselves to prevent duplication
    // std::optional<size_t> ignore_idx, std::size_t k, std::size_t n, std::vector<std::size_t> & buffer,std::mt19937 & gen
    m.def("draw_unique_k_from_n", &Seldon::draw_unique_k_from_n, "ignore_idx"_a, "k"_a, "n"_a, "buffer"_a, "gen"_a = std::random_device()());

    py::class_<Seldon::power_law_distribution<double>>(m, "Power_Law_Distribution")
        .def(py::init<double, double>(), "eps"_a, "gamma"_a)
        .def("__call__", &Seldon::power_law_distribution<double>::template operator()<std::mt19937>, "gen"_a)
        .def("pdf", &Seldon::power_law_distribution<double>::pdf, "x"_a)
        .def("inverse_cdf", &Seldon::power_law_distribution<double>::inverse_cdf, "x"_a)
        .def("mean", &Seldon::power_law_distribution<double>::mean);

    py::class_<Seldon::truncated_normal_distribution<double>>(m, "Truncated_Normal_Distribution")
        .def(py::init<double, double, double>(), "mean"_a, "sigma"_a, "eps"_a)
        .def("__call__", &Seldon::truncated_normal_distribution<double>::template operator()<std::mt19937>, "gen"_a)
        .def("pdf", &Seldon::truncated_normal_distribution<double>::pdf, "x"_a)
        .def("inverse_cdf", &Seldon::truncated_normal_distribution<double>::inverse_cdf, "y"_a);

    py::class_<Seldon::bivariate_normal_distribution<double>>(m, "Bivariate_Normal_Distribution")
        .def(py::init<double>(), "covariance"_a)
        .def("__call__", &Seldon::bivariate_normal_distribution<double>::template operator()<std::mt19937>, "gen"_a);

    py::class_<Seldon::bivariate_gaussian_copula<double, Seldon::power_law_distribution<double>, Seldon::truncated_normal_distribution<double>>>(
        m, "Bivariate_Gaussian_Copula")
        .def(py::init<double, Seldon::power_law_distribution<double>, Seldon::truncated_normal_distribution<double>>(),
             "covariance"_a,
             "dist1"_a,
             "dist2"_a)
        .def("__call__",
             &Seldon::bivariate_gaussian_copula<double, Seldon::power_law_distribution<double>, Seldon::truncated_normal_distribution<double>>::
                 template operator()<std::mt19937>,
             "gen"_a);

    m.def(
        "hamming_distance",
        [](const std::vector<double> &v1, const std::vector<double> &v2) {
            return Seldon::hamming_distance(std::span<const double>(v1), std::span<const double>(v2));
        },
        "v1"_a,
        "v2"_a);

    // m.def("reservoir_sampling_A_ExpJ", &Seldon::reservoir_sampling_A_ExpJ<std::function<double(size_t)>>, "k"_a, "n"_a,"weight"_a,"buffer"_a,
    // "gen"_a= std::random_device()());

    //---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // Connectivity(Tarjan Implementation)
    py::class_<Seldon::TarjanConnectivityAlgo>(m, "TarjanConnectivityAlgo")
        .def(py::init<const std::vector<std::vector<size_t>> &>(), "adjacency_list_arg"_a)
        .def_readwrite("scc_list", &Seldon::TarjanConnectivityAlgo::scc_list);

    m.def("print_settings", &Seldon::Config::print_settings, "options"_a);
    m.def("validate_settings", &Seldon::Config::validate_settings, "options"_a);

    py::class_<std::mt19937>(m, "RandomGenerator").def(py::init<unsigned int>());
}
