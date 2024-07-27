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
    fs::path _output_dir_path = output_dir_path.value_or(fs::path("./output"));
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
void bind_Network(py::module &m, const std::string &name){
        string Network_name = name + "Network";
        py::class_<Seldon::Network<AgentT>>(m, Network_name)
        .def(py::init<>())
        .def(py::init<const std::size_t>())
        .def(py::init<const std::vector<AgentT> &>())
        .def(py::init<>(
                 [](std::vector<std::vector<size_t>> &&neighbour_list, std::vector<std::vector<AgentT>> &&weight_list, const std::string &direction) {
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
        .def("n_edges", &Seldon::Network<AgentT>::n_edges, "agent_idx"_a) //std::nullopt
        .def("direction", &Seldon::Network<AgentT>::direction)
        .def("strongly_connected_components",
             &Seldon::Network<AgentT>::
                 strongly_connected_components) // https://stackoverflow.com/questions/64632424/interpreting-static-cast-static-castvoid-petint-syntax
                                                // // https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods
        .def("get_neighbours",
             [](Seldon::Network<AgentT> &self, std::size_t index) {
                 auto span = self.get_neighbours(index);
                 return std::vector<size_t>(span.begin(), span.end());
             })
        .def("get_weights",
             [](Seldon::Network<AgentT> &self, std::size_t index) {
                 auto span = self.get_weights(index);
                 return std::vector<AgentT>(span.begin(), span.end());
             })
        .def(
            "set_weights",
            [](Seldon::Network<AgentT> &self, std::size_t agent_idx, const std::vector<AgentT> &weights) {
                self.set_weights(agent_idx, std::span<const double>(weights));
            },
            "agent_idx"_a,
            "weights"_a)
        .def(
            "set_neighbours_and_weights",
            [](Seldon::Network<AgentT> &self,
               std::size_t agent_idx,
               const std::vector<std::size_t> &buffer_neighbours,
               const std::vector<AgentT> &buffer_weights) {
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
        .def("push_back_neighbour_and_weight",
             &Seldon::Network<AgentT>::push_back_neighbour_and_weight) // takes in (size_T, size_t, double)
        .def("transpose", &Seldon::Network<AgentT>::transpose)
        .def("toggle_incoming_outgoing", &Seldon::Network<AgentT>::toggle_incoming_outgoing)
        .def("switch_direction_flag", &Seldon::Network<AgentT>::switch_direction_flag)
        .def("remove_double_counting", &Seldon::Network<AgentT>::remove_double_counting)
        .def("clear", &Seldon::Network<AgentT>::clear)
        .def_readwrite("agent", &Seldon::Network<AgentT>::agents);

}

PYBIND11_MODULE(seldoncore, m) {
    m.doc() = "Python bindings for Seldon Cpp Engine";

    //------------------------------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------------------------------

    m.def("run_simulation",
          &run_simulation,
          "config_file_path"_a = std::optional<std::string>{},
          "options"_a = std::optional<py::object>{},
          "agent_file_path"_a = std::optional<std::string>{},
          "network_file_path"_a = std::optional<std::string>{},
          "output_dir_path"_a = std::optional<std::string>{});

    //------------------------------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------------------------------

    py::class_<Seldon::SimpleAgentData>(m, "SimpleAgentData").def(py::init<>()).def_readwrite("opinion", &Seldon::SimpleAgentData::opinion);

    py::class_<Seldon::Agent<Seldon::SimpleAgentData>>(m, "SimpleAgent")
        .def(py::init<>())
        .def(py::init<Seldon::SimpleAgentData>())
        .def_readwrite("data", &Seldon::Agent<Seldon::SimpleAgentData>::data);

    //------------------------------------------------------------------------------------------------------------------------------------------

    py::class_<Seldon::DiscreteVectorAgentData>(m, "DiscreteVectorAgentData")
        .def(py::init<>())
        .def_readwrite("opinion", &Seldon::DiscreteVectorAgentData::opinion);

    py::class_<Seldon::Agent<Seldon::DiscreteVectorAgentData>>(m, "DiscreteVectorAgent")
        .def(py::init<>())
        .def(py::init<Seldon::DiscreteVectorAgentData>())
        .def_readwrite("data", &Seldon::Agent<Seldon::DiscreteVectorAgentData>::data);

    //------------------------------------------------------------------------------------------------------------------------------------------

    py::class_<Seldon::ActivityAgentData>(m, "ActivityAgentData")
        .def(py::init<>())
        .def_readwrite("opinion", &Seldon::ActivityAgentData::opinion)
        .def_readwrite("activity", &Seldon::ActivityAgentData::activity)
        .def_readwrite("reluctance", &Seldon::ActivityAgentData::reluctance);

    py::class_<Seldon::Agent<Seldon::ActivityAgentData>>(m, "ActivityAgent")
        .def(py::init<>())
        .def(py::init<Seldon::ActivityAgentData>())
        .def_readwrite("data", &Seldon::Agent<Seldon::ActivityAgentData>::data);

    //------------------------------------------------------------------------------------------------------------------------------------------

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

    //------------------------------------------------------------------------------------------------------------------------------------------

    py::class_<Seldon::Simulation<Seldon::SimpleAgent>>(m, "SimulationSimpleAgent")
        .def(py::init<>([](const Seldon::Config::SimulationOptions &options,
                           const std::optional<std::string> &agent_file,
                           const std::optional<std::string> &network_file) {
                 Seldon::Config::validate_settings(options);
                 Seldon::Config::print_settings(options);
                 return new Seldon::Simulation<Seldon::SimpleAgent>(options, network_file, agent_file);
             }),
             "options"_a,
             "agent_file"_a = std::optional<std::string>{},
             "network_file"_a = std::optional<std::string>{})
        .def("run", &Seldon::Simulation<Seldon::SimpleAgent>::run, "output_dir_path"_a = fs::path("./output"))
        .def_readwrite("network", &Seldon::Simulation<Seldon::SimpleAgent>::network);

    //---------------------------------------------------------------------------------------------------------------------------------------

    py::class_<Seldon::Simulation<Seldon::DiscreteVectorAgent>>(m, "SimulationDiscreteVector")
        .def(py::init<>([](const Seldon::Config::SimulationOptions &options,
                           const std::optional<std::string> &agent_file,
                           const std::optional<std::string> &network_file) {
                 Seldon::Config::validate_settings(options);
                 Seldon::Config::print_settings(options);
                 return new Seldon::Simulation<Seldon::DiscreteVectorAgent>(options, network_file, agent_file);
             }),
             "options"_a,
             "agent_file"_a = std::optional<std::string>{},
             "network_file"_a = std::optional<std::string>{})
        .def("run", &Seldon::Simulation<Seldon::DiscreteVectorAgent>::run, "output_dir_path"_a = fs::path("./output"))
        .def_readwrite("network", &Seldon::Simulation<Seldon::DiscreteVectorAgent>::network);

    //---------------------------------------------------------------------------------------------------------------------------------------

    py::class_<Seldon::Simulation<Seldon::ActivityAgent>>(m, "SimulationActivityAgent")
        .def(py::init<>([](const Seldon::Config::SimulationOptions &options,
                           const std::optional<std::string> &agent_file,
                           const std::optional<std::string> &network_file) {
                 Seldon::Config::validate_settings(options);
                 Seldon::Config::print_settings(options);
                 return new Seldon::Simulation<Seldon::ActivityAgent>(options, network_file, agent_file);
             }),
             "options"_a,
             "agent_file"_a = std::optional<std::string>{},
             "network_file"_a = std::optional<std::string>{})
        .def("run", &Seldon::Simulation<Seldon::ActivityAgent>::run, "output_dir_path"_a = fs::path("./output"))
        .def_readwrite("network", &Seldon::Simulation<Seldon::ActivityAgent>::network);

    //---------------------------------------------------------------------------------------------------------------------------------------

    py::class_<Seldon::Simulation<Seldon::InertialAgent>>(m, "SimulationInertialAgent")
        .def(py::init<>([](const Seldon::Config::SimulationOptions &options,
                           const std::optional<std::string> &agent_file,
                           const std::optional<std::string> &network_file) {
                 Seldon::Config::validate_settings(options);
                 Seldon::Config::print_settings(options);
                 return new Seldon::Simulation<Seldon::InertialAgent>(options, network_file, agent_file);
             }),
             "options"_a,
             "agent_file"_a = std::optional<std::string>{},
             "network_file"_a = std::optional<std::string>{})
        .def("run", &Seldon::Simulation<Seldon::InertialAgent>::run, "output_dir_path"_a = fs::path("./output"))
        .def_readwrite("network", &Seldon::Simulation<Seldon::InertialAgent>::network);

    //------------------------------------------------------------------------------------------------------------------------------------------

    py::class_<Seldon::Config::OutputSettings>(m, "OutputSettings")
        .def(py::init([](std::optional<size_t> n_output_agents,
                         std::optional<size_t> n_output_network,
                         bool print_progress,
                         bool output_initial,
                         size_t start_output,
                         size_t start_numbering_from) {
                 Seldon::Config::OutputSettings output_settings;
                 if (n_output_agents.has_value()) {
                     output_settings.n_output_agents = n_output_agents.value();
                 }
                 if (n_output_network.has_value()) {
                     output_settings.n_output_network = n_output_network.value();
                 }
                 output_settings.print_progress = print_progress;
                 output_settings.output_initial = output_initial;
                 output_settings.start_output = start_output;
                 output_settings.start_numbering_from = start_numbering_from;
                 return output_settings;
             }),
             "n_output_agents"_a = std::optional<size_t>{},
             "n_output_network"_a = std::optional<size_t>{},
             "print_progress"_a = false,
             "output_initial"_a = true,
             "start_output"_a = 1,
             "start_numbering_from"_a = 0)
        .def_readwrite("n_output_agents", &Seldon::Config::OutputSettings::n_output_agents)
        .def_readwrite("n_output_network", &Seldon::Config::OutputSettings::n_output_network)
        .def_readwrite("print_progress", &Seldon::Config::OutputSettings::print_progress)
        .def_readwrite("output_initial", &Seldon::Config::OutputSettings::output_initial)
        .def_readwrite("start_output", &Seldon::Config::OutputSettings::start_output)
        .def_readwrite("start_numbering_from", &Seldon::Config::OutputSettings::start_numbering_from);

    // degroot setting instance to be passed in the simulation options
    py::class_<Seldon::Config::DeGrootSettings>(m, "DeGrootSettings")
        .def(py::init([](std::optional<int> max_iterations, double convergence_tol) {
                 Seldon::Config::DeGrootSettings degroot_settings;
                 if (max_iterations.has_value()) {
                     degroot_settings.max_iterations = max_iterations.value();
                 }
                 degroot_settings.convergence_tol = convergence_tol;
                 return degroot_settings;
             }),
             "max_iterations"_a = std::optional<int>{},
             "convergence_tol"_a = 1e-6)
        .def_readwrite("max_iterations", &Seldon::Config::DeGrootSettings::max_iterations)
        .def_readwrite("convergence_tol", &Seldon::Config::DeGrootSettings::convergence_tol);

    // deffuant setting instance to be passed in the simulation options
    py::class_<Seldon::Config::DeffuantSettings>(m, "DeffuantSettings")
        .def(py::init(
                 [](std::optional<int> max_iterations, double homophily_threshold, double mu, bool use_network, bool use_binary_vector, size_t dim) {
                     Seldon::Config::DeffuantSettings deffuant_settings;
                     if (max_iterations.has_value()) {
                         deffuant_settings.max_iterations = max_iterations.value();
                     }
                     deffuant_settings.homophily_threshold = homophily_threshold;
                     deffuant_settings.mu = mu;
                     deffuant_settings.use_network = use_network;
                     deffuant_settings.use_binary_vector = use_binary_vector;
                     deffuant_settings.dim = dim;
                     return deffuant_settings;
                 }),
             "max_iterations"_a = std::optional<int>{},
             "homophily_threshold"_a = 0.2,
             "mu"_a = 0.5,
             "use_network"_a = false,
             "use_binary_vector"_a = false,
             "dim"_a = 1)
        .def_readwrite("max_iterations", &Seldon::Config::DeffuantSettings::max_iterations)
        .def_readwrite("homophily_threshold", &Seldon::Config::DeffuantSettings::homophily_threshold)
        .def_readwrite("mu", &Seldon::Config::DeffuantSettings::mu)
        .def_readwrite("use_network", &Seldon::Config::DeffuantSettings::use_network)
        .def_readwrite("use_binary_vector", &Seldon::Config::DeffuantSettings::use_binary_vector)
        .def_readwrite("dim", &Seldon::Config::DeffuantSettings::dim);

    // ActivityDriven setting instance to be passed in the simulation options
    py::class_<Seldon::Config::ActivityDrivenSettings>(m, "ActivityDrivenSettings")
        .def(py::init([](std::optional<int> max_iterations,
                         double dt,
                         int m,
                         double eps,
                         double gamma,
                         double alpha,
                         double homophily,
                         double reciprocity,
                         double K,
                         bool mean_activities,
                         bool mean_weights,
                         size_t n_bots,
                         std::vector<int> bot_m,
                         std::vector<double> bot_activity,
                         std::vector<double> bot_opinion,
                         std::vector<double> bot_homophily,
                         bool use_reluctances,
                         double reluctance_mean,
                         double reluctance_sigma,
                         double reluctance_eps,
                         double covariance_factor) {
                 Seldon::Config::ActivityDrivenSettings activity_driven_settings;
                 if (max_iterations.has_value()) {
                     activity_driven_settings.max_iterations = max_iterations.value();
                 }
                 activity_driven_settings.dt = dt;
                 activity_driven_settings.m = m;
                 activity_driven_settings.eps = eps;
                 activity_driven_settings.gamma = gamma;
                 activity_driven_settings.alpha = alpha;
                 activity_driven_settings.homophily = homophily;
                 activity_driven_settings.reciprocity = reciprocity;
                 activity_driven_settings.K = K;
                 activity_driven_settings.mean_activities = mean_activities;
                 activity_driven_settings.mean_weights = mean_weights;
                 activity_driven_settings.n_bots = n_bots;
                 activity_driven_settings.bot_m = bot_m;
                 activity_driven_settings.bot_activity = bot_activity;
                 activity_driven_settings.bot_opinion = bot_opinion;
                 activity_driven_settings.bot_homophily = bot_homophily;
                 activity_driven_settings.use_reluctances = use_reluctances;
                 activity_driven_settings.reluctance_mean = reluctance_mean;
                 activity_driven_settings.reluctance_sigma = reluctance_sigma;
                 activity_driven_settings.reluctance_eps = reluctance_eps;
                 activity_driven_settings.covariance_factor = covariance_factor;
                 return activity_driven_settings;
             }),
             "max_iterations"_a = std::optional<int>{},
             "dt"_a = 0.01,
             "m"_a = 10,
             "eps"_a = 0.01,
             "gamma"_a = 2.1,
             "alpha"_a = 3.0,
             "homophily"_a = 0.5,
             "reciprocity"_a = 0.5,
             "K"_a = 3.0,
             "mean_activities"_a = false,
             "mean_weights"_a = false,
             "n_bots"_a = 0,
             "bot_m"_a = std::vector<int>(0),
             "bot_activity"_a = std::vector<double>(0),
             "bot_opinion"_a = std::vector<double>(0),
             "bot_homophily"_a = std::vector<double>(0),
             "use_reluctances"_a = false,
             "reluctance_mean"_a = 1.0,
             "reluctance_sigma"_a = 0.25,
             "reluctance_eps"_a = 0.01,
             "covariance_factor"_a = 0.0)
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

    // ActivityDrivenInertial setting instance to be passed in the simulation options
    py::class_<Seldon::Config::ActivityDrivenInertialSettings>(m, "ActivityDrivenInertialSettings")
        .def(py::init([](std::optional<int> max_iterations,
                         double dt,
                         int m,
                         double eps,
                         double gamma,
                         double alpha,
                         double homophily,
                         double reciprocity,
                         double K,
                         bool mean_activities,
                         bool mean_weights,
                         size_t n_bots,
                         std::vector<int> bot_m,
                         std::vector<double> bot_activity,
                         std::vector<double> bot_opinion,
                         std::vector<double> bot_homophily,
                         bool use_reluctances,
                         double reluctance_mean,
                         double reluctance_sigma,
                         double reluctance_eps,
                         double covariance_factor,
                         double friction_coefficient) {
                 Seldon::Config::ActivityDrivenInertialSettings activity_driven_inertial_settings;
                 if (max_iterations.has_value()) {
                     activity_driven_inertial_settings.max_iterations = max_iterations.value();
                 }
                 activity_driven_inertial_settings.dt = dt;
                 activity_driven_inertial_settings.m = m;
                 activity_driven_inertial_settings.eps = eps;
                 activity_driven_inertial_settings.gamma = gamma;
                 activity_driven_inertial_settings.alpha = alpha;
                 activity_driven_inertial_settings.homophily = homophily;
                 activity_driven_inertial_settings.reciprocity = reciprocity;
                 activity_driven_inertial_settings.K = K;
                 activity_driven_inertial_settings.mean_activities = mean_activities;
                 activity_driven_inertial_settings.mean_weights = mean_weights;
                 activity_driven_inertial_settings.n_bots = n_bots;
                 activity_driven_inertial_settings.bot_m = bot_m;
                 activity_driven_inertial_settings.bot_activity = bot_activity;
                 activity_driven_inertial_settings.bot_opinion = bot_opinion;
                 activity_driven_inertial_settings.bot_homophily = bot_homophily;
                 activity_driven_inertial_settings.use_reluctances = use_reluctances;
                 activity_driven_inertial_settings.reluctance_mean = reluctance_mean;
                 activity_driven_inertial_settings.reluctance_sigma = reluctance_sigma;
                 activity_driven_inertial_settings.reluctance_eps = reluctance_eps;
                 activity_driven_inertial_settings.covariance_factor = covariance_factor;
                 activity_driven_inertial_settings.friction_coefficient = friction_coefficient;
                 return activity_driven_inertial_settings;
             }),
             "max_iterations"_a = std::optional<int>{},
             "dt"_a = 0.01,
             "m"_a = 10,
             "eps"_a = 0.01,
             "gamma"_a = 2.1,
             "alpha"_a = 3.0,
             "homophily"_a = 0.5,
             "reciprocity"_a = 0.5,
             "K"_a = 3.0,
             "mean_activities"_a = false,
             "mean_weights"_a = false,
             "n_bots"_a = 0,
             "bot_m"_a = std::vector<int>(0),
             "bot_activity"_a = std::vector<double>(0),
             "bot_opinion"_a = std::vector<double>(0),
             "bot_homophily"_a = std::vector<double>(0),
             "use_reluctances"_a = false,
             "reluctance_mean"_a = 1.0,
             "reluctance_sigma"_a = 0.25,
             "reluctance_eps"_a = 0.01,
             "covariance_factor"_a = 0.0,
             "friction_coefficient"_a = 1.0)
        .def_readwrite("friction_coefficient", &Seldon::Config::ActivityDrivenInertialSettings::friction_coefficient)
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

    // InitialNetwork setting instance to be passed in the simulation options
    py::class_<Seldon::Config::InitialNetworkSettings>(m, "InitialNetworkSettings")
        .def(py::init([](std::optional<std::string> file, size_t number_of_agents = 200, size_t connections_per_agent = 10) {
                 Seldon::Config::InitialNetworkSettings initial_network_settings;
                 if (file.has_value()) {
                     initial_network_settings.file = file;
                 }
                 initial_network_settings.n_agents = number_of_agents;
                 initial_network_settings.n_connections = connections_per_agent;
                 return initial_network_settings;
             }),
             "file"_a = std::optional<std::string>{},
             "number_of_agents"_a = 200,
             "connections_per_agent"_a = 10)
        .def_readwrite("file", &Seldon::Config::InitialNetworkSettings::file)
        .def_readwrite("number_of_agents", &Seldon::Config::InitialNetworkSettings::n_agents)
        .def_readwrite("connections_per_agent", &Seldon::Config::InitialNetworkSettings::n_connections);

    py::class_<Seldon::Config::SimulationOptions>(m, "SimulationOptions")
        .def(py::init([](std::string model_string,
                         std::size_t rng_seed,
                         Seldon::Config::OutputSettings output_settings,
                         std::variant<Seldon::Config::DeGrootSettings,
                                      Seldon::Config::ActivityDrivenSettings,
                                      Seldon::Config::ActivityDrivenInertialSettings,
                                      Seldon::Config::DeffuantSettings> model_settings,
                         Seldon::Config::InitialNetworkSettings network_settings) {
                 Seldon::Config::SimulationOptions simulation_options;
                 simulation_options.model_string = model_string;
                 simulation_options.output_settings = output_settings;
                 simulation_options.rng_seed = rng_seed;
                 simulation_options.model_settings = model_settings;
                 simulation_options.network_settings = network_settings;
                 if (simulation_options.model_string == "DeGroot") {
                     simulation_options.model = Seldon::Config::Model::DeGroot;
                 } else if (simulation_options.model_string == "ActivityDriven") {
                     simulation_options.model = Seldon::Config::Model::ActivityDrivenModel;
                 } else if (simulation_options.model_string == "Deffuant") {
                     simulation_options.model = Seldon::Config::Model::DeffuantModel;
                 } else if (simulation_options.model_string == "ActivityDrivenInertial") {
                     simulation_options.model = Seldon::Config::Model::ActivityDrivenInertial;
                 } else {
                     throw std::runtime_error("Invalid model string. Supported models are DeGroot, ActivityDriven, Deffuant, ActivityDrivenInertial");
                 }
                 return simulation_options;
             }),
             "model_string"_a = "DeGroot",
             "rng_seed"_a = std::random_device()(),
             "output_settings"_a = Seldon::Config::OutputSettings{},
             "model_settings"_a = Seldon::Config::DeGrootSettings{},
             "network_settings"_a = Seldon::Config::InitialNetworkSettings{})
        .def_readwrite("model_string", &Seldon::Config::SimulationOptions::model_string)
        .def_readwrite("rng_seed", &Seldon::Config::SimulationOptions::rng_seed)
        .def_readwrite("output_settings", &Seldon::Config::SimulationOptions::output_settings)
        .def_readwrite("model_settings", &Seldon::Config::SimulationOptions::model_settings)
        .def_readwrite("network_settings", &Seldon::Config::SimulationOptions::network_settings);

    bind_network<double>(m, "");
    bind_network<Seldon::SimpleAgent>(m, "SimpleAgent");
    bind_network<Seldon::DiscreteVectorAgent>(m, "DiscreteVectorAgent");
    bind_network<Seldon::ActivityAgent>(m, "ActivityAgent");
    bind_network<Seldon::InertialAgent>(m, "InertialAgent");

    // py::class_<Seldon::Network<double>>(m, "Network")
    //     .def(py::init<>())
    //     .def(py::init<const std::size_t>())
    //     .def(py::init<const std::vector<double> &>())
    //     .def(py::init<>(
    //              [](std::vector<std::vector<size_t>> &&neighbour_list, std::vector<std::vector<double>> &&weight_list, const std::string &direction) {
    //                  typename Seldon::Network<double>::EdgeDirection edge_direction;
    //                  if (direction == "Incoming") {
    //                      edge_direction = Seldon::Network<double>::EdgeDirection::Incoming;
    //                  } else {
    //                      edge_direction = Seldon::Network<double>::EdgeDirection::Outgoing;
    //                  }
    //                  return Seldon::Network<double>(std::move(neighbour_list), std::move(weight_list), edge_direction);
    //              }),
    //          "neighbour_list"_a,
    //          "weight_list"_a,
    //          "direction"_a = "Incoming")
    //     .def("n_agents", &Seldon::Network<double>::n_agents)
    //     .def("n_edges", &Seldon::Network<double>::n_edges, "agent_idx"_a) //std::nullopt
    //     .def("direction", &Seldon::Network<double>::direction)
    //     .def("strongly_connected_components",
    //          &Seldon::Network<double>::
    //              strongly_connected_components) // https://stackoverflow.com/questions/64632424/interpreting-static-cast-static-castvoid-petint-syntax
    //                                             // // https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods
    //     .def("get_neighbours",
    //          [](Seldon::Network<double> &self, std::size_t index) {
    //              auto span = self.get_neighbours(index);
    //              return std::vector<size_t>(span.begin(), span.end());
    //          })
    //     .def("get_weights",
    //          [](Seldon::Network<double> &self, std::size_t index) {
    //              auto span = self.get_weights(index);
    //              return std::vector<double>(span.begin(), span.end());
    //          })
    //     .def(
    //         "set_weights",
    //         [](Seldon::Network<double> &self, std::size_t agent_idx, const std::vector<double> &weights) {
    //             self.set_weights(agent_idx, std::span<const double>(weights));
    //         },
    //         "agent_idx"_a,
    //         "weights"_a)
    //     .def(
    //         "set_neighbours_and_weights",
    //         [](Seldon::Network<double> &self,
    //            std::size_t agent_idx,
    //            const std::vector<std::size_t> &buffer_neighbours,
    //            const std::vector<double> &buffer_weights) {
    //             self.set_neighbours_and_weights(agent_idx, std::span<const std::size_t>(buffer_neighbours), std::span<const double>(buffer_weights));
    //         },
    //         "agent_idx"_a,
    //         "buffer_neighbours"_a,
    //         "buffer_weights"_a)
    //     .def(
    //         "set_neighbours_and_weights",
    //         [](Seldon::Network<double> &self, std::size_t agent_idx, const std::vector<std::size_t> &buffer_neighbours, const double &weight) {
    //             self.set_neighbours_and_weights(agent_idx, std::span<const std::size_t>(buffer_neighbours), weight);
    //         },
    //         "agent_idx"_a,
    //         "buffer_neighbours"_a,
    //         "weight"_a)
    //     .def("push_back_neighbour_and_weight",
    //          &Seldon::Network<double>::push_back_neighbour_and_weight) // takes in (size_T, size_t, double)
    //     .def("transpose", &Seldon::Network<double>::transpose)
    //     .def("toggle_incoming_outgoing", &Seldon::Network<double>::toggle_incoming_outgoing)
    //     .def("switch_direction_flag", &Seldon::Network<double>::switch_direction_flag)
    //     .def("remove_double_counting", &Seldon::Network<double>::remove_double_counting)
    //     .def("clear", &Seldon::Network<double>::clear)
    //     .def_readwrite("agent", &Seldon::Network<double>::agents);

    // py::class_<Seldon::Network<Seldon::SimpleAgent>>(m, "SimpleAgentNetwork")
    //     .def(py::init<>())
    //     .def(py::init<const std::size_t>())
    //     .def(py::init<const std::vector<Seldon::SimpleAgent> &>())
    //     .def(py::init<>(
    //              [](std::vector<std::vector<size_t>> &&neighbour_list, std::vector<std::vector<double>> &&weight_list, const std::string &direction) {
    //                  typename Seldon::Network<Seldon::SimpleAgent>::EdgeDirection edge_direction;
    //                  if (direction == "Incoming") {
    //                      edge_direction = Seldon::Network<Seldon::SimpleAgent>::EdgeDirection::Incoming;
    //                  } else {
    //                      edge_direction = Seldon::Network<Seldon::SimpleAgent>::EdgeDirection::Outgoing;
    //                  }
    //                  return Seldon::Network<Seldon::SimpleAgent>(std::move(neighbour_list), std::move(weight_list), edge_direction);
    //              }),
    //          "neighbour_list"_a,
    //          "weight_list"_a,
    //          "direction"_a = "Incoming")
    //     .def("n_agents", &Seldon::Network<Seldon::SimpleAgent>::n_agents)
    //     .def("n_edges", &Seldon::Network<Seldon::SimpleAgent>::n_edges, "agent_idx"_a)  //std::nullopt
    //     .def("direction", &Seldon::Network<Seldon::SimpleAgent>::direction)
    //     .def("strongly_connected_components",
    //          &Seldon::Network<Seldon::SimpleAgent>::
    //              strongly_connected_components) // https://stackoverflow.com/questions/64632424/interpreting-static-cast-static-castvoid-petint-syntax
    //                                             // // https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods
    //     .def("get_neighbours",
    //          [](Seldon::Network<Seldon::SimpleAgent> &self, std::size_t index) {
    //              auto span = self.get_neighbours(index);
    //              return std::vector<size_t>(span.begin(), span.end());
    //          })
    //     .def("get_weights",
    //          [](Seldon::Network<Seldon::SimpleAgent> &self, std::size_t index) {
    //              auto span = self.get_weights(index);
    //              return std::vector<double>(span.begin(), span.end());
    //          })
    //     .def(
    //         "set_weights",
    //         [](Seldon::Network<Seldon::SimpleAgent> &self, std::size_t agent_idx, const std::vector<double> &weights) {
    //             self.set_weights(agent_idx, std::span<const double>(weights));
    //         },
    //         "agent_idx"_a,
    //         "weights"_a)
    //     .def(
    //         "set_neighbours_and_weights",
    //         [](Seldon::Network<Seldon::SimpleAgent> &self,
    //            std::size_t agent_idx,
    //            const std::vector<std::size_t> &buffer_neighbours,
    //            const std::vector<double> &buffer_weights) {
    //             self.set_neighbours_and_weights(agent_idx, std::span<const std::size_t>(buffer_neighbours), std::span<const double>(buffer_weights));
    //         },
    //         "agent_idx"_a,
    //         "buffer_neighbours"_a,
    //         "buffer_weights"_a)
    //     .def(
    //         "set_neighbours_and_weights",
    //         [](Seldon::Network<Seldon::SimpleAgent> &self,
    //            std::size_t agent_idx,
    //            const std::vector<std::size_t> &buffer_neighbours,
    //            const double &weight) { self.set_neighbours_and_weights(agent_idx, std::span<const std::size_t>(buffer_neighbours), weight); },
    //         "agent_idx"_a,
    //         "buffer_neighbours"_a,
    //         "weight"_a)
    //     .def("push_back_neighbour_and_weight",
    //          &Seldon::Network<Seldon::SimpleAgent>::push_back_neighbour_and_weight) // takes in (size_T, size_t, double)
    //     .def("transpose", &Seldon::Network<Seldon::SimpleAgent>::transpose)
    //     .def("toggle_incoming_outgoing", &Seldon::Network<Seldon::SimpleAgent>::toggle_incoming_outgoing)
    //     .def("switch_direction_flag", &Seldon::Network<Seldon::SimpleAgent>::switch_direction_flag)
    //     .def("remove_double_counting", &Seldon::Network<Seldon::SimpleAgent>::remove_double_counting)
    //     .def("clear", &Seldon::Network<Seldon::SimpleAgent>::clear)
    //     .def_readwrite("agent", &Seldon::Network<Seldon::SimpleAgent>::agents);

    // py::class_<Seldon::Network<Seldon::DiscreteVectorAgent>>(m, "DiscreteVectorAgentNetwork")
    //     .def(py::init<>())
    //     .def(py::init<const std::size_t>())
    //     .def(py::init<const std::vector<Seldon::DiscreteVectorAgent> &>())
    //     .def(py::init<>(
    //              [](std::vector<std::vector<size_t>> &&neighbour_list, std::vector<std::vector<double>> &&weight_list, const std::string &direction) {
    //                  typename Seldon::Network<Seldon::DiscreteVectorAgent>::EdgeDirection edge_direction;
    //                  if (direction == "Incoming") {
    //                      edge_direction = Seldon::Network<Seldon::DiscreteVectorAgent>::EdgeDirection::Incoming;
    //                  } else {
    //                      edge_direction = Seldon::Network<Seldon::DiscreteVectorAgent>::EdgeDirection::Outgoing;
    //                  }
    //                  return Seldon::Network<Seldon::DiscreteVectorAgent>(std::move(neighbour_list), std::move(weight_list), edge_direction);
    //              }),
    //          "neighbour_list"_a,
    //          "weight_list"_a,
    //          "direction"_a = "Incoming")
    //     .def("n_agents", &Seldon::Network<Seldon::DiscreteVectorAgent>::n_agents)
    //     .def("n_edges", &Seldon::Network<Seldon::DiscreteVectorAgent>::n_edges, "agent_idx"_a) //std::nullopt
    //     .def("direction", &Seldon::Network<Seldon::DiscreteVectorAgent>::direction)
    //     .def("strongly_connected_components",
    //          &Seldon::Network<Seldon::DiscreteVectorAgent>::
    //              strongly_connected_components) // https://stackoverflow.com/questions/64632424/interpreting-static-cast-static-castvoid-petint-syntax
    //                                             // // https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods
    //     .def("get_neighbours",
    //          [](Seldon::Network<Seldon::DiscreteVectorAgent> &self, std::size_t index) {
    //              auto span = self.get_neighbours(index);
    //              return std::vector<size_t>(span.begin(), span.end());
    //          })
    //     .def("get_weights",
    //          [](Seldon::Network<Seldon::DiscreteVectorAgent> &self, std::size_t index) {
    //              auto span = self.get_weights(index);
    //              return std::vector<double>(span.begin(), span.end());
    //          })
    //     .def(
    //         "set_weights",
    //         [](Seldon::Network<Seldon::DiscreteVectorAgent> &self, std::size_t agent_idx, const std::vector<double> &weights) {
    //             self.set_weights(agent_idx, std::span<const double>(weights));
    //         },
    //         "agent_idx"_a,
    //         "weights"_a)
    //     .def(
    //         "set_neighbours_and_weights",
    //         [](Seldon::Network<Seldon::DiscreteVectorAgent> &self,
    //            std::size_t agent_idx,
    //            const std::vector<std::size_t> &buffer_neighbours,
    //            const std::vector<double> &buffer_weights) {
    //             self.set_neighbours_and_weights(agent_idx, std::span<const std::size_t>(buffer_neighbours), std::span<const double>(buffer_weights));
    //         },
    //         "agent_idx"_a,
    //         "buffer_neighbours"_a,
    //         "buffer_weights"_a)
    //     .def(
    //         "set_neighbours_and_weights",
    //         [](Seldon::Network<Seldon::DiscreteVectorAgent> &self,
    //            std::size_t agent_idx,
    //            const std::vector<std::size_t> &buffer_neighbours,
    //            const double &weight) { self.set_neighbours_and_weights(agent_idx, std::span<const std::size_t>(buffer_neighbours), weight); },
    //         "agent_idx"_a,
    //         "buffer_neighbours"_a,
    //         "weight"_a)
    //     .def("push_back_neighbour_and_weight",
    //          &Seldon::Network<Seldon::DiscreteVectorAgent>::push_back_neighbour_and_weight) // takes in (size_T, size_t, double)
    //     .def("transpose", &Seldon::Network<Seldon::DiscreteVectorAgent>::transpose)
    //     .def("toggle_incoming_outgoing", &Seldon::Network<Seldon::DiscreteVectorAgent>::toggle_incoming_outgoing)
    //     .def("switch_direction_flag", &Seldon::Network<Seldon::DiscreteVectorAgent>::switch_direction_flag)
    //     .def("remove_double_counting", &Seldon::Network<Seldon::DiscreteVectorAgent>::remove_double_counting)
    //     .def("clear", &Seldon::Network<Seldon::DiscreteVectorAgent>::clear)
    //     .def_readwrite("agent", &Seldon::Network<Seldon::DiscreteVectorAgent>::agents);

    // py::class_<Seldon::Network<Seldon::ActivityAgent>>(m, "ActivityAgentNetwork")
    //     .def(py::init<>())
    //     .def(py::init<const std::size_t>())
    //     .def(py::init<const std::vector<Seldon::ActivityAgent> &>())
    //     .def(py::init<>(
    //              [](std::vector<std::vector<size_t>> &&neighbour_list, std::vector<std::vector<double>> &&weight_list, const std::string &direction) {
    //                  typename Seldon::Network<Seldon::ActivityAgent>::EdgeDirection edge_direction;
    //                  if (direction == "Incoming") {
    //                      edge_direction = Seldon::Network<Seldon::ActivityAgent>::EdgeDirection::Incoming;
    //                  } else {
    //                      edge_direction = Seldon::Network<Seldon::ActivityAgent>::EdgeDirection::Outgoing;
    //                  }
    //                  return Seldon::Network<Seldon::ActivityAgent>(std::move(neighbour_list), std::move(weight_list), edge_direction);
    //              }),
    //          "neighbour_list"_a,
    //          "weight_list"_a,
    //          "direction"_a = "Incoming")
    //     .def("n_agents", &Seldon::Network<Seldon::ActivityAgent>::n_agents)
    //     .def("n_edges", &Seldon::Network<Seldon::ActivityAgent>::n_edges, "agent_idx"_a) //std::nullopt
    //     .def("direction", &Seldon::Network<Seldon::ActivityAgent>::direction)
    //     .def("strongly_connected_components",
    //          &Seldon::Network<Seldon::ActivityAgent>::
    //              strongly_connected_components) // https://stackoverflow.com/questions/64632424/interpreting-static-cast-static-castvoid-petint-syntax
    //                                             // // https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods
    //     .def("get_neighbours",
    //          [](Seldon::Network<Seldon::ActivityAgent> &self, std::size_t index) {
    //              auto span = self.get_neighbours(index);
    //              return std::vector<size_t>(span.begin(), span.end());
    //          })
    //     .def("get_weights",
    //          [](Seldon::Network<Seldon::ActivityAgent> &self, std::size_t index) {
    //              auto span = self.get_weights(index);
    //              return std::vector<double>(span.begin(), span.end());
    //          })
    //     .def(
    //         "set_weights",
    //         [](Seldon::Network<Seldon::ActivityAgent> &self, std::size_t agent_idx, const std::vector<double> &weights) {
    //             self.set_weights(agent_idx, std::span<const double>(weights));
    //         },
    //         "agent_idx"_a,
    //         "weights"_a)
    //     .def(
    //         "set_neighbours_and_weights",
    //         [](Seldon::Network<Seldon::ActivityAgent> &self,
    //            std::size_t agent_idx,
    //            const std::vector<std::size_t> &buffer_neighbours,
    //            const std::vector<double> &buffer_weights) {
    //             self.set_neighbours_and_weights(agent_idx, std::span<const std::size_t>(buffer_neighbours), std::span<const double>(buffer_weights));
    //         },
    //         "agent_idx"_a,
    //         "buffer_neighbours"_a,
    //         "buffer_weights"_a)
    //     .def(
    //         "set_neighbours_and_weights",
    //         [](Seldon::Network<Seldon::ActivityAgent> &self,
    //            std::size_t agent_idx,
    //            const std::vector<std::size_t> &buffer_neighbours,
    //            const double &weight) { self.set_neighbours_and_weights(agent_idx, std::span<const std::size_t>(buffer_neighbours), weight); },
    //         "agent_idx"_a,
    //         "buffer_neighbours"_a,
    //         "weight"_a)
    //     .def("push_back_neighbour_and_weight",
    //          &Seldon::Network<Seldon::ActivityAgent>::push_back_neighbour_and_weight) // takes in (size_T, size_t, double)
    //     .def("transpose", &Seldon::Network<Seldon::ActivityAgent>::transpose)
    //     .def("toggle_incoming_outgoing", &Seldon::Network<Seldon::ActivityAgent>::toggle_incoming_outgoing)
    //     .def("switch_direction_flag", &Seldon::Network<Seldon::ActivityAgent>::switch_direction_flag)
    //     .def("remove_double_counting", &Seldon::Network<Seldon::ActivityAgent>::remove_double_counting)
    //     .def("clear", &Seldon::Network<Seldon::ActivityAgent>::clear)
    //     .def_readwrite("agent", &Seldon::Network<Seldon::ActivityAgent>::agents);

    // py::class_<Seldon::Network<Seldon::InertialAgent>>(m, "InertialAgentNetwork")
    //     .def(py::init<>())
    //     .def(py::init<const std::size_t>())
    //     .def(py::init<const std::vector<Seldon::InertialAgent> &>())
    //     .def(py::init<>(
    //              [](std::vector<std::vector<size_t>> &&neighbour_list, std::vector<std::vector<double>> &&weight_list, const std::string &direction) {
    //                  typename Seldon::Network<Seldon::InertialAgent>::EdgeDirection edge_direction;
    //                  if (direction == "Incoming") {
    //                      edge_direction = Seldon::Network<Seldon::InertialAgent>::EdgeDirection::Incoming;
    //                  } else {
    //                      edge_direction = Seldon::Network<Seldon::InertialAgent>::EdgeDirection::Outgoing;
    //                  }
    //                  return Seldon::Network<Seldon::InertialAgent>(std::move(neighbour_list), std::move(weight_list), edge_direction);
    //              }),
    //          "neighbour_list"_a,
    //          "weight_list"_a,
    //          "direction"_a = "Incoming")
    //     .def("n_agents", &Seldon::Network<Seldon::InertialAgent>::n_agents)
    //     .def("n_edges", &Seldon::Network<Seldon::InertialAgent>::n_edges, "agent_idx"_a) //std::nullopt
    //     .def("direction", &Seldon::Network<Seldon::InertialAgent>::direction)
    //     .def("strongly_connected_components",
    //          &Seldon::Network<Seldon::InertialAgent>::
    //              strongly_connected_components) // https://stackoverflow.com/questions/64632424/interpreting-static-cast-static-castvoid-petint-syntax
    //                                             // // https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods
    //     .def("get_neighbours",
    //          [](Seldon::Network<Seldon::InertialAgent> &self, std::size_t index) {
    //              auto span = self.get_neighbours(index);
    //              return std::vector<size_t>(span.begin(), span.end());
    //          })
    //     .def("get_weights",
    //          [](Seldon::Network<Seldon::InertialAgent> &self, std::size_t index) {
    //              auto span = self.get_weights(index);
    //              return std::vector<double>(span.begin(), span.end());
    //          })
    //     .def(
    //         "set_weights",
    //         [](Seldon::Network<Seldon::InertialAgent> &self, std::size_t agent_idx, const std::vector<double> &weights) {
    //             self.set_weights(agent_idx, std::span<const double>(weights));
    //         },
    //         "agent_idx"_a,
    //         "weights"_a)
    //     .def(
    //         "set_neighbours_and_weights",
    //         [](Seldon::Network<Seldon::InertialAgent> &self,
    //            std::size_t agent_idx,
    //            const std::vector<std::size_t> &buffer_neighbours,
    //            const std::vector<double> &buffer_weights) {
    //             self.set_neighbours_and_weights(agent_idx, std::span<const std::size_t>(buffer_neighbours), std::span<const double>(buffer_weights));
    //         },
    //         "agent_idx"_a,
    //         "buffer_neighbours"_a,
    //         "buffer_weights"_a)
    //     .def(
    //         "set_neighbours_and_weights",
    //         [](Seldon::Network<Seldon::InertialAgent> &self,
    //            std::size_t agent_idx,
    //            const std::vector<std::size_t> &buffer_neighbours,
    //            const double &weight) { self.set_neighbours_and_weights(agent_idx, std::span<const std::size_t>(buffer_neighbours), weight); },
    //         "agent_idx"_a,
    //         "buffer_neighbours"_a,
    //         "weight"_a)
    //     .def("push_back_neighbour_and_weight",
    //          &Seldon::Network<Seldon::InertialAgent>::push_back_neighbour_and_weight) // takes in (size_T, size_t, double)
    //     .def("transpose", &Seldon::Network<Seldon::InertialAgent>::transpose)
    //     .def("toggle_incoming_outgoing", &Seldon::Network<Seldon::InertialAgent>::toggle_incoming_outgoing)
    //     .def("switch_direction_flag", &Seldon::Network<Seldon::InertialAgent>::switch_direction_flag)
    //     .def("remove_double_counting", &Seldon::Network<Seldon::InertialAgent>::remove_double_counting)
    //     .def("clear", &Seldon::Network<Seldon::InertialAgent>::clear)
    //     .def_readwrite("agent", &Seldon::Network<Seldon::InertialAgent>::agents);

    // generate_networks_bindings<double>(m, "Network");

    //--------------------------------------------------------------------------------------------------------------------

    m.def(
        "generate_n_connections",
        [](std::size_t n_agents, std::size_t n_connections, bool self_interaction, std::size_t seed) {
            std::mt19937 gen(seed);
            return Seldon::NetworkGeneration::generate_n_connections<double>(n_agents, n_connections, self_interaction, gen);
        },
        "n_agents"_a,
        "n_connections"_a,
        "self_interaction"_a = false,
        "seed"_a = 0);

    m.def(
        "generate_n_connections_degroot",
        [](std::size_t n_agents, std::size_t n_connections, bool self_interaction, std::size_t seed) {
            std::mt19937 gen(seed);
            return Seldon::NetworkGeneration::generate_n_connections<Seldon::DeGrootModel::AgentT>(n_agents, n_connections, self_interaction, gen);
        },
        "n_agents"_a,
        "n_connections"_a,
        "self_interaction"_a = false,
        "seed"_a = 0);

    m.def(
        "generate_n_connections_deffuant",
        [](std::size_t n_agents, std::size_t n_connections, bool self_interaction, std::size_t seed) {
            std::mt19937 gen(seed);
            return Seldon::NetworkGeneration::generate_n_connections<Seldon::DeffuantModel::AgentT>(n_agents, n_connections, self_interaction, gen);
        },
        "n_agents"_a,
        "n_connections"_a,
        "self_interaction"_a = false,
        "seed"_a = 0);

    m.def(
        "generate_n_connections_deffuant_vector",
        [](std::size_t n_agents, std::size_t n_connections, bool self_interaction, std::size_t seed) {
            std::mt19937 gen(seed);
            return Seldon::NetworkGeneration::generate_n_connections<Seldon::DeffuantModelVector::AgentT>(
                n_agents, n_connections, self_interaction, gen);
        },
        "n_agents"_a,
        "n_connections"_a,
        "self_interaction"_a = false,
        "seed"_a = 0);

    m.def(
        "generate_n_connections_activity_driven",
        [](std::size_t n_agents, std::size_t n_connections, bool self_interaction, std::size_t seed) {
            std::mt19937 gen(seed);
            return Seldon::NetworkGeneration::generate_n_connections<Seldon::ActivityDrivenModel::AgentT>(
                n_agents, n_connections, self_interaction, gen);
        },
        "n_agents"_a,
        "n_connections"_a,
        "self_interaction"_a = false,
        "seed"_a = 0);

    m.def(
        "generate_n_connections_activity_driven_inertial",
        [](std::size_t n_agents, std::size_t n_connections, bool self_interaction, std::size_t seed) {
            std::mt19937 gen(seed);
            return Seldon::NetworkGeneration::generate_n_connections<Seldon::InertialModel::AgentT>(n_agents, n_connections, self_interaction, gen);
        },
        "n_agents"_a,
        "n_connections"_a,
        "self_interaction"_a = false,
        "seed"_a = 0);

    m.def(
        "generate_fully_connected",
        [](size_t n_agents, std::optional<typename Seldon::Network<Seldon::DeGrootModel::AgentT>::WeightT> weight, std::optional<size_t> seed) {
            if (seed.has_value()) {
                std::mt19937 gen(seed.value());
                return Seldon::NetworkGeneration::generate_fully_connected<double>(n_agents, gen);
            } else if (weight.has_value()) {
                return Seldon::NetworkGeneration::generate_fully_connected<double>(n_agents, weight.value());
            } else {
                return Seldon::NetworkGeneration::generate_fully_connected<double>(n_agents, 0.0);
            }
        },
        "n_agents"_a,
        "weight"_a,
        "seed"_a);

    m.def(
        "generate_fully_connected_degroot",
        [](size_t n_agents, std::optional<typename Seldon::Network<Seldon::DeGrootModel::AgentT>::WeightT> weight, std::optional<size_t> seed) {
            if (seed.has_value()) {
                std::mt19937 gen(seed.value());
                return Seldon::NetworkGeneration::generate_fully_connected<Seldon::DeGrootModel::AgentT>(n_agents, gen);
            } else if (weight.has_value()) {
                return Seldon::NetworkGeneration::generate_fully_connected<Seldon::DeGrootModel::AgentT>(n_agents, weight.value());
            } else {
                return Seldon::NetworkGeneration::generate_fully_connected<Seldon::DeGrootModel::AgentT>(n_agents, 0.0);
            }
        },
        "n_agents"_a,
        "weight"_a,
        "seed"_a);

    m.def(
        "generate_fully_connected_deffuant",
        [](size_t n_agents, std::optional<typename Seldon::Network<Seldon::DeffuantModel::AgentT>::WeightT> weight, std::optional<size_t> seed) {
            if (seed.has_value()) {
                std::mt19937 gen(seed.value());
                return Seldon::NetworkGeneration::generate_fully_connected<Seldon::DeffuantModel::AgentT>(n_agents, gen);
            } else if (weight.has_value()) {
                return Seldon::NetworkGeneration::generate_fully_connected<Seldon::DeffuantModel::AgentT>(n_agents, weight.value());
            } else {
                return Seldon::NetworkGeneration::generate_fully_connected<Seldon::DeffuantModel::AgentT>(n_agents, 0.0);
            }
        },
        "n_agents"_a,
        "weight"_a,
        "seed"_a);

    m.def(
        "generate_fully_connected_deffuant_vector",
        [](size_t n_agents,
           std::optional<typename Seldon::Network<Seldon::DeffuantModelVector::AgentT>::WeightT> weight,
           std::optional<size_t> seed) {
            if (seed.has_value()) {
                std::mt19937 gen(seed.value());
                return Seldon::NetworkGeneration::generate_fully_connected<Seldon::DeffuantModelVector::AgentT>(n_agents, gen);
            } else if (weight.has_value()) {
                return Seldon::NetworkGeneration::generate_fully_connected<Seldon::DeffuantModelVector::AgentT>(n_agents, weight.value());
            } else {
                return Seldon::NetworkGeneration::generate_fully_connected<Seldon::DeffuantModelVector::AgentT>(n_agents, 0.0);
            }
        },
        "n_agents"_a,
        "weight"_a,
        "seed"_a);

    m.def(
        "generate_fully_connected_activity_driven",
        [](size_t n_agents,
           std::optional<typename Seldon::Network<Seldon::ActivityDrivenModel::AgentT>::WeightT> weight,
           std::optional<size_t> seed) {
            if (seed.has_value()) {
                std::mt19937 gen(seed.value());
                return Seldon::NetworkGeneration::generate_fully_connected<Seldon::ActivityDrivenModel::AgentT>(n_agents, gen);
            } else if (weight.has_value()) {
                return Seldon::NetworkGeneration::generate_fully_connected<Seldon::ActivityDrivenModel::AgentT>(n_agents, weight.value());
            } else {
                return Seldon::NetworkGeneration::generate_fully_connected<Seldon::ActivityDrivenModel::AgentT>(n_agents, 0.0);
            }
        },
        "n_agents"_a,
        "weight"_a,
        "seed"_a);

    m.def(
        "generate_fully_connected_activity_driven_inertial",
        [](size_t n_agents, std::optional<double> weight, std::optional<size_t> seed) {
            if (seed.has_value()) {
                std::mt19937 gen(seed.value());
                return Seldon::NetworkGeneration::generate_fully_connected<Seldon::InertialAgent>(n_agents, gen);
            } else if (weight.has_value()) {
                return Seldon::NetworkGeneration::generate_fully_connected<Seldon::InertialAgent>(n_agents, weight.value());
            } else {
                return Seldon::NetworkGeneration::generate_fully_connected<Seldon::InertialAgent>(n_agents, 0.0);
            }
        },
        "n_agents"_a,
        "weight"_a,
        "seed"_a);

    m.def("generate_from_file", &Seldon::NetworkGeneration::generate_from_file<Seldon::SimpleAgent>, "file"_a);

    m.def("generate_from_file", &Seldon::NetworkGeneration::generate_from_file<Seldon::DiscreteVectorAgent>, "file"_a);

    m.def("generate_from_file", &Seldon::NetworkGeneration::generate_from_file<Seldon::ActivityAgent>, "file"_a);

    m.def("generate_from_file", &Seldon::NetworkGeneration::generate_from_file<Seldon::InertialAgent>, "file"_a);

    m.def("generate_square_lattice_simple_agent", &Seldon::NetworkGeneration::generate_square_lattice<Seldon::SimpleAgent>, "n_edge"_a, "weight"_a);

    m.def("generate_square_lattice_discrete_vector_agent",
          &Seldon::NetworkGeneration::generate_square_lattice<Seldon::DiscreteVectorAgent>,
          "n_edge"_a,
          "weight"_a);

    m.def(
        "generate_square_lattice_activity_agent", &Seldon::NetworkGeneration::generate_square_lattice<Seldon::ActivityAgent>, "n_edge"_a, "weight"_a);

    m.def(
        "generate_square_lattice_inertial_agent", &Seldon::NetworkGeneration::generate_square_lattice<Seldon::InertialAgent>, "n_edge"_a, "weight"_a);

    m.def("parse_config_file", &Seldon::Config::parse_config_file, "file"_a);

    // network
    m.def("network_to_dot_file", &Seldon::network_to_dot_file<Seldon::SimpleAgent>, "network"_a, "file_path"_a);
    m.def("network_to_dot_file", &Seldon::network_to_dot_file<Seldon::DiscreteVectorAgent>, "network"_a, "file_path"_a);
    m.def("network_to_dot_file", &Seldon::network_to_dot_file<Seldon::ActivityAgent>, "network"_a, "file_path"_a);
    m.def("network_to_dot_file", &Seldon::network_to_dot_file<Seldon::InertialAgent>, "network"_a, "file_path"_a);

    m.def("network_to_file", &Seldon::network_to_file<Seldon::SimpleAgent>, "network"_a, "file_path"_a);
    m.def("network_to_file", &Seldon::network_to_file<Seldon::DiscreteVectorAgent>, "network"_a, "file_path"_a);
    m.def("network_to_file", &Seldon::network_to_file<Seldon::ActivityAgent>, "network"_a, "file_path"_a);
    m.def("network_to_file", &Seldon::network_to_file<Seldon::InertialAgent>, "network"_a, "file_path"_a);

    m.def("agents_from_file", &Seldon::agents_from_file<Seldon::SimpleAgent>, "file"_a);
    m.def("agents_from_file", &Seldon::agents_from_file<Seldon::DiscreteVectorAgent>, "file"_a);
    m.def("agents_from_file", &Seldon::agents_from_file<Seldon::ActivityAgent>, "file"_a);
    m.def("agents_from_file", &Seldon::agents_from_file<Seldon::InertialAgent>, "file"_a);

    m.def("agents_to_file", &Seldon::agents_to_file<Seldon::SimpleAgent>, "network"_a, "file_path"_a);
    m.def("agents_to_file", &Seldon::agents_to_file<Seldon::DiscreteVectorAgent>, "network"_a, "file_path"_a);
    m.def("agents_to_file", &Seldon::agents_to_file<Seldon::ActivityAgent>, "network"_a, "file_path"_a);
    m.def("agents_to_file", &Seldon::agents_to_file<Seldon::InertialAgent>, "network"_a, "file_path"_a);

    //--------------------------------------------------------------------------------------------------------------------
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

    py::class_<Seldon::bivariate_gaussian_copula<double, std::uniform_real_distribution<double>, std::uniform_real_distribution<double>>>(
        m, "Bivariate_Gaussian_Copula")
        .def(py::init<double, std::uniform_real_distribution<double>, std::uniform_real_distribution<double>>(), "covariance"_a, "dist1"_a, "dist2"_a)
        .def("__call__",
             &Seldon::bivariate_gaussian_copula<double, std::uniform_real_distribution<double>, std::uniform_real_distribution<double>>::template
             operator()<std::mt19937>,
             "gen"_a);

    m.def("hamming_distance", &Seldon::hamming_distance<double>, "v1"_a, "v2"_a);
}
