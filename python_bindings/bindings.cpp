#include "config_parser.hpp"
#include "model.hpp"
#include "model_factory.hpp"
#include "models/ActivityDrivenModel.hpp"
#include "models/DeGroot.hpp"
#include "models/DeffuantModel.hpp"
#include "models/InertialModel.hpp"
#include "network.hpp"
#include "simulation.hpp"

#include <cstddef>
#include <optional>
#include <random>
#include <variant>

// pybind11 headers
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

using namespace std::string_literals;
using namespace pybind11::literals;
namespace py = pybind11;

namespace fs = std::filesystem;

void run_simulation(const std::optional<std::string> &config_file_path,
                    const std::optional<Seldon::Config::SimulationOptions> &options,
                    const std::optional<std::string> agent_file,
                    const std::optional<std::string> network_file,
                    const std::optional<std::string> output_dir_path_cli) {

    fs::path output_dir_path = output_dir_path_cli.value_or(fs::path("./output"));
    fs::create_directories(output_dir_path);
    Seldon::Config::SimulationOptions simulation_options;

    if (config_file_path) {
        simulation_options = Seldon::Config::parse_config_file(config_file_path.value());
    } else if (options) {
        simulation_options = options.value();
        if (simulation_options.model_string == "DeGroot") {
            simulation_options.model = Seldon::Config::Model::DeGroot;
        } else if (simulation_options.model_string == "ActivityDriven") {
            simulation_options.model = Seldon::Config::Model::ActivityDrivenModel;
        } else if (simulation_options.model_string == "Deffuant") {
            simulation_options.model = Seldon::Config::Model::DeffuantModel;
        } else if (simulation_options.model_string == "ActivityDrivenInertial") {
            simulation_options.model = Seldon::Config::Model::ActivityDrivenInertial;
        } else {
            throw std::runtime_error(fmt::format("Invalid model string {}", simulation_options.model_string));
        }
    } else {
        throw std::runtime_error("Either config_file_path or simulation_options must be provided");
    }

    Seldon::Config::validate_settings(simulation_options);
    Seldon::Config::print_settings(simulation_options);

    std::unique_ptr<Seldon::SimulationInterface> simulation;

    if (simulation_options.model == Seldon::Config::Model::DeGroot) {
        simulation = std::make_unique<Seldon::Simulation<Seldon::DeGrootModel::AgentT>>(simulation_options, network_file, agent_file);
    } else if (simulation_options.model == Seldon::Config::Model::ActivityDrivenModel) {
        simulation = std::make_unique<Seldon::Simulation<Seldon::ActivityDrivenModel::AgentT>>(simulation_options, network_file, agent_file);
    } else if (simulation_options.model == Seldon::Config::Model::ActivityDrivenInertial) {
        simulation = std::make_unique<Seldon::Simulation<Seldon::InertialModel::AgentT>>(simulation_options, network_file, agent_file);
    } else if (simulation_options.model == Seldon::Config::Model::DeffuantModel) {
        auto model_settings = std::get<Seldon::Config::DeffuantSettings>(simulation_options.model_settings);
        if (model_settings.use_binary_vector) {
            simulation = std::make_unique<Seldon::Simulation<Seldon::DeffuantModelVector::AgentT>>(simulation_options, network_file, agent_file);
        } else {
            simulation = std::make_unique<Seldon::Simulation<Seldon::DeffuantModel::AgentT>>(simulation_options, network_file, agent_file);
        }
    } else {
        throw std::runtime_error("Model has not been created");
    }

    simulation->run(output_dir_path);
}

template <typename AgentT, typename WeightT = double>
void generate_networks_bindings(py::module &m, std::string network_model) {
    // string to store the type_network_model for naming the class , to do ask the name for each class
    std::string network_classname;

    if (network_model == "DeGroot") {
        network_classname = "DeGrootNetwork";
    } else if (network_model == "Deffuant") {
        network_classname = "DeffuantNetwork";
    } else if (network_model == "ActivityDriven") {
        network_classname = "ActivityDrivenNetwork";
    } else if (network_model == "ActivityDrivenInertial") {
        network_classname = "InertialNetwork";
    } else {
        network_classname = "Network";
    }

    py::class_<Seldon::Network<AgentT, WeightT>>(m, network_classname.c_str())
        .def(py::init<>())
        .def(py::init<const std::size_t>())
        .def(py::init<const std::vector<AgentT> &>())
        .def(py::init<std::vector<std::vector<size_t>> &&,
                      std::vector<std::vector<WeightT>> &&,
                      typename Seldon::Network<AgentT, WeightT>::EdgeDirection>())
        .def("n_agents", &Seldon::Network<AgentT, WeightT>::n_agents, "Number of agents in the network")
        .def("n_edges", &Seldon::Network<AgentT, WeightT>::n_edges, "Number of edges in the network")
        .def("direction", &Seldon::Network<AgentT, WeightT>::direction, "Direction of the network")
        .def(
            "strongly_connected_components",
            &Seldon::Network<AgentT, WeightT>::strongly_connected_components,
            "Strongly connected components in the graph") // https://stackoverflow.com/questions/64632424/interpreting-static-cast-static-castvoid-petint-syntax
                                                          // // https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods
        .def("get_neighbours",
             static_cast<std::span<const size_t> (Seldon::Network<AgentT, WeightT>::*)(std::size_t) const>(
                 &Seldon::Network<AgentT, WeightT>::get_neighbours),
             "Gives a view into the neighbour indices going out/coming in at agent_idx")
        .def("get_neighbours",
             static_cast<std::span<size_t> (Seldon::Network<AgentT, WeightT>::*)(std::size_t)>(&Seldon::Network<AgentT, WeightT>::get_neighbours))
        .def("get_weights",
             static_cast<std::span<const double> (Seldon::Network<AgentT, WeightT>::*)(std::size_t) const>(
                 &Seldon::Network<AgentT, WeightT>::get_weights),
             "Gives a view into the weights going out/coming in at agent_idx")
        .def("get_weights",
             static_cast<std::span<double> (Seldon::Network<AgentT, WeightT>::*)(std::size_t)>(&Seldon::Network<AgentT, WeightT>::get_weights))
        .def("set_weights", &Seldon::Network<AgentT, WeightT>::set_weights, "Set the weights of the network")
        .def("set_neighbours_and_weights",
             static_cast<void (Seldon::Network<AgentT, WeightT>::*)(std::size_t, std::span<const size_t>, const WeightT &)>(
                 &Seldon::Network<AgentT, WeightT>::set_neighbours_and_weights),
             "Set the neighbours and weights of the network from a view only a constant value "
             "weight is assigned")
        .def("set_neighbours_and_weights",
             static_cast<void (Seldon::Network<AgentT, WeightT>::*)(std::size_t, std::span<const size_t>, std::span<const WeightT>)>(
                 &Seldon::Network<AgentT, WeightT>::set_neighbours_and_weights),
             "Set the neighbours and weights of the network from a view takes in the neighbours "
             "and weights list")
        .def("push_back_neighbour_and_weight",
             &Seldon::Network<AgentT, WeightT>::push_back_neighbour_and_weight,
             "Push back a neighbour and weight to the network") // takes in (size_T, size_t, double)
        .def("transpose", &Seldon::Network<AgentT, WeightT>::transpose, "Transpose the network, without switching the direction flag (expensive).")
        .def("toggle_incoming_outgoing",
             &Seldon::Network<AgentT, WeightT>::toggle_incoming_outgoing,
             "Switches the direction flag *without* transposing the network (expensive)")
        .def("switch_direction_flag",
             &Seldon::Network<AgentT, WeightT>::switch_direction_flag,
             "Only switches the direction flag. This effectively transposes the network and, "
             "simultaneously, changes its representation.")
        .def("remove_double_counting", &Seldon::Network<AgentT, WeightT>::remove_double_counting, "Removes doubly counted edges in the network")
        .def("clear", &Seldon::Network<AgentT, WeightT>::clear, "Clear the network");
}

// network generating functions
template <typename AgentType>
void generate_network_generation_bindings(py::module &m) {
    m.def("generate_n_connections", &Seldon::NetworkGeneration::generate_n_connections<AgentType>);
    m.def("generate_fully_connected", &Seldon::NetworkGeneration::generate_fully_connected<AgentType>);
    m.def("generate_fully_connected",
          static_cast < Seldon::Network<AgentType (Seldon::NetworkGeneration::*)(std::size_t &, typename Network<AgentType>::WeightT)>(
              &Seldon::NetworkGeneration::generate_fully_connected<AgentType>));
    m.def("generate_fully_connected",
          static_cast < Seldon::Network<AgentType (Seldon::NetworkGeneration::*)(std::size_t &, std::mt19937 &)>(
              &Seldon::NetworkGeneration::generate_fully_connected<AgentType>));
    m.def("generate_from_file", &Seldon::NetworkGeneration::generate_from_file<AgentType>);
    m.def("generate_square_lattice", &Seldon::NetworkGeneration::generate_square_lattice<AgentType>);
}

// great agent opretions but might be not much used in python
template <typename AgentType>
void generate_io_bindings(py::module &m) {
    // agents
    m.def("agent_to_string", &Seldon::agent_to_string<AgentType>);
    m.def("opinion_to_string", &Seldon::opinion_to_string<AgentType>);
    m.def("agent_from_string", &Seldon::agent_from_string<AgentType>);
    // m.def("agent_to_string_column_names", &Seldon::agent_to_string_column_names<AgentType>); //this one is not usable in python
    m.def("agents_to_file", &Seldon::agents_to_file<AgentType>, py::arg("network"), py::arg("file_path"));
    m.def("agents_from_file", &Seldon::agents_from_file<AgentType>, py::arg("file"));

    // network
    m.def("network_to_dot_file", &Seldon::network_to_dot_file<AgentType>, py::arg("network"), py::arg("file_path"));
    m.def("network_to_file", &Seldon::network_to_file<AgentType>, py::arg("network"), py::arg("file_path"));
}

PYBIND11_MODULE(seldoncore, m) {
    m.doc() = "Python bindings for Seldon Cpp Engine";

    m.def("run_simulation",
          &run_simulation,
          "config_file"_a = std::optional<std::string>{},
          "agent_file"_a = std::optional<std::string>{},
          "network_file"_a = std::optional<std::string>{},
          "output_dir_path_cli"_a = std::optional<std::string>{},
          "options"_a = std::optional<Seldon::Config::SimulationOptions>{});

    //--------------------------------------------------------------------
    // output settings instance to be passed in the simulation options
    py::class_<Seldon::Config::OutputSettings>(m, "OutputSettings")
        .def(py::init([](std::optional<size_t> n_output_agents = std::nullopt,
                         std::optional<size_t> n_output_network = std::nullopt,
                         bool print_progress = false,
                         bool output_initial = true,
                         size_t start_output = 1,
                         size_t start_numbering_from = 0) {
            Seldon::Config::OutputSettings output_settings;
            output_settings.n_output_agents = n_output_agents.value_or(0);
            output_settings.n_output_network = n_output_network.value_or(0);
            output_settings.print_progress = print_progress;
            output_settings.output_initial = output_initial;
            output_settings.start_output = start_output;
            output_settings.start_numbering_from = start_numbering_from;
            py::print("Using Output Settings");
            py::print(py::str("n_output_agents      : {} (Int)")
                          .format(n_output_agents)); //($) // assuming some default values if not specified this is the default value symbol($)
            py::print(py::str("n_output_network     : {} (Int)").format(n_output_network)); //($)
            py::print(py::str("print_progress       : {}").format(print_progress));
            py::print(py::str("output_initial       : {}").format(output_initial));
            py::print(py::str("start_output         : {}").format(start_output));
            py::print(py::str("start_numbering_from : {}").format(start_numbering_from));
            py::print("Which can be changed using the OutputSettings Instance");
            return output_settings;
        }))
        .def_readwrite("n_output_agents", &Seldon::Config::OutputSettings::n_output_agents, "Number of agents to output")
        .def_readwrite("n_output_network", &Seldon::Config::OutputSettings::n_output_network, "Number of network to output")
        .def_readwrite("print_progress", &Seldon::Config::OutputSettings::print_progress, "Print progress?")
        .def_readwrite("output_initial", &Seldon::Config::OutputSettings::output_initial, "Output initial?")
        .def_readwrite("start_output", &Seldon::Config::OutputSettings::start_output, "Start output")
        .def_readwrite("start_numbering_from", &Seldon::Config::OutputSettings::start_numbering_from, "Start numbering from");

    // degroot setting instance to be passed in the simulation options
    py::class_<Seldon::Config::DeGrootSettings>(m, "DeGrootSettings")
        .def(py::init(
            [](int max_iterations = 10, double convergence_tol = 1e-6) {
                Seldon::Config::DeGrootSettings degroot_settings;
                degroot_settings.max_iterations = max_iterations;
                degroot_settings.convergence_tol = convergence_tol;
                py::print("Using DeGroot Settings");
                py::print(py::str("max_iterations    : {} (Int)").format(max_iterations)); //($)
                py::print(py::str("convergence_tol  : {}").format(convergence_tol));
                py::print("Which can be changed using the DeGrootSettings instance");
                return degroot_settings;
            }),
            py::arg("max_iterations") = 10,
            py::arg("convergence_tol")=1e-6)
        .def_readwrite("max_iterations", &Seldon::Config::DeGrootSettings::max_iterations, "Maximum number of iterations")
        .def_readwrite("convergence_tol", &Seldon::Config::DeGrootSettings::convergence_tol, "Convergence tolerance");

    // deffuant setting instance to be passed in the simulation options
    py::class_<Seldon::Config::DeffuantSettings>(m, "DeffuantSettings")
        .def(py::init([](int max_iterations = 10,
                         double homophily_threshold = 0.2,
                         double mu = 0.5,
                         bool use_network = false,
                         bool use_binary_vector = false,
                         size_t dim = 1) {
                 Seldon::Config::DeffuantSettings deffuant_settings;
                 deffuant_settings.max_iterations = max_iterations;
                 deffuant_settings.homophily_threshold = homophily_threshold;
                 deffuant_settings.mu = mu;
                 deffuant_settings.use_network = use_network;
                 deffuant_settings.use_binary_vector = use_binary_vector;
                 deffuant_settings.dim = dim;
                 py::print("Using Deffuant Settings");
                 py::print(py::str("max_iterations     : {} (Int)").format(max_iterations)); //($)
                 py::print(py::str("homophily_threshold: {}").format(homophily_threshold));
                 py::print(py::str("mu                 : {}").format(mu));
                 py::print(py::str("use_network        : {}").format(use_network));
                 py::print(py::str("use_binary_vector  : {}").format(use_binary_vector));
                 py::print(py::str("dim                : {}").format(dim));
                 py::print("Which can be changed using the DeffuantSettings instance");
                 return deffuant_settings;
             }),
             py::arg("max_iterations") = 10,
             py::arg("homophily_threshold") = 0.2,
             py::arg("mu") = 0.5,
             py::arg("use_network") = false,
             py::arg("use_binary_vector") = false,
             py::arg("dim") = 1)
        .def_readwrite("max_iterations", &Seldon::Config::DeffuantSettings::max_iterations, "Maximum number of iterations")
        .def_readwrite("homophily_threshold", &Seldon::Config::DeffuantSettings::homophily_threshold, "Homophily threshold")
        .def_readwrite("mu", &Seldon::Config::DeffuantSettings::mu, "Convergence parameter")
        .def_readwrite("use_network", &Seldon::Config::DeffuantSettings::use_network, "Use network?")
        .def_readwrite("use_binary_vector", &Seldon::Config::DeffuantSettings::use_binary_vector, "Use binary vector?")
        .def_readwrite("dim", &Seldon::Config::DeffuantSettings::dim);

    // ActivityDriven setting instance to be passed in the simulation options
    py::class_<Seldon::Config::ActivityDrivenSettings>(m, "ActivityDrivenSettings")
        .def(py::init([](int max_iterations = 10,
                         double dt = 0.01,
                         int m = 10,
                         double eps = 0.01,
                         double gamma = 2.1,
                         double alpha = 3.0,
                         double homophily = 0.5,
                         double reciprocity = 0.5,
                         double K = 3.0,
                         bool mean_activities = false,
                         bool mean_weights = false,
                         size_t n_bots = 0,
                         std::vector<int> bot_m = std::vector<int>(0),
                         std::vector<double> bot_activity = std::vector<double>(0),
                         std::vector<double> bot_opinion = std::vector<double>(0),
                         std::vector<double> bot_homophily = std::vector<double>(0),
                         bool use_reluctances = false,
                         double reluctance_mean = 1.0,
                         double reluctance_sigma = 0.25,
                         double reluctance_eps = 0.01,
                         double covariance_factor = 0.0) {
            Seldon::Config::ActivityDrivenSettings activity_driven_settings;
            py::print("Using Activity Driven Settings");
            py::print(py::str("max_iterations    : {}").format(max_iterations));
            py::print(py::str("dt                : {}").format(dt));
            py::print(py::str("m                 : {}").format(m));
            py::print(py::str("eps               : {}").format(eps));
            py::print(py::str("gamma             : {}").format(gamma));
            py::print(py::str("alpha             : {}").format(alpha));
            py::print(py::str("homophily         : {}").format(homophily));
            py::print(py::str("reciprocity       : {}").format(reciprocity));
            py::print(py::str("K                 : {}").format(K));
            py::print(py::str("mean_activities   : {} (boolean)").format(mean_activities));
            py::print(py::str("mean_weights      : {} (boolean)").format(mean_weights));
            py::print(py::str("n_bots            : {}").format(n_bots)); //@TODO why is this here? from seldon codebase
            py::print(py::str("bot_m             : {}").format(bot_m));
            py::print(py::str("bot_activity      : {}").format(bot_activity));
            py::print(py::str("bot_opinion       : {}").format(bot_opinion));
            py::print(py::str("bot_homophily     : {}").format(bot_homophily));
            py::print(py::str("use_reluctances   : {}").format(use_reluctances));
            py::print(py::str("reluctance_mean   : {}").format(reluctance_mean));
            py::print(py::str("reluctance_sigma  : {}").format(reluctance_sigma));
            py::print(py::str("reluctance_eps    : {}").format(reluctance_eps));
            py::print(py::str("covariance_factor : {}").format(covariance_factor));
            py::print("Which can be changed using the ActivityDrivenSettings instance");
            activity_driven_settings.max_iterations = max_iterations;
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
        }),py::arg("max_iterations") = 10, py::arg("dt") = 0.01, py::arg("m") = 10, py::arg("eps") = 0.01, py::arg("gamma") = 2.1, py::arg("alpha") = 3.0, py::arg("homophily") = 0.5, py::arg("reciprocity") = 0.5, py::arg("K") = 3.0, py::arg("mean_activities") = false, py::arg("mean_weights") = false, py::arg("n_bots") = 0, py::arg("bot_m") = std::vector<int>(0), py::arg("bot_activity") = std::vector<double>(0), py::arg("bot_opinion") = std::vector<double>(0), py::arg("bot_homophily") = std::vector<double>(0), py::arg("use_reluctances") = false, py::arg("reluctance_mean") = 1.0, py::arg("reluctance_sigma") = 0.25, py::arg("reluctance_eps") = 0.01, py::arg("covariance_factor") = 0.0)
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
        .def(py::init([](int max_iterations = 10,
                         double dt = 0.01,
                         int m = 10,
                         double eps = 0.01,
                         double gamma = 2.1,
                         double alpha = 3.0,
                         double homophily = 0.5,
                         double reciprocity = 0.5,
                         double K = 3.0,
                         bool mean_activities = false,
                         bool mean_weights = false,
                         size_t n_bots = 0,
                         std::vector<int> bot_m = std::vector<int>(0),
                         std::vector<double> bot_activity = std::vector<double>(0),
                         std::vector<double> bot_opinion = std::vector<double>(0),
                         std::vector<double> bot_homophily = std::vector<double>(0),
                         bool use_reluctances = false,
                         double reluctance_mean = 1.0,
                         double reluctance_sigma = 0.25,
                         double reluctance_eps = 0.01,
                         double covariance_factor = 0.0,
                         double friction_coefficient = 1.0) {
            Seldon::Config::ActivityDrivenInertialSettings activity_driven_inertial_settings;
            py::print(py::str("max_iterations       : {}").format(max_iterations));
            py::print(py::str("dt                   : {}").format(dt));
            py::print(py::str("m                    : {}").format(m));
            py::print(py::str("eps                  : {}").format(eps));
            py::print(py::str("gamma                : {}").format(gamma));
            py::print(py::str("alpha                : {}").format(alpha));
            py::print(py::str("homophily            : {}").format(homophily));
            py::print(py::str("reciprocity          : {}").format(reciprocity));
            py::print(py::str("K                    : {}").format(K));
            py::print(py::str("mean_activities      : {} (boolean)").format(mean_activities));
            py::print(py::str("mean_weights         : {} (boolean)").format(mean_weights));
            py::print(py::str("n_bots               : {}").format(n_bots)); //@TODO why is this here? from seldon codebase
            py::print(py::str("bot_m                : {}").format(bot_m));
            py::print(py::str("bot_activity         : {}").format(bot_activity));
            py::print(py::str("bot_opinion          : {}").format(bot_opinion));
            py::print(py::str("bot_homophily        : {}").format(bot_homophily));
            py::print(py::str("use_reluctances      : {}").format(use_reluctances));
            py::print(py::str("reluctance_mean      : {}").format(reluctance_mean));
            py::print(py::str("reluctance_sigma     : {}").format(reluctance_sigma));
            py::print(py::str("reluctance_eps       : {}").format(reluctance_eps));
            py::print(py::str("covariance_factor    : {}").format(covariance_factor));
            py::print(py::str("friction_coefficient :").format(friction_coefficient));
            py::print("Which can be changed using the ActivityDrivenInertialSettings instance");
            activity_driven_inertial_settings.max_iterations = 10;
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
            activity_driven_settings.friction_coefficient = friction_coefficient;
            return activity_driven_inertial_settings;
        }), py::arg("max_iterations")= 10, py::arg("dt")= 0.01, py::arg("m")= 10, py::arg("eps")= 0.01, py::arg("gamma")= 2.1, py::arg("alpha")= 3.0, py::arg("homophily")= 0.5, py::arg("reciprocity")= 0.5, py::arg("K")= 3.0, py::arg("mean_activities")= false, py::arg("mean_weights")= false, py::arg("n_bots")= 0, py::arg("bot_m")= std::vector<int>(0), py::arg("bot_activity")= std::vector<double>(0), py::arg("bot_opinion")= std::vector<double>(0), py::arg("bot_homophily")= std::vector<double>(0), py::arg("use_reluctances")= false, py::arg("reluctance_mean")= 1.0, py::arg("reluctance_sigma")= 0.25, py::arg("reluctance_eps")= 0.01, py::arg("covariance_factor")= 0.0, py::arg("friction_coefficient")= 1.0)
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
        .def_readwrite("covariance_factor", &Seldon::Config::ActivityDrivenInertialSettings::covariance_factor);

    // InitialNetwork setting instance to be passed in the simulation options
    py::class_<Seldon::Config::InitialNetworkSettings>(m, "InitialNetworkSettings")
        .def(py::init([]() {
            Seldon::Config::InitialNetworkSettings initial_network_settings;
            py::print("Using Initial Network Settings");
            py::print("file            : None (String)");
            py::print("n_agents        : 200 (Int)");
            py::print("n_connections   : 10 (Int)");
            py::print("Which can be changed using the InitialNetworkSettings instance");
            return initial_network_settings;
        }))
        .def_readwrite("file", &Seldon::Config::InitialNetworkSettings::file)
        .def_readwrite("n_agents", &Seldon::Config::InitialNetworkSettings::n_agents)
        .def_readwrite("n_connections", &Seldon::Config::InitialNetworkSettings::n_connections);

    py::class_<Seldon::Config::SimulationOptions>(m, "SimulationOptions")
        .def(py::init([]() {
            Seldon::Config::SimulationOptions simulation_options;
            py::print("Using Simulation Options");
            py::print("model_string     : None (String)");
            py::print("rng_seed         : Random seed");
            py::print("output_settings  : OutputSettings");
            py::print("model_settings   : ModelVariant(Instance of DeGrootSettings, "
                      "ActivityDrivenSettings, ActivityDrivenInertialSettings, DeffuantSettings)");
            py::print("network_settings : InitialNetworkSettings(Instance)");
            py::print("Which can be changed using the SimulationOptions instance");
            return simulation_options;
        }))
        .def_readwrite("model_string", &Seldon::Config::SimulationOptions::model_string)
        .def_readwrite("rng_seed", &Seldon::Config::SimulationOptions::rng_seed)
        .def_readwrite("output_settings", &Seldon::Config::SimulationOptions::output_settings)
        .def_readwrite("model_settings", &Seldon::Config::SimulationOptions::model_settings)
        .def_readwrite("network_settings", &Seldon::Config::SimulationOptions::network_settings);

    //--------------------------------------------------------------------------

    // network bindings creation
    generate_networks_bindings<int>(m, "Network"); // default network
    generate_networks_bindings<double>(m, "FloatTypeNetwork");
    generate_networks_bindings<Seldon::DeGrootModel::AgentT>(m, "DeGrootNetwork");
    generate_networks_bindings<Seldon::DeffuantModel::AgentT>(m, "DeffuantNetwork");
    generate_networks_bindings<Seldon::ActivityDrivenModel::AgentT>(m, "ActivityDrivenNetwork");
    generate_networks_bindings<Seldon::InertialModel::AgentT>(m, "ActivityDrivenInertialNetwork");
}
