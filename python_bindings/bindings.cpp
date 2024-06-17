#include "models/DeGroot.hpp"
#include "models/DeffuantModel.hpp"
#include "models/ActivityDrivenModel.hpp"
#include "models/InertialModel.hpp"
#include "config_parser.hpp"
#include "model.hpp"
#include "network.hpp"
#include "run_model.hpp"
#include "simulation.hpp"
#include "model_factory.hpp"
#include <optional>
#include <cstddef>

// pybind11 headers
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/operators.h>

using namespace std::string_literals;
using namespace pybind11::literals;
namespace py = pybind11;


PYBIND11_MODULE(seldoncore, m)
{
    m.doc() = "Python bindings for Seldon Cpp Engine";

//Agents

//Network
//exposing the network class so that we can directly create a model using the nework and run the simulation

    // py::class_<Seldon::Network>(m, "Network")
    //     .def(py::init<>())
    //     .def(py::init<const std::size_t>())
    //     .def(py::init<const std::vector<int> &>())//int
    //     .def(py::init<const std::vector<double> &>()) //double ->AgentT
    //     .def(py::init<const std::vector<std::vector<size_t>> &&, const std::vector<std::vector<double>> &&>()) //using WeightT->double
    //     .def();



//Config-Parser

    // So here the basic approach is making an object of each setting passing it to the SimulationOptions and also making an object of Simulation options and passing it to the Simulation class constructor
    //enum class Model (might require to change this name as its a todo in the seldon codebase )
    py::enum_<Seldon::Config::Model>(m, "Model")
        .value("DeGroot", Seldon::Config::Model::DeGroot, "DeGroot Model")
        .value("DeffuantModel", Seldon::Config::Model::DeffuantModel, "Deffuant Model")
        .value("ActivityDrivenModel", Seldon::Config::Model::ActivityDrivenModel, "Activity Driven Model")
        .value("ActivityDrivenInertial", Seldon::Config::Model::ActivityDrivenInertial, "Activity Driven Inertial Model"); //how to use this in python: Model.DeGroot

    //output settings object to be passed in the simulation options
    py::class_<Seldon::Config::OutputSettings>(m, "OutputSettings")
        .def(py::init<>())
        .def_readwrite("n_output_agents", &Seldon::Config::OutputSettings::n_output_agents, "Number of agents to output")
        .def_readwrite("n_output_network", &Seldon::Config::OutputSettings::n_output_network, "Number of network to output")
        .def_readwrite("print_progress", &Seldon::Config::OutputSettings::print_progress, "Print progress?")
        .def_readwrite("output_initial", &Seldon::Config::OutputSettings::output_initial, "Output initial?")
        .def_readwrite("start_output", &Seldon::Config::OutputSettings::start_output, "Start output")
        .def_readwrite("start_numbering_from", &Seldon::Config::OutputSettings::start_numbering_from, "Start numbering from");

    //degroot setting object to be passed in the simulation options
    py::class_<Seldon::Config::DeGrootSettings>(m, "DeGrootSettings")
        .def(py::init<>())
        .def_readwrite("max_iterations", &Seldon::Config::DeGrootSettings::max_iterations, "Maximum number of iterations")
        .def_readwrite("convergence_tol", &Seldon::Config::DeGrootSettings::convergence_tol, "Convergence tolerance");
    
    //deffuant setting object to be passed in the simulation options
    py::class_<Seldon::Config::DeffuantSettings>(m, "DeffuantSettings")
        .def(py::init<>())
        .def_readwrite("max_iterations", &Seldon::Config::DeffuantSettings::max_iterations, "Maximum number of iterations")
        .def_readwrite("homophily_threshold", &Seldon::Config::DeffuantSettings::homophily_threshold, "Homophily threshold")
        .def_readwrite("mu", &Seldon::Config::DeffuantSettings::mu, "Convergence parameter")
        .def_readwrite("use_network", &Seldon::Config::DeffuantSettings::use_network, "Use network?")
        .def_readwrite("use_binary_vector", &Seldon::Config::DeffuantSettings::use_binary_vector, "Use binary vector?")
        .def_readwrite("dim", &Seldon::Config::DeffuantSettings::dim);

    //ActivityDriven setting object to be passed in the simulation options
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

    //ActivityDrivenInertial setting object to be passed in the simulation options 
    py::class_<Seldon::Config::ActivityDrivenInertialSettings>(m, "ActivityDrivenInertialSettings")
        .def(py::init<>())
        .def_readwrite("friction_coefficient", &Seldon::Config::ActivityDrivenInertialSettings::friction_coefficient);

    //InitialNetwork setting object to be passed in the simulation options
    py::class_<Seldon::Config::InitialNetworkSettings>(m, "InitialNetworkSettings")
        .def(py::init<>())
        .def_readwrite("file", &Seldon::Config::InitialNetworkSettings::file)
        .def_readwrite("n_agents", &Seldon::Config::InitialNetworkSettings::n_agents)
        .def_readwrite("n_connections", &Seldon::Config::InitialNetworkSettings::n_connections);

    //SimulationOptions class object creation which is to be passed in the simulation class
    py::class_<Seldon::Config::SimulationOptions>(m, "SimulationOptions")
        .def(py::init<>())
        .def_readwrite("model_string", &Seldon::Config::SimulationOptions::model_string)
        .def_readwrite("model", &Seldon::Config::SimulationOptions::model)
        .def_readwrite("model_settings", &Seldon::Config::SimulationOptions::model_settings)
        .def_readwrite("rng_seed", &Seldon::Config::SimulationOptions::rng_seed)
        .def_readwrite("output_settings", &Seldon::Config::SimulationOptions::output_settings)
        .def_readwrite("network_settings", &Seldon::Config::SimulationOptions::network_settings);
    

//Simulation class
    py::class_<Seldon::Simulation, Seldon::SimulationInterface>(m , "Simulation")
        .def(py::init<Config::SimulationOptions &, const std::optional<std::string &, const std::optional<std::string> &>())
        .def("run", &Seldon::Simulation::run, "Run the Simulation",
        py::arg("output_dir_path" = "./output")
        )
        .def("create_network", &Seldon::Simulation::create_network, "Create the network") //exposing it here so that it can be used to create a network file
        .def("create_model", &Sledon::Simulation::create_model, "Create the model") //exposing it here so that it can be used to create a model according to the simulation options

    




// Models

    //Model class
    // py::class_<Seldon::Model<int>>(m, "Model")
    //     .def(py::init<>())
    //     .def(py::init<std::optional<size_t>>())
    //     .def("initialize_iterations", &Seldon::Model<int>::initialize_iterations);

    // py::class_<Seldon::Model<int>>(m, "Model")
    //     .def(py::init<>())
    //     .def(py::init<std::optional<size_t>>())
    //     .def("initialize_iterations", &Seldon::Model<int>::initialize_iterations);


    //DeGroot --getting errors due to private variables
    // Assuming <AgentType> as int and double only
    //int
    // py::class_<Seldon::DeGrootModel, Seldon::Model<int>>( m, "DeGrootModel" )
    //     .def(py::init<Seldon::Config::DeGrootSettings&, Seldon::Config::Netw>())

    // //double
    //  py::class_<Seldon::DeGrootModel, Seldon::Model<double>>( m, "DeGrootModel" )
    //     .def(py::init<Seldon::Config::DeGrootSettings, Seldon::NetworkT &>())
    //     .def_readwrite("convergence_tol", &Seldon::DeGrootModel::convergence_tol)
    //     .def_readwrite("network", &Seldon::DeGrootModel::network)
    //     .def_readwrite("agents_current_copy", &Seldon::DeGrootModel::agents_current_copy);

// Main wrapper
// use this to achieve complete abstration of whats happening underneath ;)
// but added include/main.hpp and include/run_model for the compiler to indentify this
    m.def("run_model", &run_model ,"The Wrapper for running the model",
        py::arg("config_file_path"),
        py::arg("agent_file"),
        py::arg("network_file"),
        py::arg("output_dir_path") = "./output");
}