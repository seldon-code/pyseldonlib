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

//Config-Parser



//Network


//Model enum used in simulation options
    py::enum_<Seldon::Config::Model>(m, "Model")
        .value("DeGroot")

//Simulation class
    //this is a simulations options object which can be passed in the Simulation class constructor
    py::class_<Seldon::Config::SimulationOptions>(m, "SimulationOptions")
        .def(py::init<>())
        .def_readwrite();

    py::class_<Seldon::Simulation, Seldon::SimulationInterface>(m , "Simulation")
        .def(py::init<Config::SimulationOptions &, const std::optional<std::string &, const std::optional<std::string> &>())
        .def("run", &Seldon::Simulation::run, "Run the Simulation",
        py::arg("output_dir_path" = "./output")
        );


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
    //     .def(py::init<Seldon::Config::DeGrootSettings, Seldon::Network &>())
    //     .def_readwrite("convergence_tol", &Seldon::DeGrootModel::convergence_tol)
    //     .def_readwrite("network", &Seldon::DeGrootModel::network)
    //     .def_readwrite("agents_current_copy", &Seldon::DeGrootModel::agents_current_copy);

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