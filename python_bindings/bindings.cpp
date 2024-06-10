#include "config_parser.hpp"
#include "model.hpp"
#include "network.hpp"
#include "run_model.hpp"



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

//Simulation


// Models

    //DeGroot 
    // py::class_<Seldon::DeGrootModel>( m, "DeGrootModel" )
    //     .def(py::init<>(Seldon::Config::DeGrootSettings, Seldon::)) //->takes Config::DeGrootSettings settings, NetworkT & network 

// Main wrapper
// use this to achieve complete abstration of whats happening underneath ;)
// but added include/main.hpp for the compiler to indentify this
    m.def("run_model", &run_model ,"The Wrapper for running the model",
        py::arg("config_file_path"),
        py::arg("agent_file") = "",
        py::arg("network_file") = "",
        py::arg("output_dir_path") = "./output");
}