#include "../subprojects/seldon/include/models/DeGroot.hpp"
#include "../subprojects/seldon/include/agents/simple_agent.hpp"
#include "../subprojects/seldon/include/config_parser.hpp"
#include "../subprojects/seldon/include/model.hpp"
#include "../subprojects/seldon/include/network.hpp"

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
    py::class_<Seldon::DeGrootModel>( m, "DeGrootModel" )
        .def(py::init<>()) //Config::DeGrootSettings settings, NetworkT & network 
        
}