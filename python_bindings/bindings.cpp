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

// pybind11 headers
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

using namespace std::string_literals;
using namespace pybind11::literals;
namespace py = pybind11;

namespace fs = std::filesystem;

void run_simulation(const std::string &config_file_path,
                    const std::optional<std::string> agent_file,
                    const std::optional<std::string> network_file,
                    const std::optional<std::string> output_dir_path_cli)
{
    fs::path output_dir_path = output_dir_path_cli.value_or( fs::path( "./output" ) );
    fs::create_directories( output_dir_path ); 


    auto simulation_options = Seldon::Config::parse_config_file( config_file_path );
    Seldon::Config::validate_settings( simulation_options );
    Seldon::Config::print_settings( simulation_options );

    std::unique_ptr<Seldon::SimulationInterface> simulation;

    if( simulation_options.model == Seldon::Config::Model::DeGroot )
    {
        simulation = std::make_unique<Seldon::Simulation<Seldon::DeGrootModel::AgentT>>(
            simulation_options, network_file, agent_file );
    }
    else if( simulation_options.model == Seldon::Config::Model::ActivityDrivenModel )
    {
        simulation = std::make_unique<Seldon::Simulation<Seldon::ActivityDrivenModel::AgentT>>(
            simulation_options, network_file, agent_file );
    }
    else if( simulation_options.model == Seldon::Config::Model::ActivityDrivenInertial )
    {
        simulation = std::make_unique<Seldon::Simulation<Seldon::InertialModel::AgentT>>(
            simulation_options, network_file, agent_file );
    }
    else if( simulation_options.model == Seldon::Config::Model::DeffuantModel )
    {
        auto model_settings = std::get<Seldon::Config::DeffuantSettings>( simulation_options.model_settings );
        if( model_settings.use_binary_vector )
        {
            simulation = std::make_unique<Seldon::Simulation<Seldon::DeffuantModelVector::AgentT>>(
                simulation_options, network_file, agent_file );
        }
        else
        {
            simulation = std::make_unique<Seldon::Simulation<Seldon::DeffuantModel::AgentT>>(
                simulation_options, network_file, agent_file );
        }
    }
    else
    {
        throw std::runtime_error( "Model has not been created" );
    }

    simulation->run( output_dir_path );
}

PYBIND11_MODULE(seldoncore, m)
{
    m.doc() = "Python bindings for Seldon Cpp Engine";

    m.def("run_simulation", &run_simulation,
        py::arg("config_file"),
        py::arg("agent_file")= std::optional<std::string>{},
        py::arg("network_file")= std::optional<std::string>{},
        py::arg("output_dir_path_cli")= std::optional<std::string>{}
        );
}