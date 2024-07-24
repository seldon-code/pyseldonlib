"""This module provides functions to run the simulation using the configuration file or the simulation options object."""

from pyseldon import seldoncore


def run_simulation_from_config_file(
    config_file_path, agent_file_path=None, network_file_path=None, output_dir_path=None
):
    """
    Run the simulation using the configuration file.

    Args:
        config_file_path (str): The path to the configuration file.
        agent_file_path (str): The path to the agent file.
        network_file_path (str): The path to the network file.
        output_dir_path (str): The path to the output file.
    """
    seldoncore.run_simulation(
        config_file_path=config_file_path,
        options=None,
        agent_file_path=agent_file_path,
        network_file_path=network_file_path,
        output_dir_path=output_dir_path,
    )


def run_simulation_from_options(
    options, agent_file_path=None, network_file_path=None, output_dir_path=None
):
    """
    Run the simulation using the simulation options object.

    Args:
        options (object): The simulation options object.
        agent_file_path (str): The path to the agent file.
        network_file_path (str): The path to the network file.
        output_dir_path (str): The path to the output directory.
    """
    seldoncore.run_simulation(
        options=options,
        config_file_path=None,
        agent_file_path=agent_file_path,
        network_file_path=network_file_path,
        output_dir_path=output_dir_path,
    )


class Simulation:
    """The Simulation class provides functions to run the simulation using the simulation options object, agent file, and network file."""
    def __init__(self, model_string="DeGroot", agent_file_path=None, network_file_path=None):
        """
        Initialize the Simulation object.
        
        Args:
            model_string (str): The model string.
            agent_file_path (str, optional): The path to the agent file.
            network_file_path (str, optional): The path to the network file.
        """
        if(model_string == "DeGroot"):
            self.simulation = seldoncore.SimulationDeGroot()

        elif(model_string == "Deffuant"):
            self.simulation = seldoncore.SimulationDeffuant()

        elif(model_string == "DeffuantVector"):
            self.simulation = seldoncore.SimulationDeffuantVector()

        elif(model_string == "ActivityDriven"):
            self.simulation = seldoncore.SimulationActivityDriven()
        
        elif(model_string == "ActivityDrivenInertial"):
            self.simulation = seldoncore.SimulationInertial()
        
        else:
            TypeError("Model not found")
        