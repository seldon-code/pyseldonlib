"""This module provides functions to run the simulation using the configuration file or the simulation options object."""
# from . import core
from pyseldon import seldoncore

def run_simulation_from_config_file(config_file_path, agent_file_path = None, network_file_path = None, output_dir_path = None, initial_agents = [], final_agents = []):
    """Run the simulation using the configuration file.
    
    Parameters
    ----------
    config_file_path : str
        The path to the configuration file.
    
    agent_file_path : str
        The path to the agent file.
    
    network_file_path : str
        The path to the network file.
    
    output_dir_path : str
        The path to the output file.
    """
    seldoncore.run_simulation(config_file_path = config_file_path, options = None, agent_file_path = agent_file_path, network_file_path = network_file_path, output_dir_path = output_dir_path, initial_agents = initial_agents, final_agents = final_agents)


def run_simulation_from_options(options, agent_file_path = None, network_file_path = None, output_dir_path = None, initial_agents = [], final_agents = []):
    """Run the simulation using the simulation options object.
    
    Parameters
    ----------
    options : object
        The simulation options object.
    
    agent_file_path : str
        The path to the agent file.
    
    network_file_path : str
        The path to the network file.
    
    output_dir_path : str
        The path to the output file.
    """
    seldoncore.run_simulation(options = options, config_file_path = None,  agent_file_path = agent_file_path, network_file_path = network_file_path, output_dir_path = output_dir_path, initial_agents = initial_agents, final_agents = final_agents)

class simulation:
    """The simulation class."""
    def __init__(self, config_file_path = None, agent_file_path = None, network_file_path = None, output_dir_path = None, initial_agents = [], final_agents = []):
        """The simulation class constructor"""