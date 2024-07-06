"""This module provides functions to run the simulation using the configuration file or the simulation options object."""

import pyseldon.seldoncore 

def run_simulation_from_config_file(config_file_path, agent_file_path = None, network_file_path = None, output_file_path = None):
    """Run the simulation using the configuration file.
    
    Parameters
    ----------
    config_file_path : str
        The path to the configuration file.
    
    agent_file_path : str
        The path to the agent file.
    
    network_file_path : str
        The path to the network file.
    
    output_file_path : str
        The path to the output file.
    """
    
    pyseldon.seldoncore.run_simulation(config_file_path = config_file_path, agent_file_path = agent_file_path, network_file_path = network_file_path, output_file_path = output_file_path)


def run_simulation_from_options(options, agent_file_path = None, network_file_path = None, output_file_path = None):
    """Run the simulation using the simulation options object.
    
    Parameters
    ----------
    options : object
        The simulation options object.
    
    agent_file_path : str
        The path to the agent file.
    
    network_file_path : str
        The path to the network file.
    
    output_file_path : str
        The path to the output file.
    """
    
    pyseldon.seldoncore.run_simulation(options = options, agent_file_path = agent_file_path, network_file_path = network_file_path, output_file_path = output_file_path)