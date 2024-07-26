"""This module provides the main interface to the pyseldon package."""
from pyseldon import seldoncore
from typing import List, Optional, Union, LiteralString
# from .simulation import run_simulation_from_config_file, run_simulation_from_options

__all__ = ["run_simulation_from_config_file", "run_simulation_from_options"]

# """This module provides functions to run the simulation using the configuration file or the simulation options object."""

def run_simulation_from_config_file(
    config_file_path: str, agent_file_path: Optional[str] =None, network_file_path: Optional[str] =None, output_dir_path: Optional[str] =None
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

class Network:
    """The Network class provides functions to create a network object."""
    def __init__(self, model_string = None, n_agents = None, agents = None, neighbour_list = None,  weight_list = None, direction = None):
        """
        Initialize the Network object.
        
         Args:
            model_string (str): The model string.
            n_agents (int): The number of agents.
            agents (list): The list of agents.
            neighbour_list (list): The list of neighbours.
            weight_list (list): The list of weights.
            direction (str): The direction of the network."""
        
        if(model_string == 'DeGroot' or model_string == 'Deffuant'):
            if(n_agents):
                self.network = seldoncore.SimpleAgentNetwork(n_agents)
            elif(agents):
                self.network = seldoncore.SimpleAgentNetwork(agents)  
            elif(neighbour_list and weight_list and direction):
                if(direction == 'Incoming' or direction == 'Outgoing'):
                    self.network = seldoncore.SimpleAgentNetwork(neighbour_list, weight_list, direction)   
                else:
                    TypeError("Direction allowed values are 'Incoming' or 'Outgoing'")
            else:
                self.network = seldoncore.SimpleAgentNetwork()


        elif(model_string== "DeffuantVector"):
            if(n_agents):
                self.network = seldoncore.DiscreteVectorAgentNetwork(n_agents)
            elif(agents):
                self.network = seldoncore.DiscreteVectorAgentNetwork(agents)  
            elif(neighbour_list and weight_list and direction):
                if(direction == 'Incoming' or direction == 'Outgoing'):
                    self.network = seldoncore.DiscreteVectorAgentNetwork(neighbour_list, weight_list, direction)   
                else:
                    TypeError("Direction allowed values are 'Incoming' or 'Outgoing'")
            else:
                self.network = seldoncore.DiscreteVectorAgentNetwork()
        
        elif(model_string == "ActivityDriven"):
            if(n_agents):
                self.network = seldoncore.ActivityDrivenAgentNetwork(n_agents)
            elif(agents):
                self.network = seldoncore.ActivityDrivenAgentNetwork(agents)  
            elif(neighbour_list and weight_list and direction):
                if(direction == 'Incoming' or direction == 'Outgoing'):
                    self.network = seldoncore.ActivityDrivenAgentNetwork(neighbour_list, weight_list, direction)   
                else:
                    TypeError("Direction allowed values are 'Incoming' or 'Outgoing'")
            else:
                self.network = seldoncore.ActivityDrivenAgentNetwork()
        
        elif(model_string == "ActivityDrivenInertial" or model_string == "Inertial"):
            if(n_agents):
                self.network = seldoncore.InertialAgentNetwork(n_agents)
            elif(agents):
                self.network = seldoncore.InertialAgentNetwork(agents)  
            elif(neighbour_list and weight_list and direction):
                if(direction == 'Incoming' or direction == 'Outgoing'):
                    self.network = seldoncore.InertialAgentNetwork(neighbour_list, weight_list, direction)   
                else:
                    TypeError("Direction allowed values are 'Incoming' or 'Outgoing'")
            else:
                self.network = seldoncore.InertialAgentNetwork()

        else:
            print("This is a float type network that can't be used for the simulation as it doesn't contain any agents and their data like opinions, etc.")
            if(n_agents):
                self.network = seldoncore.Network(n_agents)
            elif(agents):
                self.network = seldoncore.Network(agents)
            elif(neighbour_list and weight_list and direction):
                if(direction == 'Incoming' or direction == 'Outgoing'):
                    self.network = seldoncore.Network(neighbour_list, weight_list, direction)
                else:
                    TypeError("Direction allowed values are 'Incoming' or 'Outgoing'")
            else:
                self.network = seldoncore.Network()

    
    @property
    def n_agents(self):
        """The number of agents in the network.

        Returns:
            int: The number of agents in the network."""
        return self.network.n_agents()
    
    @property
    def n_edges(self):
        """The number of edges in the network.

        Args:
            agent_idx (int): The index of the agent. If not provided, the number of edges in the network is returned.
        
        Returns:
            int: The number of edges in the network."""
        
        return self.network.n_edges(agent_idx = None)
    


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
        