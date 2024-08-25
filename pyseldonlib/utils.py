"""This module contains functions to generate networks and save them to files."""

from bindings import seldoncore
import logging
import bindings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_n_connections(model_string, n_agents:int, n_connections:int, self_interaction:bool= False, rng_seed:int=None):
    """
    Generate n_connections Network for n_agents.

    Parameters
    -----------
    model_string : str
      The model string. Allowed values are "DeGroot", "Deffuant", "DeffuantVector", "ActivityDriven", "Inertial", "Float".
    n_agents : int
      The number of agents.
    n_connections : int
      The number of connections.
    self_interaction : bool, default=False
      If True, self iteraction is allowed.
    rng_seed : int, default=None
      The seed for the random number generator. If not provided, a random seed is picked.

    Returns
    -----------
    Network: The network.
    """
    if model_string == "DeGroot" or model_string == "Deffuant":
        if rng_seed is not None:
            return seldoncore.generate_n_connections_simple_agent(n_agents= n_agents,n_connections = n_connections,self_interaction = self_interaction,seed = rng_seed)
        else:
            return seldoncore.generate_n_connections_simple_agent(n_agents= n_agents,n_connections = n_connections,self_interaction = self_interaction)
    elif model_string == "DeffuantVector":
        if rng_seed is not None:
            return seldoncore.generate_n_connections_discrete_vector_agent(n_agents= n_agents,n_connections = n_connections,self_interaction = self_interaction,seed = rng_seed)
        else:
            return seldoncore.generate_n_connections_discrete_vector_agent(n_agents= n_agents,n_connections = n_connections,self_interaction = self_interaction)
    elif model_string == "ActivityDriven":
        if rng_seed is not None:
            return seldoncore.generate_n_connections_activity_agent(n_agents= n_agents,n_connections = n_connections,self_interaction = self_interaction,seed = rng_seed)
        else:
            return seldoncore.generate_n_connections_activity_agent(n_agents= n_agents,n_connections = n_connections,self_interaction = self_interaction)
    elif model_string == "Inertial":
        if rng_seed is not None:
            return seldoncore.generate_n_connections_inertial_agent(n_agents= n_agents,n_connections = n_connections,self_interaction = self_interaction,seed = rng_seed)
        else:
            return seldoncore.generate_n_connections_inertial_agent(n_agents= n_agents,n_connections = n_connections,self_interaction = self_interaction)
    elif model_string == "Float":
        if rng_seed is not None:
            return seldoncore.generate_n_connections_(n_agents= n_agents,n_connections = n_connections,self_interaction = self_interaction,seed = rng_seed)
        else:
            return seldoncore.generate_n_connections_(n_agents= n_agents,n_connections = n_connections,self_interaction = self_interaction)        

def generate_fully_connected(model_string, n_agents:int, weight:float= None, rng_seed:int=None):
    """
    Generate a fully connected Network for n_agents.

    Parameters
    -----------
    model_string : str
      The model string. Allowed values are "DeGroot", "Deffuant", "DeffuantVector", "ActivityDriven", "Inertial", "Float".
    n_agents : int
      The number of agents.
    weight : float
      The weight of the agent.
    rng_seed : int, default=None
      The seed for the random number generator. If not provided, a random seed is picked.

    Returns
    -----------
    Network: The fully connected network.
    """
    if model_string == "DeGroot" or model_string == "Deffuant":
        return seldoncore.generate_fully_connected_simple_agent(n_agents= n_agents,weight = weight,seed = rng_seed)
    elif model_string == "DeffuantVector":
        return seldoncore.generate_fully_connected_discrete_vector_agent(n_agents= n_agents,weight = weight,seed = rng_seed)
    elif model_string == "ActivityDriven":
        return seldoncore.generate_fully_connected_activity_agent(n_agents= n_agents,weight = weight,seed = rng_seed)
    elif model_string == "Inertial":
        return seldoncore.generate_fully_connected_inertial_agent(n_agents= n_agents,weight = weight,seed = rng_seed)
    elif model_string == "Float":
        return seldoncore.generate_fully_connected_(n_agents= n_agents,weight = weight,seed = rng_seed)

def generate_square_lattice(model_string, n_edge: int, weight: float):
    """
    Generate a square lattice Network.

    Parameters
    -----------
    model_string : str
      The model string. Allowed values are "DeGroot", "Deffuant", "DeffuantVector", "ActivityDriven", "Inertial", "Float".
    n_edge : int
      The number of edges.
    weight : float
      The weight of the agent.

    Returns
    -----------
    Network: The square lattice network.
    """
    if model_string == "DeGroot" or model_string == "Deffuant":
        return seldoncore.generate_square_lattice_simple_agent(n_edge= n_edge,weight = weight)
    elif model_string == "DeffuantVector":
        return seldoncore.generate_square_lattice_discrete_vector_agent(n_edge= n_edge,weight = weight)
    elif model_string == "ActivityDriven":
        return seldoncore.generate_square_lattice_activity_agent(n_edge= n_edge,weight = weight)
    elif model_string == "Inertial":
        return seldoncore.generate_square_lattice_inertial_agent(n_edge= n_edge,weight = weight)
    elif model_string == "Float":
        return seldoncore.generate_square_lattice_(n_edge= n_edge,weight = weight)
    
def generate_network_from_file(model_string,file_path):
    """
    Generate a Network from a file.

    Parameters
    -----------
    model_string : str
      The model string. Allowed values are "DeGroot", "Deffuant", "DeffuantVector", "ActivityDriven", "Inertial", "Float".
    file_path : str
        The file path.

    Returns
    -----------
    Network: The network.
    """
    if model_string == "DeGroot" or model_string == "Deffuant":
        return seldoncore.generate_network_from_file_simple_agent(file_path)
    elif model_string == "DeffuantVector":
        return seldoncore.generate_network_from_file_discrete_vector_agent(file_path)
    elif model_string == "ActivityDriven":
        return seldoncore.generate_network_from_file_activity_agent(file_path)
    elif model_string == "Inertial":
        return seldoncore.generate_network_from_file_inertial_agent(file_path)
    elif model_string == "Float":
        return seldoncore.generate_network_from_file_(file_path)
    
def network_to_file(network, file_path):
    """
    Save the network to a file along with weights.
    
    Parameters
    -----------
    network : Network
      The network.
      
    file_path : str
      The file path.
    """
    if isinstance(network, bindings.seldoncore.SimpleAgentNetwork):
        return network.network_to_file_simple_agent(network, file_path)
    elif isinstance(network, bindings.seldoncore.DiscreteVectorAgentNetwork):
        return network.network_to_file_discrete_vector_agent(network, file_path)
    elif isinstance(network, bindings.seldoncore.ActivityDrivenAgentNetwork):
        return network.network_to_file_activity_agent(network, file_path)
    elif isinstance(network, bindings.seldoncore.InertialAgentNetwork):
        return network.network_to_file_inertial_agent(network, file_path)
    elif isinstance(network, bindings.seldoncore.Network):
        return network.network_to_file(network, file_path)
    else:
        raise TypeError("Invalid network type")
    
def network_to_dot_file(network, file_path):
    """
    Save the network to a dot file. This is a graph description language file.
    
    Parameters
    -----------
    network : Network
      The network.
      
    file_path : str
      The file path.
    """
    if isinstance(network, bindings.seldoncore.SimpleAgentNetwork):
        return network.network_to_dot_file_simple_agent(network, file_path)
    elif isinstance(network, bindings.seldoncore.DiscreteVectorAgentNetwork):
        return network.network_to_dot_file_discrete_vector_agent(network, file_path)
    elif isinstance(network, bindings.seldoncore.ActivityDrivenAgentNetwork):
        return network.network_to_dot_file_activity_agent(network, file_path)
    elif isinstance(network, bindings.seldoncore.InertialAgentNetwork):
        return network.network_to_dot_file_inertial_agent(network, file_path)
    elif isinstance(network, bindings.seldoncore.Network):
        return network.network_to_dot_file(network, file_path)
    else:
        raise TypeError("Invalid network type")
    
def agents_from_file(model_string, file_path):
    """
    Load agents from a file.
    
    Parameters
    -----------
    model_string : str
      The model string. Allowed values are "DeGroot", "Deffuant", "DeffuantVector", "ActivityDriven", "Inertial", "Float".
    file_path : str
      The file path.
      
    Returns
    -----------
    list: The list of agents.
    """

    if model_string == "DeGroot" or model_string == "Deffuant":
        return seldoncore.agents_from_file_simple_agent(file_path)
    elif model_string == "DeffuantVector":
        return seldoncore.agents_from_file_discrete_vector_agent(file_path)
    elif model_string == "ActivityDriven":
        return seldoncore.agents_from_file_activity_agent(file_path)
    elif model_string == "Inertial":
        return seldoncore.agents_from_file_inertial_agent(file_path)
    elif model_string == "Float":
        return seldoncore.agents_from_file(file_path)
    else:
        raise TypeError("Invalid model string")
    
def agents_to_file(network, file_path):
    """
    Save agents to a file.
    
    Parameters
    -----------
    network : Network
      The network.
      
    file_path : str
      The file path.
    """
    if isinstance(network, bindings.seldoncore.SimpleAgentNetwork):
        return network.agents_to_file_simple_agent(network, file_path)
    elif isinstance(network, bindings.seldoncore.DiscreteVectorAgentNetwork):
        return network.agents_to_file_discrete_vector_agent(network, file_path)
    elif isinstance(network, bindings.seldoncore.ActivityDrivenAgentNetwork):
        return network.agents_to_file_activity_agent(network, file_path)
    elif isinstance(network, bindings.seldoncore.InertialAgentNetwork):
        return network.agents_to_file_inertial_agent(network, file_path)
    elif isinstance(network, bindings.seldoncore.Network):
        return network.agents_to_file(network, file_path)
    else:
        raise TypeError("Invalid network type")