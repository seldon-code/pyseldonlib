"""The Network module contains the Network class and functions to generate networks and save them to files."""

from bindings import seldoncore
import logging
import bindings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Network:
    def __init__(
        self,
        model_string: str = None,
        n_agents: int = None,
        agents: int = None,
        neighbour_list: list[int] = [],
        weight_list: list[float] = [],
        direction: str = "Incoming",
    ):
        """
        Initialize the Network object.

        There are six types of networks that can be created using this class:
        1. DeGroot: The DeGroot network. (opinions only)
        2. Deffuant: The Deffuant network. (opinions only)
        3. DeffuantVector: The DeffuantVector network. (binary vector opinions)
        4. ActivityDriven: The ActivityDriven network. (opinions, activity, and reluctance)
        5. Inertial: The Inertial network. (opinions, activity, reluctance, and velocity)
        6. Float: The float network. (just nodes and edges)

        Also, it can be instantiated in different ways:
        - model_string (str): Is compulsory, the model string.
        1. By providing the number of agents.
        2. By providing the list of agents.
        3. By providing the list of neighbours, weights, and direction.
        4. Default constructor.

        Parameters
        -----------
        model_string : str, default=None
          The model string.

        n_agents : int, default=None
          The number of agents.

        agents : list[int], default=None
          The list of agents.

        neighbour_list : list[int], default=[]
          The list of neighbours.

        weight_list : list[float], default=[]
          The list of weights.

        direction : str
          The direction of the network.

        """

        if model_string == "DeGroot" or model_string == "Deffuant":
            if n_agents:
                self.network = seldoncore.SimpleAgentNetwork(n_agents)
            elif agents:
                self.network = seldoncore.SimpleAgentNetwork(agents)
            elif neighbour_list and weight_list and direction:
                if direction == "Incoming" or direction == "Outgoing":
                    self.network = seldoncore.SimpleAgentNetwork(
                        neighbour_list, weight_list, direction
                    )
                else:
                    TypeError("Direction allowed values are 'Incoming' or 'Outgoing'")
            else:
                self.network = seldoncore.SimpleAgentNetwork()

        elif model_string == "DeffuantVector":
            if n_agents:
                self.network = seldoncore.DiscreteVectorAgentNetwork(n_agents)
            elif agents:
                self.network = seldoncore.DiscreteVectorAgentNetwork(agents)
            elif neighbour_list and weight_list and direction:
                if direction == "Incoming" or direction == "Outgoing":
                    self.network = seldoncore.DiscreteVectorAgentNetwork(
                        neighbour_list, weight_list, direction
                    )
                else:
                    TypeError("Direction allowed values are 'Incoming' or 'Outgoing'")
            else:
                self.network = seldoncore.DiscreteVectorAgentNetwork()

        elif model_string == "ActivityDriven":
            if n_agents:
                self.network = seldoncore.ActivityDrivenAgentNetwork(n_agents)
            elif agents:
                self.network = seldoncore.ActivityDrivenAgentNetwork(agents)
            elif neighbour_list and weight_list and direction:
                if direction == "Incoming" or direction == "Outgoing":
                    self.network = seldoncore.ActivityDrivenAgentNetwork(
                        neighbour_list, weight_list, direction
                    )
                else:
                    TypeError("Direction allowed values are 'Incoming' or 'Outgoing'")
            else:
                self.network = seldoncore.ActivityDrivenAgentNetwork()

        elif model_string == "ActivityDrivenInertial" or model_string == "Inertial":
            if n_agents:
                self.network = seldoncore.InertialAgentNetwork(n_agents)
            elif agents:
                self.network = seldoncore.InertialAgentNetwork(agents)
            elif neighbour_list and weight_list and direction:
                if direction == "Incoming" or direction == "Outgoing":
                    self.network = seldoncore.InertialAgentNetwork(
                        neighbour_list, weight_list, direction
                    )
                else:
                    TypeError("Direction allowed values are 'Incoming' or 'Outgoing'")
            else:
                self.network = seldoncore.InertialAgentNetwork()

        else:
            logger.warning(
                "This is a float type network that can't be used for the simulation as it doesn't contain any agents and their data like opinions, etc."
            )
            if n_agents:
                self.network = seldoncore.Network(n_agents)
            elif agents:
                self.network = seldoncore.Network(agents)
            elif neighbour_list and weight_list and direction:
                if direction == "Incoming" or direction == "Outgoing":
                    self.network = seldoncore.Network(
                        neighbour_list, weight_list, direction
                    )
                else:
                    TypeError("Direction allowed values are 'Incoming' or 'Outgoing'")
            else:
                self.network = seldoncore.Network()

    @property
    def n_agents(self):
        """The number of nodes/agents in the network."""
        return self.network.n_agents()

    def n_edges(self, agent_idx: int):
        """The number of edges going out/coming in at `agent_idx` in the network.

        Parameters
        -----------
        agent_idx : int
          The index of the agent. If not provided, the total number of edges in the network is returned.

        """
        return self.network.n_edges(agent_idx)

    @property
    def get_direction(self):
        """The direction of the network."""
        return self.network.direction()

    @property
    def strongly_connected_components(self):
        """The strongly connected components of the network.

        Returns
            list: The strongly connected components of the network.
        """
        return self.network.strongly_connected_components()

    def get_neighbours(self, index: int):
        """The neighbours of the node/agent in the network.

        Parameters
        -----------
        index : int
          The index of the agent.

        """
        return self.network.get_neighbours(index)

    def get_weights(self, index: int):
        """The weights of the agent.

        Parameters
        -----------
        index : int
          The index of the agent.

        """

        return self.network.get_weights(index)

    def set_weights(self, agent_idx: int, weights: list):
        """Set the weights of the agent.

        Parameters
        -----------
        index : int
          The index of the agent.
        weights : list[float]
          The weights of the agent.

        """

        return self.network.set_weights(agent_idx, weights)

    def set_neighbours_and_weights(
        self, agent_idx: int, buffer_neighbours: list[int], buffer_weights: list[float]
    ):
        """Sets the neighbour indices and weights at agent_idx

        Parameters
        -----------
        agent_idx : int
          The index of the agent.
        buffer_neighbours : list[int]
          The list of neighbours.
        buffer_weights : list[float]
          The list of weights.

        """
        return self.network.set_neighbours_and_weights(
            agent_idx, buffer_neighbours, buffer_weights
        )

    def set_neighbours_and_weights(
        self, agent_idx: int, buffer_neighbours: list[int], weight: float
    ):
        """Sets the neighbour indices and sets the weight to a constant value at agent_idx in the network.

        Parameters
        -----------
        agent_idx : int
          The index of the agent.
        buffer_neighbours : list[int]
          The list of neighbours.
        weight : float
          The weight of the agent.

        """
        return self.network.set_neighbours_and_weights(
            agent_idx, buffer_neighbours, weight
        )

    def push_back_neighbour_and_weight(
        self, agent_idx_i: int, agent_idx_j: int, weight: float
    ):
        """Adds an edge between agent_idx_i and agent_idx_j with weight w

        Parameters
        ------------
        agent_idx_i : int
          The index of the agent.
        agent_idx_j : int
          The index of the agent.
        weight : float
          The weight of the agent.

        """
        return self.network.push_back_neighbour_and_weight(
            agent_idx_i, agent_idx_j, weight
        )

    @property
    def transpose(self):
        """Transposes the network, without switching the direction flag (expensive).

        Example:
        --------
          N(inc) -> N(inc)^T

        """
        return self.network.transpose()

    @property
    def toggle_incoming_outgoing(self):
        """Switches the direction flag *without* transposing the network (expensive)

        Example:
        --------
          N(inc) -> N(out)

        """

        return self.network.toggle_incoming_outgoing()

    @property
    def switch_direction_flag(self):
        """Only switches the direction flag. This effectively transposes the network and, simultaneously, changes its representation.

        Example:
        --------
          N(inc) -> N^T(out)

        """
        return self.network.switch_direction_flag()

    @property
    def remove_double_counting(self):
        """Sorts the neighbours by index and removes doubly counted edges by summing the weights of the corresponding edges."""
        return self.network.remove_double_counting()

    @property
    def clear(self):
        """Clears the network."""

        return self.network.clear()


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