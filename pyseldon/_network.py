from bindings import seldoncore


class Network:
    """The Network class provides functions to create a network object."""

    def __init__(
        self,
        model_string: str = None,
        n_agents: int = None,
        agents: int = None,
        neighbour_list: list[int] = [],
        weight_list: list[float] = [],
        direction: str = "Incoming",
    ):
        """Initialize the Network object.

        Note: This class is used to create a network object that can be used to run the simulation by writing it out to a file and then using it.

        There are six types of networks that can be created using this class:
        1. DeGroot: The DeGroot network. (opinions only)
        2. Deffuant: The Deffuant network. (opinions only)
        3. DeffuantVector: The DeffuantVector network. (binary vector opinions)
        4. ActivityDriven: The ActivityDriven network. (opinions, activity, and reluctance)
        5. Inertial: The Inertial network. (opinions, activity, reluctance, and velocity)
        6. Float: The float network. (just nodes and edges)

        Also it can be instantiated in different ways:
        model_string(str): Is Compulsory, The model string.
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
            print(
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
