"""This module provides the main interface to the pyseldon package."""

from pyseldon import seldoncore
from typing import List, Optional, Union, LiteralString
# from .simulation import run_simulation_from_config_file, run_simulation_from_options

__all__ = ["run_simulation_from_config_file", "run_simulation_from_options"]

# """This module provides functions to run the simulation using the configuration file or the simulation options object."""


def run_simulation_from_config_file(
    config_file_path: str,
    agent_file_path: Optional[str] = None,
    network_file_path: Optional[str] = None,
    output_dir_path: Optional[str] = None,
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

    def __init__(
        self,
        model_string=None,
        n_agents=None,
        agents=None,
        neighbour_list=None,
        weight_list=None,
        direction=None,
    ):
        """
        Initialize the Network object.

         Args:
            model_string (str): The model string.
            n_agents (int): The number of agents.
            agents (list): The list of agents.
            neighbour_list (list): The list of neighbours.
            weight_list (list): The list of weights.
            direction (str): The direction of the network."""

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

            self._agent_type = "SimpleAgent"

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

            self._agent_type = "DiscreteVectorAgent"

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
            self._agent_type = "ActivityDrivenAgent"

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
            self._agent_type = "InertialAgent"

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
            self._agent_type = "float"

    @property
    def n_agents(self):
        """The number of nodes/agents in the network.

        Returns:
            int: The number of nodes/agents in the network."""
        return self.network.n_agents()

    @property
    def n_edges(self, agent_idx=None):
        """The number of edges going out/coming in at `agent_idx` in the network.

        Args:
            agent_idx (int): The index of the agent. If not provided, the total number of edges in the network is returned.

        Returns:
            int: The number of edges in the network."""

        return self.network.n_edges(agent_idx)

    @property
    def get_direction(self):
        """The direction of the network.

        Returns:
            str: The direction of the network."""
        return self.network.direction()

    @property
    def strongly_connected_components(self):
        """The strongly connected components of the network.

        Returns:
            list: The strongly connected components of the network."""
        return self.network.strongly_connected_components()

    def get_neighbours(self, index):
        """The neighbours of the node/agent in the network.

        Args:
            index (int): The index of the agent.

        Returns:
            list: The neighbours of the agent."""
        return self.network.get_neighbours(index)

    def get_weights(self, index):
        """The weights of the agent.

        Args:
            index (int): The index of the agent.

        Returns:
            list: The weights of the agent."""

        return self.network.get_weights(index)

    def set_weights(self, agent_idx, weights):
        """Set the weights of the agent.

        Args:
            index (int): The index of the agent.
            weights (list): The weights of the agent.

        Returns:
            None"""

        return self.network.set_weights(agent_idx, weights)

    def set_neighbours_and_weights(self, agent_idx, buffer_neighbours, buffer_weights):
        """Sets the neighbour indices and weights at agent_idx

        Args:
            agent_idx (int): The index of the agent.
            buffer_neighbours (list): The list of neighbours.
            buffer_weights (list): The list of weights.

        Returns:
            None"""
        return self.network.set_neighbours_and_weights(
            agent_idx, buffer_neighbours, buffer_weights
        )

    def set_neighbours_and_weights(self, agent_idx, buffer_neighbours, weight):
        """Sets the neighbour indices and sets the weight to a constant value at agent_idx in the network.

        Args:
            agent_idx (int): The index of the agent.
            buffer_neighbours (list): The list of neighbours.
            weight (float): The weight of the agent.

        Returns:
            None"""
        return self.network.set_neighbours_and_weights(
            agent_idx, buffer_neighbours, weight
        )

    def push_back_neighbour_and_weight(self, agent_idx_i, agent_idx_j, weight):
        """Adds an edge between agent_idx_i and agent_idx_j with weight w

        Args:
            agent_idx_i (int): The index of the agent.
            agent_idx_j (int): The index of the agent.
            weight (float): The weight of the agent.

        Returns:
            None"""
        return self.network.push_back_neighbour_and_weight(
            agent_idx_i, agent_idx_j, weight
        )

    def transpose(self):
        """Transposes the network, without switching the direction flag (expensive).

        Example: N(inc) -> N(inc)^T

        Returns:
            None"""
        return self.network.transpose()

    def toggle_incoming_outgoing(self):
        """Switches the direction flag *without* transposing the network (expensive)

        Example: N(inc) -> N(out)

        Returns:
            None"""

        return self.network.toggle_incoming_outgoing()

    def switch_direction_flag(self):
        """Only switches the direction flag. This effectively transposes the network and, simultaneously, changes its representation.

        Example: N(inc) -> N^T(out)

        Returns:
            None"""
        return self.network.switch_direction_flag()

    def remove_double_counting(self):
        """Sorts the neighbours by index and removes doubly counted edges by summing the weights of the corresponding edges.

        Returns:
            None"""
        return self.network.remove_double_counting()

    def clear(self):
        """Clears the network.

        Returns:
            None"""
        return self.network.clear()

    def get_agents_data(self, index=None):
        """Access the network's agents data.

        Args:
            index (int, Optional): The index of the agent. If not provided, the data of all agents is returned.


        Returns:
            for DeGroot and Deffuant:
                list: The opinion of the agent at the given index.
            for DeffuantVector:
                list: The opinion of the agent at the given index.
            for ActivityDriven:
                list: opinion, activity, and reluctance of the agent at the given index.
            for Inertial:
                list: opinion, activity, reluctance, and velocity of the agent at the given index."""

        if self._agent_type == "SimpleAgent":
            if index is None:
                return [
                    self.network.agent[i].data.opinion for i in range(self.n_agents)
                ]
            return self.network.agent[index].data.opinion

        elif self._agent_type == "DiscreteVectorAgent":
            if index is None:
                return [
                    self.network.agent[i].data.opinion for i in range(self.n_agents)
                ]
            return self.network.agent[index].data.opinion

        elif self._agent_type == "ActivityDrivenAgent":
            if index is None:
                return [
                    (
                        self.network.agent[i].data.opinion,
                        self.network.agent[i].data.activity,
                        self.network.agent[i].data.reluctance,
                    )
                    for i in range(self.n_agents)
                ]
            return (
                self.network.agent[index].data.opinion,
                self.network.agent[index].data.activity,
                self.network.agent[index].data.reluctance,
            )

        elif self._agent_type == "InertialAgent":
            if index is None:
                return [
                    (
                        self.network.agent[i].data.opinion,
                        self.network.agent[i].data.activity,
                        self.network.agent[i].data.reluctance,
                        self.network.agent[i].data.velocity,
                    )
                    for i in range(self.n_agents)
                ]
            return (
                self.network.agent[index].data.opinion,
                self.network.agent[index].data.activity,
                self.network.agent[index].data.reluctance,
                self.network.agent[index].data.velocity,
            )

        else:
            print(
                "This is a float type network that can't be used for the simulation as it doesn't contain any agents and their data like opinions, etc."
            )
            return None

    def set_agents_data(
        self, index, opinion, activity=None, reluctance=None, velocity=None
    ):
        """Set the agent's data.

        Args:
            index (int): The index of the agent.
            for DeGroot, Deffuant, DeffuantVector, ActivityDriven, and Inertial:
                opinion (float): The opinion of the agent.
            for ActivityDriven and Inertial:
                activity (float, Optional): The activity of the agent.
            for ActivityDriven and Inertial:
                reluctance (float, Optional): The reluctance of the agent.
            for Inertial:
                velocity (float, Optional): The velocity of the agent.

            if any is not provided, the data is not changed.

        Returns:
            None"""

        if self._agent_type == "SimpleAgent":
            self.network.agent[index].data.opinion = opinion

        elif self._agent_type == "DiscreteVectorAgent":
            self.network.agent[index].data.opinion = opinion

        elif self._agent_type == "ActivityDrivenAgent":
            if opinion is not None:
                self.network.agent[index].data.opinion = opinion
            if activity is not None:
                self.network.agent[index].data.activity = activity
            if reluctance is not None:
                self.network.agent[index].data.reluctance = reluctance

        elif self._agent_type == "InertialAgent":
            if opinion is not None:
                self.network.agent[index].data.opinion = opinion
            if activity is not None:
                self.network.agent[index].data.activity = activity
            if reluctance is not None:
                self.network.agent[index].data.reluctance = reluctance
            if velocity is not None:
                self.network.agent[index].data.velocity = velocity

        else:
            print(
                "This is a float type network that can't be used for the simulation as it doesn't contain any agents and their data like opinions, etc."
            )
            return None


class Simulation:
    """The Simulation class provides functions to run the simulation using the simulation options object, agent file, and network file."""

    def __init__(
        self, model_string="DeGroot", agent_file_path=None, network_file_path=None
    ):
        """
        Initialize the Simulation object.

        Args:
            model_string (str): The model string.
            agent_file_path (str, optional): The path to the agent file.
            network_file_path (str, optional): The path to the network file.
        """
        if model_string == "DeGroot":
            self.simulation = seldoncore.SimulationDeGroot()

        elif model_string == "Deffuant":
            self.simulation = seldoncore.SimulationDeffuant()

        elif model_string == "DeffuantVector":
            self.simulation = seldoncore.SimulationDeffuantVector()

        elif model_string == "ActivityDriven":
            self.simulation = seldoncore.SimulationActivityDriven()

        elif model_string == "ActivityDrivenInertial":
            self.simulation = seldoncore.SimulationInertial()

        else:
            TypeError("Model not found")
