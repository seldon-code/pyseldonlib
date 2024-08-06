"""This module provides the main interface to the pyseldon package."""

# from bindings import seldoncore
from typing import Optional, Union
# from .simulation import run_simulation_from_config_file, run_simulation_from_options

# __all__ = ["run_simulation_from_config_file", "run_simulation_from_options"]

# """This module provides functions to run the simulation using the configuration file or the simulation options object."""

from .DeGrootModel import Other_Settings, DeGrootModel
from .DeffuantModel import DeffuantModel
from .DeffuantVectorModel import Deffuant_Vector_Model
from .InertialModel import InertialModel
from .ActivityDrivenModel import Activity_Driven_Model

# def run_simulation_from_config_file(
#     config_file_path: str,
#     agent_file_path: Optional[str] = None,
#     network_file_path: Optional[str] = None,
#     output_dir_path: Optional[str] = None,
# ):
#     """
#     Run the simulation using the configuration(toml) file.

#     Args:
#         config_file_path (str): The path to the configuration(toml) file.
#         agent_file_path (str, optional): The path to the agent file.
#         network_file_path (str, optional): The path to the network file.
#         output_dir_path (str, optional): The path to the output file.
#     """
#     seldoncore.run_simulation(
#         config_file_path=config_file_path,
#         options=None,
#         agent_file_path=agent_file_path,
#         network_file_path=network_file_path,
#         output_dir_path=output_dir_path,
#     )


# def run_simulation_from_options(
#     options: object,
#     agent_file_path: Optional[str] = None,
#     network_file_path: Optional[str] = None,
#     output_dir_path: Optional[str] = None,
# ):
#     """
#     Run the simulation using the simulation options object.

#     Note: The options object must be created using the SimulationOptions class.

#     Args:
#         options (object): The simulation options object.
#         agent_file_path (str, optional): The path to the agent file.
#         network_file_path (str, optional): The path to the network file.
#         output_dir_path (str, optional): The path to the output directory.
#     """
#     seldoncore.run_simulation(
#         options=options.options,
#         config_file_path=None,
#         agent_file_path=agent_file_path,
#         network_file_path=network_file_path,
#         output_dir_path=output_dir_path,
#     )


# class Network:
#     """The Network class provides functions to create a network object."""

#     def __init__(
#         self,
#         model_string: str = None,
#         n_agents: int = None,
#         agents: int = None,
#         neighbour_list: list[int] = None,
#         weight_list: list[float] = None,
#         direction: str = None,
#     ):
#         """
#         Initialize the Network object.

#         Note: This class is used to create a network object that can be used to run the simulation by writing it out to a file and then using it.

#         There are six types of networks that can be created using this class:
#         1. DeGroot: The DeGroot network.
#         2. Deffuant: The Deffuant network.
#         3. DeffuantVector: The DeffuantVector network.
#         4. ActivityDriven: The ActivityDriven network.
#         5. Inertial: The Inertial network.
#         6. Float: The float network.

#         Also it can be instantiated in different ways:
#         model_string(str): Is Compulsory, The model string.
#         1. By providing the number of agents.
#         2. By providing the list of agents.
#         3. By providing the list of neighbours, weights, and direction.
#         4. Default constructor.

#          Args:
#             model_string (str): The model string.
#             n_agents (int): The number of agents.
#             agents (list[int]): The list of agents.
#             neighbour_list (list[int]): The list of neighbours.
#             weight_list (list[float]): The list of weights.
#             direction (str): The direction of the network.
#         """

#         if model_string == "DeGroot" or model_string == "Deffuant":
#             if n_agents:
#                 self.network = seldoncore.SimpleAgentNetwork(n_agents)
#             elif agents:
#                 self.network = seldoncore.SimpleAgentNetwork(agents)
#             elif neighbour_list and weight_list and direction:
#                 if direction == "Incoming" or direction == "Outgoing":
#                     self.network = seldoncore.SimpleAgentNetwork(
#                         neighbour_list, weight_list, direction
#                     )
#                 else:
#                     TypeError("Direction allowed values are 'Incoming' or 'Outgoing'")
#             else:
#                 self.network = seldoncore.SimpleAgentNetwork()

#         elif model_string == "DeffuantVector":
#             if n_agents:
#                 self.network = seldoncore.DiscreteVectorAgentNetwork(n_agents)
#             elif agents:
#                 self.network = seldoncore.DiscreteVectorAgentNetwork(agents)
#             elif neighbour_list and weight_list and direction:
#                 if direction == "Incoming" or direction == "Outgoing":
#                     self.network = seldoncore.DiscreteVectorAgentNetwork(
#                         neighbour_list, weight_list, direction
#                     )
#                 else:
#                     TypeError("Direction allowed values are 'Incoming' or 'Outgoing'")
#             else:
#                 self.network = seldoncore.DiscreteVectorAgentNetwork()


#         elif model_string == "ActivityDriven":
#             if n_agents:
#                 self.network = seldoncore.ActivityDrivenAgentNetwork(n_agents)
#             elif agents:
#                 self.network = seldoncore.ActivityDrivenAgentNetwork(agents)
#             elif neighbour_list and weight_list and direction:
#                 if direction == "Incoming" or direction == "Outgoing":
#                     self.network = seldoncore.ActivityDrivenAgentNetwork(
#                         neighbour_list, weight_list, direction
#                     )
#                 else:
#                     TypeError("Direction allowed values are 'Incoming' or 'Outgoing'")
#             else:
#                 self.network = seldoncore.ActivityDrivenAgentNetwork()

#         elif model_string == "ActivityDrivenInertial" or model_string == "Inertial":
#             if n_agents:
#                 self.network = seldoncore.InertialAgentNetwork(n_agents)
#             elif agents:
#                 self.network = seldoncore.InertialAgentNetwork(agents)
#             elif neighbour_list and weight_list and direction:
#                 if direction == "Incoming" or direction == "Outgoing":
#                     self.network = seldoncore.InertialAgentNetwork(
#                         neighbour_list, weight_list, direction
#                     )
#                 else:
#                     TypeError("Direction allowed values are 'Incoming' or 'Outgoing'")
#             else:
#                 self.network = seldoncore.InertialAgentNetwork()

#         else:
#             print(
#                 "This is a float type network that can't be used for the simulation as it doesn't contain any agents and their data like opinions, etc."
#             )
#             if n_agents:
#                 self.network = seldoncore.Network(n_agents)
#             elif agents:
#                 self.network = seldoncore.Network(agents)
#             elif neighbour_list and weight_list and direction:
#                 if direction == "Incoming" or direction == "Outgoing":
#                     self.network = seldoncore.Network(
#                         neighbour_list, weight_list, direction
#                     )
#                 else:
#                     TypeError("Direction allowed values are 'Incoming' or 'Outgoing'")
#             else:
#                 self.network = seldoncore.Network()

#     @property
#     def n_agents(self):
#         """The number of nodes/agents in the network.

#         Returns:
#             int: The number of nodes/agents in the network.
#         """
#         return self.network.n_agents()

#     def n_edges(self, agent_idx: int = None):
#         """The number of edges going out/coming in at `agent_idx` in the network.

#         Args:
#             agent_idx (int): The index of the agent. If not provided, the total number of edges in the network is returned.

#         Returns:
#             int: The number of edges in the network.
#         """
#         return self.network.n_edges(agent_idx)

#     @property
#     def get_direction(self):
#         """The direction of the network.

#         Returns:
#             str: The direction of the network.
#         """
#         return self.network.direction()

#     @property
#     def strongly_connected_components(self):
#         """The strongly connected components of the network.

#         Returns:
#             list: The strongly connected components of the network.
#         """
#         return self.network.strongly_connected_components()

#     def get_neighbours(self, index: int):
#         """The neighbours of the node/agent in the network.

#         Args:
#             index (int): The index of the agent.

#         Returns:
#             list: The neighbours of the agent.
#         """
#         return self.network.get_neighbours(index)

#     def get_weights(self, index: int):
#         """The weights of the agent.

#         Args:
#             index (int): The index of the agent.

#         Returns:
#             list: The weights of the agent.
#         """

#         return self.network.get_weights(index)

#     def set_weights(self, agent_idx: int, weights: list):
#         """Set the weights of the agent.

#         Args:
#             index (int): The index of the agent.
#             weights (list[float]): The weights of the agent.

#         Returns:
#             None
#         """

#         return self.network.set_weights(agent_idx, weights)

#     def set_neighbours_and_weights(
#         self, agent_idx: int, buffer_neighbours: list[int], buffer_weights: list[float]
#     ):
#         """Sets the neighbour indices and weights at agent_idx

#         Args:
#             agent_idx (int): The index of the agent.
#             buffer_neighbours (list[int]): The list of neighbours.
#             buffer_weights (list[float]): The list of weights.

#         Returns:
#             None
#         """
#         return self.network.set_neighbours_and_weights(
#             agent_idx, buffer_neighbours, buffer_weights
#         )

#     def set_neighbours_and_weights(
#         self, agent_idx: int, buffer_neighbours: list[int], weight: float
#     ):
#         """Sets the neighbour indices and sets the weight to a constant value at agent_idx in the network.

#         Args:
#             agent_idx (int): The index of the agent.
#             buffer_neighbours (list[int]): The list of neighbours.
#             weight (float): The weight of the agent.

#         Returns:
#             None
#         """
#         return self.network.set_neighbours_and_weights(
#             agent_idx, buffer_neighbours, weight
#         )

#     def push_back_neighbour_and_weight(
#         self, agent_idx_i: int, agent_idx_j: int, weight: float
#     ):
#         """Adds an edge between agent_idx_i and agent_idx_j with weight w

#         Args:
#             agent_idx_i (int): The index of the agent.
#             agent_idx_j (int): The index of the agent.
#             weight (float): The weight of the agent.

#         Returns:
#             None
#         """
#         return self.network.push_back_neighbour_and_weight(
#             agent_idx_i, agent_idx_j, weight
#         )

#     @property
#     def transpose(self):
#         """Transposes the network, without switching the direction flag (expensive).

#         Example: N(inc) -> N(inc)^T

#         Returns:
#             None
#         """
#         return self.network.transpose()

#     @property
#     def toggle_incoming_outgoing(self):
#         """Switches the direction flag *without* transposing the network (expensive)

#         Example: N(inc) -> N(out)

#         Returns:
#             None
#         """

#         return self.network.toggle_incoming_outgoing()

#     @property
#     def switch_direction_flag(self):
#         """Only switches the direction flag. This effectively transposes the network and, simultaneously, changes its representation.

#         Example: N(inc) -> N^T(out)

#         Returns:
#             None
#         """
#         return self.network.switch_direction_flag()

#     @property
#     def remove_double_counting(self):
#         """Sorts the neighbours by index and removes doubly counted edges by summing the weights of the corresponding edges.

#         Returns:
#             None
#         """
#         return self.network.remove_double_counting()

#     @property
#     def clear(self):
#         """Clears the network.

#         Returns:
#             None
#         """
#         return self.network.clear()

#     def get_agents_data(self, index:int=None):
#         """Access the network's agents data.

#         Args:
#             index (int, optional): The index of the agent. If not provided, the data of all agents is returned.


#         Returns:
#             DeGroot and Deffuant: (list) The opinion of the agent at the given index.
#             for DeffuantVector: (list) The opinion of the agent at the given index.
#             for ActivityDriven: (list) opinion, activity, and reluctance of the agent at the given index.
#             for Inertial: (list) opinion, activity, reluctance, and velocity of the agent at the given index.
#         """
#         def get_agent_data(agent):
#             result = []

#             opinion = agent.data.opinion if hasattr(agent.data, 'opinion') else None
#             activity = agent.data.activity if hasattr(agent.data, 'activity') else None
#             reluctance = agent.data.reluctance if hasattr(agent.data, 'reluctance') else None
#             velocity = agent.data.velocity if hasattr(agent.data, 'velocity') else None
#             if opinion is not None:
#                     result.append(opinion)
#             if activity is not None:
#                     result.append(activity)
#             if reluctance is not None:
#                     result.append(reluctance)
#             if velocity is not None:
#                     result.append(velocity)
#             return result
            
#         if index is None:
#             return [get_agent_data(self.network.agent[i]) for i in range(self.n_agents)]
#         else:
#             return get_agent_data(self.network.agent[index])


#     def set_agents_data(
#         self,
#         index: int,
#         opinion: float = None,
#         activity: float = None,
#         reluctance: float = None,
#         velocity=None,
#     ):
#         """Set the agent's data.

#         Args:
#             index (int): The index of the agent.
#             for DeGroot, Deffuant, DeffuantVector, ActivityDriven, and Inertial:
#                 opinion (float): The opinion of the agent.
#             for ActivityDriven and Inertial:
#                 activity (float, optional): The activity of the agent.
#             for ActivityDriven and Inertial:
#                 reluctance (float, optional): The reluctance of the agent.
#             for Inertial:
#                 velocity (float, optional): The velocity of the agent.

#             if any is not provided, the data is not changed.

#         Note: The terms like opinion, activity, reluctance, and velocity are discussed in the user guide.

#         Returns:
#             None
#         """

#         if self.network._agent_type == "SimpleAgent":
#             if opinion is not None:
#                 self.network.agent[index].data.opinion = opinion

#         elif self.network._agent_type == "DiscreteVectorAgent":
#             if opinion is not None:
#                 self.network.agent[index].data.opinion = opinion

#         elif self.network._agent_type == "ActivityDrivenAgent":
#             if opinion is not None:
#                 self.network.agent[index].data.opinion = opinion
#             if activity is not None:
#                 self.network.agent[index].data.activity = activity
#             if reluctance is not None:
#                 self.network.agent[index].data.reluctance = reluctance

#         elif self.network.network._agent_type == "InertialAgent":
#             if opinion is not None:
#                 self.network.agent[index].data.opinion = opinion
#             if activity is not None:
#                 self.network.agent[index].data.activity = activity
#             if reluctance is not None:
#                 self.network.agent[index].data.reluctance = reluctance
#             if velocity is not None:
#                 self.network.agent[index].data.velocity = velocity

#         else:
#             print(
#                 "This is a float type network that can't be used for the simulation as it doesn't contain any agents and their data like opinions, etc."
#             )
#             return None

#     def __getattr__(self, name):
#         return getattr(self.network, name)

#     def __setattr__(self, name, value):
#         if name == "network":
#             super().__setattr__(name, value)
#         else:
#             setattr(self.network, name, value)

#     def __delattr__(self, name):
#         delattr(self.network, name)


# class Simulation:
#     """The Simulation class provides functions to run the simulation using the simulation options object, agent file, and network file."""

#     def __init__(
#         self,
#         options: object = None,
#         agent_file_path: str = None,
#         network_file_path: str = None,
#     ):
#         """
#         Initialize the Simulation object.
#         This is a more advanced way to run the simulation using the simulation options object, agent file, and network file.
#         Though the simulation can be run using the run_simulation_from_options function, this class provides more control over the simulation.
#         As you have access to the network when using this object, you can modify the network and agent data at any point in the simulation.

#         Args:
#             model_string (str): The model string.
#             options (object): The simulation options object.
#             agent_file_path (str, optional): The path to the agent file.
#             network_file_path (str, optional): The path to the network file.

#         Returns:
#             The simulation object.
#         """
#         if options is None:
#             TypeError("Options object is required to run the simulation.")

#         if options.options.model_string == "DeGroot":
#             self.simulation = seldoncore.SimulationSimpleAgent(
#                 options.options, agent_file_path, network_file_path
#             )

#         elif options.options.model_string == "Deffuant":
#             self.simulation = seldoncore.SimulationSimpleAgent(
#                 options.options, agent_file_path, network_file_path
#             )

#         elif options.options.model_string == "DeffuantVector":
#             self.simulation = seldoncore.SimulationDiscreteVector(
#                 options.options, agent_file_path, network_file_path
#             )

#         elif options.options.model_string == "ActivityDriven":
#             self.simulation = seldoncore.SimulationActivityDriven(
#                 options.options, agent_file_path, network_file_path
#             )

#         elif options.options.model_string == "ActivityDrivenInertial":
#             self.simulation = seldoncore.SimulationInertial(
#                 options.options, agent_file_path, network_file_path
#             )

#         else:
#             TypeError(
#                 "Model not found, the allowed models are DeGroot, Deffuant, DeffuantVector, ActivityDriven, and ActivityDrivenInertial"
#             )

#     def run(self, output_dir_path: str = None):
#         """
#         Run the simulation.

#         Args:
#             output_dir_path (str, optional): The path to the output directory. If not provided, the output is saved in the current directory `./output`.

#         Returns:
#             None
#         """

#         self.simulation.run(output_dir_path or "./output")

#     @property
#     def network(self):
#         """Access the network object used in the simulation.

#         Returns:
#             object: The network class object.
#         """
#         return self.network


# class OutputSettings:
#     def __init__(
#         self,
#         n_output_agents: Optional[int] = None,
#         n_output_network: Optional[int] = None,
#         print_progress: bool = False,
#         output_initial: bool = True,
#         start_output: int = 1,
#         start_numbering_from: int = 0,
#     ):
#         """Initialize the OutputSettings object.
#         Which is used to set the output_settings for the simulation. Usually goes into the simulation options object. Which is then used to run the simulation.

#         Note: Another way to just pass in all the settings and make the simulations object is using the `parse_config_file` function which directly reads the toml file and creates the simulation object. Toml file specifications are discussed in the user guide. This object can still be a good way to set the output settings as it gives more control over the settings.

#         Args:
#             n_output_agents (int, optional):  Write out the agents every n iterations,by default is None.
#             n_output_network (int, optional): Write out the network every n iterations.by default is None.
#             print_progress (bool): Print the progress of the simulation, by default is False.
#             output_initial (bool): Output initial opinions and network, by default always outputs.
#             start_output (int): Start printing opinion and/or network files from this iteration number, by default is 1.
#             start_numbering_from (int):  The initial step number, before the simulation runs, is this value. The first step would be (1+start_numbering_from). By default, 0

#         Returns:
#             The OutputSettings object.
#         """
#         self.settings = seldoncore.OutputSettings()
#         self.settings.n_output_agents = n_output_agents
#         self.settings.n_output_network = n_output_network
#         self.settings.print_progress = print_progress
#         self.settings.output_initial = output_initial
#         self.settings.start_output = start_output
#         self.settings.start_numbering_from = start_numbering_from

#     def __getattr__(self, name):
#         return getattr(self.settings, name)

#     def __setattr__(self, name, value):
#         if name == "settings":
#             super().__setattr__(name, value)
#         else:
#             setattr(self.settings, name, value)

#     def __delattr__(self, name):
#         delattr(self.settings, name)

#     @property
#     def print_settings(self):
#         print(f"n_output_agents: {self.n_output_agents}")
#         print(f"n_output_network: {self.n_output_network}")
#         print(f"print_progress: {self.print_progress}")
#         print(f"output_initial: {self.output_initial}")
#         print(f"start_output: {self.start_output}")
#         print(f"start_numbering_from: {self.start_numbering_from}")


# class DeGrootSettings:
#     def __init__(self, max_iterations: int = None, convergence_tol: float = None):
#         """Initialize the DeGrootSettings object.
#         Which is used to set the settings for the DeGroot model. Usually goes into the simulation options object. Which is then used to run the simulation.

#         Note: Another way to just pass in all the settings and make the simulations object is using the `parse_config_file` function which directly reads the toml file and creates the simulation object. Toml file specifications are discussed in the user guide. This object can still be a good way to set the model settings as it gives more control over the settings.

#         Args:
#             max_iterations (int, optional): The maximum number of iterations, by default is None(infinite iterations).
#             convergence_tol (float, optional): The convergence tolerance, by default is None. Usually set to 1e-6.

#         Returns:
#             object: The DeGrootSettings object.
#         """
#         self.settings = seldoncore.DeGrootSettings()
#         if max_iterations is not None:
#             self.settings.max_iterations = max_iterations
#         self.settings.convergence_tol = convergence_tol

#     def __getattr__(self, name):
#         return getattr(self.settings, name)

#     def __setattr__(self, name, value):
#         if name == "settings":
#             super().__setattr__(name, value)
#         else:
#             setattr(self.settings, name, value)

#     def __delattr__(self, name):
#         delattr(self.settings, name)

#     @property
#     def print_settings(self):
#         print(f"max_iterations: {self.max_iterations}")
#         print(f"convergence_tol: {self.convergence_tol}")


# class DeffuantSettings:
#     def __init__(
#         self,
#         max_iterations: int = None,
#         homophily_threshold: float = 0.2,
#         mu: float = 0.5,
#         use_network: bool = False,
#         use_binary_vector: bool = False,
#         dim: int = 1,
#     ):
#         """Initialize the DeffuantSettings object.
#         Which is used to set the settings for the Deffuant model. Usually goes into the simulation options object. Which is then used to run the simulation.

#         Note: Another way to just pass in all the settings and make the simulations object is using the `parse_config_file` function which directly reads the toml file and creates the simulation object. Toml file specifications are discussed in the user guide. This object can still be a good way to set the model settings as it gives more control over the settings.

#         Args:
#             max_iterations (int, optional): The maximum number of iterations, by default is None(infinite iterations).
#             homophily_threshold (float): The homophily threshold, agents interact if difference in opinion is less than this value, by default is 0.2.
#             mu (float): The mu value,convergence parameter; similar to social interaction strength K (0,0.5]
#             use_network (bool): For using a square lattice network.
#             Note: Square Lattice Network can be created by `generate_square_lattice_network` function.
#             use_binary_vector (bool): Set to use the multi-dimensional DeffuantVectorModel, by default set to false
#             dim (int): The size of the opinions vector. This is used for the multi-dimensional DeffuantVector model.

#         Note: The terms like homophily, mu, use_network, use_binary_vector, and dim are discussed in the user guide.

#         Returns:
#             object: The DeffuantSettings object.
#         """
#         self.settings = seldoncore.DeffuantSettings()
#         if max_iterations is not None:
#             self.settings.max_iterations = max_iterations
#         self.settings.homophily_threshold = homophily_threshold
#         self.settings.mu = mu
#         self.settings.use_network = use_network
#         self.settings.use_binary_vector = use_binary_vector
#         self.settings.dim = dim

#     def __getattr__(self, name):
#         return getattr(self.settings, name)

#     def __setattr__(self, name, value):
#         if name == "settings":
#             super().__setattr__(name, value)
#         else:
#             setattr(self.settings, name, value)

#     def __delattr__(self, name):
#         delattr(self.settings, name)

#     @property
#     def print_settings(self):
#         print(f"max_iterations: {self.max_iterations}")
#         print(f"homophily_threshold: {self.homophily_threshold}")
#         print(f"mu: {self.mu}")
#         print(f"use_network: {self.use_network}")
#         print(f"use_binary_vector: {self.use_binary_vector}")
#         print(f"dim: {self.dim}")


# class ActivityDrivenSettings:
#     def __init__(
#         self,
        # max_iterations: int = None,
        # dt: float = 0.01,
        # m: int = 10,
        # eps: float = 0.01,
        # gamma: float = 2.1,
        # alpha: float = 3.0,
        # homophily: float = 0.5,
        # reciprocity: float = 0.5,
        # K: float = 3.0,
        # mean_activities: bool = False,
        # mean_weights: bool = False,
        # n_bots: int = 0,
        # bot_m: list[int] = [],
        # bot_activity: list[float] = [],
        # bot_opinion: list[float] = [],
        # bot_homophily: list[float] = [],
        # use_reluctances: int = False,
        # reluctance_mean: float = 1.0,
        # reluctance_sigma: float = 0.25,
        # reluctance_eps: float = 0.01,
        # covariance_factor: float = 0.0,
#     ):
#         """Initialize the ActivityDrivenSettings object.
#         Which is used to set the settings for the ActivityDriven model. Usually goes into the simulation options object. Which is then used to run the simulation.

#         Note: Another way to just pass in all the settings and make the simulations object is using the `parse_config_file` function which directly reads the toml file and creates the simulation object. Toml file specifications are discussed in the user guide. This object can still be a good way to set the model settings as it gives more control over the settings.

#         Args:
#             max_iterations (int, optional): Maximum number of iterations, by default is None(infinite iterations).
#             dt (float): Timestep for the integration of the coupled ODEs(Ordinary Differential Equations), by default is 0.01.
#             m (int): Number of agents contacted, when the agent is active, by default is 10.
#             eps (float): The minimum activity epsilon, by default is 0.01.
#             gamma (float): Exponent of activity power law distribution of activities, by default is 2.1.
#             alpha (float): Controversialness of the issue, must be greater than 0, by default is 3.0.
#             homophily (float): The extent to which similar agents interact with similar other, by default is 0.5. Example: if zero, agents pick their interaction partners at random
#             reciprocity (float): Probability that when agent i contacts j via weighted reservoir sampling, j also sends feedback to i. So every agent can have more than m incoming connections, by default is 0.5.
#             K (float): Social interaction strength, by default is 3.0.
#             mean_activities (bool): Whether use the mean value of the powerlaw distribution for the activities of all agents, by default is False.
#             mean_weights (bool): Whether use the meanfield approximation of the network edges, by default is False.
#             n_bots (int): The number of bots to be used, by default is 0 (which means bots are deactivated).
#             Note: Bots are agents that are not influenced by the opinions of other agents, but they can influence the opinions of other agents. So they have fixed opinions and different parameters, the parameters are specified in the following lists
#             bot_m (list[int]): Value of m for the bots, If not specified, defaults to `m`.
#             bot_activity (list[float]): The list of bot activities, If not specified, defaults to 0
#             bot_opinion (list[float]): The fixed opinions of the bots, by default is [].
#             bot_homophily (list[float]): The list of bot homophily, If not specified, defaults to `homophily`
#             use_reluctances (bool): Whether use reluctances, by default is False and every agent has a reluctance of 1.
#             reluctance_mean (float): Mean of distribution before drawing from a truncated normal distribution, by default is 1.0.
#             reluctance_sigma (float): Width of normal distribution (before truncating), by default is 0.25.
#             reluctance_eps (float): Minimum such that the normal distribution is truncated at this value, by default is 0.01.
#             covariance_factor (float): Covariance Factor, defines the correlation between reluctances and activities. Should be in the range of [-1,1], by default is 0.0.

#             Note: The terms like activity, reluctance, homophily, and reciprocity are discussed in the user guide.

#         Returns:
#             object: The ActivityDrivenSettings object.
#         """
#         self.settings = seldoncore.ActivityDrivenSettings()
#         if max_iterations is not None:
#             self.settings.max_iterations = max_iterations
#         self.settings.dt = dt
#         self.settings.m = m
#         self.settings.eps = eps
#         self.settings.gamma = gamma
#         self.settings.alpha = alpha
#         self.settings.homophily = homophily
#         self.settings.reciprocity = reciprocity
#         self.settings.K = K
#         self.settings.mean_activities = mean_activities
#         self.settings.mean_weights = mean_weights
#         self.settings.n_bots = n_bots
#         self.settings.bot_m = bot_m
#         self.settings.bot_activity = bot_activity
#         self.settings.bot_opinion = bot_opinion
#         self.settings.bot_homophily = bot_homophily
#         self.settings.use_reluctances = use_reluctances
#         self.settings.reluctance_mean = reluctance_mean
#         self.settings.reluctance_sigma = reluctance_sigma
#         self.settings.reluctance_eps = reluctance_eps
#         self.settings.covariance_factor = covariance_factor

#     def __getattr__(self, name):
#         return getattr(self.settings, name)

#     def __setattr__(self, name, value):
#         if name == "settings":
#             super().__setattr__(name, value)
#         else:
#             setattr(self.settings, name, value)

#     def __delattr__(self, name):
#         delattr(self.settings, name)

#     @property
#     def print_settings(self):
#         print(f"max_iterations: {self.max_iterations}")
#         print(f"dt: {self.dt}")
#         print(f"m: {self.m}")
#         print(f"eps: {self.eps}")
#         print(f"gamma: {self.gamma}")
#         print(f"alpha: {self.alpha}")
#         print(f"homophily: {self.homophily}")
#         print(f"reciprocity: {self.reciprocity}")
#         print(f"K: {self.K}")
#         print(f"mean_activities: {self.mean_activities}")
#         print(f"mean_weights: {self.mean_weights}")
#         print(f"n_bots: {self.n_bots}")
#         print(f"bot_m: {self.bot_m}")
#         print(f"bot_activity: {self.bot_activity}")
#         print(f"bot_opinion: {self.bot_opinion}")
#         print(f"bot_homophily: {self.bot_homophily}")
#         print(f"use_reluctances: {self.use_reluctances}")
#         print(f"reluctance_mean: {self.reluctance_mean}")
#         print(f"reluctance_sigma: {self.reluctance_sigma}")
#         print(f"reluctance_eps: {self.reluctance_eps}")
#         print(f"covariance_factor: {self.covariance_factor}")


# class InertialSettings:
#     def __init__(
#         self,
#         max_iterations: int = None,
#         dt: float = 0.01,
#         m: int = 10,
#         eps: float = 0.01,
#         gamma: float = 2.1,
#         alpha: float = 3.0,
#         homophily: float = 0.5,
#         reciprocity: float = 0.5,
#         K: float = 3.0,
#         mean_activities: bool = False,
#         mean_weights: bool = False,
#         n_bots: int = 0,
#         bot_m: list[int] = [],
#         bot_activity: list[float] = [],
#         bot_opinion: list[float] = [],
#         bot_homophily: list[float] = [],
#         use_reluctances: int = False,
#         reluctance_mean: float = 1.0,
#         reluctance_sigma: float = 0.25,
#         reluctance_eps: float = 0.01,
#         covariance_factor: float = 0.0,
#         friction_coefficient: float = 1.0,
#     ):
#         """Initialize the ActivityDrivenSettings object.
#         Which is used to set the settings for the ActivityDriven model. Usually goes into the simulation options object. Which is then used to run the simulation.

#         Note: Another way to just pass in all the settings and make the simulations object is using the `parse_config_file` function which directly reads the toml file and creates the simulation object. Toml file specifications are discussed in the user guide. This object can still be a good way to set the Model settings as it gives more control over the settings.

#         Args:
#             max_iterations (int, optional): Maximum number of iterations, by default is None(infinite iterations).
#             dt (float): Timestep for the integration of the coupled ODEs(Ordinary Differential Equations), by default is 0.01.
#             m (int): Number of agents contacted, when the agent is active, by default is 10.
#             eps (float): The minimum activity epsilon, by default is 0.01.
#             gamma (float): Exponent of activity power law distribution of activities, by default is 2.1.
#             alpha (float): Controversialness of the issue, must be greater than 0, by default is 3.0.
#             homophily (float): The extent to which similar agents interact with similar other, by default is 0.5. Example: if zero, agents pick their interaction partners at random
#             reciprocity (float): Probability that when agent i contacts j via weighted reservoir sampling, j also sends feedback to i. So every agent can have more than m incoming connections, by default is 0.5.
#             K (float): Social interaction strength, by default is 3.0.
#             mean_activities (bool): Whether use the mean value of the powerlaw distribution for the activities of all agents, by default is False.
#             mean_weights (bool): Whether use the meanfield approximation of the network edges, by default is False.
#             n_bots (int): The number of bots to be used, by default is 0 (which means bots are deactivated).
#             Note: Bots are agents that are not influenced by the opinions of other agents, but they can influence the opinions of other agents. So they have fixed opinions and different parameters, the parameters are specified in the following lists
#             bot_m (list[int]): Value of m for the bots, If not specified, defaults to `m`.
#             bot_activity (list[float]): The list of bot activities, If not specified, defaults to 0
#             bot_opinion (list[float]): The fixed opinions of the bots, by default is [].
#             bot_homophily (list[float]): The list of bot homophily, If not specified, defaults to `homophily`
#             use_reluctances (bool): Whether use reluctances, by default is False and every agent has a reluctance of 1.
#             reluctance_mean (float): Mean of distribution before drawing from a truncated normal distribution, by default is 1.0.
#             reluctance_sigma (float): Width of normal distribution (before truncating), by default is 0.25.
#             reluctance_eps (float): Minimum such that the normal distribution is truncated at this value, by default is 0.01.
#             covariance_factor (float): Covariance Factor, defines the correlation between reluctances and activities. Should be in the range of [-1,1], by default is 0.0.
#             friction_coefficient (float): Friction Coefficient of the agents, by default is 1.0 (making agents tend to go to rest without acceleration).
#             Note: friction_coefficient should be in the range of [0,1]. If it is 0, the agents will not stop moving, and if it is 1, the agents will stop moving immediately.

#             Note: The terms like opinion, activity, reluctance, and velocity are discussed in the user guide.

#         Returns:
#             object: The ActivityDrivenSettings object.
#         """
#         self.settings = seldoncore.ActivityDrivenInertialSettings()
#         if max_iterations is not None:
#             self.settings.max_iterations = max_iterations
#         self.settings.dt = dt
#         self.settings.m = m
#         self.settings.eps = eps
#         self.settings.gamma = gamma
#         self.settings.alpha = alpha
#         self.settings.homophily = homophily
#         self.settings.reciprocity = reciprocity
#         self.settings.K = K
#         self.settings.mean_activities = mean_activities
#         self.settings.mean_weights = mean_weights
#         self.settings.n_bots = n_bots
#         self.settings.bot_m = bot_m
#         self.settings.bot_activity = bot_activity
#         self.settings.bot_opinion = bot_opinion
#         self.settings.bot_homophily = bot_homophily
#         self.settings.use_reluctances = use_reluctances
#         self.settings.reluctance_mean = reluctance_mean
#         self.settings.reluctance_sigma = reluctance_sigma
#         self.settings.reluctance_eps = reluctance_eps
#         self.settings.covariance_factor = covariance_factor
#         self.settings.friction_coefficient = friction_coefficient

#     def __getattr__(self, name):
#         return getattr(self.settings, name)

#     def __setattr__(self, name, value):
#         if name == "settings":
#             super().__setattr__(name, value)
#         else:
#             setattr(self.settings, name, value)

#     def __delattr__(self, name):
#         delattr(self.settings, name)

#     @property
#     def print_settings(self):
#         print(f"max_iterations: {self.max_iterations}")
#         print(f"dt: {self.dt}")
#         print(f"m: {self.m}")
#         print(f"eps: {self.eps}")
#         print(f"gamma: {self.gamma}")
#         print(f"alpha: {self.alpha}")
#         print(f"homophily: {self.homophily}")
#         print(f"reciprocity: {self.reciprocity}")
#         print(f"K: {self.K}")
#         print(f"mean_activities: {self.mean_activities}")
#         print(f"mean_weights: {self.mean_weights}")
#         print(f"n_bots: {self.n_bots}")
#         print(f"bot_m: {self.bot_m}")
#         print(f"bot_activity: {self.bot_activity}")
#         print(f"bot_opinion: {self.bot_opinion}")
#         print(f"bot_homophily: {self.bot_homophily}")
#         print(f"use_reluctances: {self.use_reluctances}")
#         print(f"reluctance_mean: {self.reluctance_mean}")
#         print(f"reluctance_sigma: {self.reluctance_sigma}")
#         print(f"reluctance_eps: {self.reluctance_eps}")
#         print(f"covariance_factor: {self.covariance_factor}")
#         print(f"friction_coefficient: {self.friction_coefficient}")


# class InitialNetworkSettings:
#     def __init__(
#         self,
#         file: str = None,
#         number_of_agents: int = 200,
#         connections_per_agent: int = 10,
#     ):
#         self.settings = seldoncore.InitialNetworkSettings()

#         """Initialize the InitialNetworkSettings object.
#         Which is used to set the initial network settings for the simulation. Usually goes into the simulation options object. Which is then used to run the simulation.
        
#         Note: Another way to just pass in all the settings and make the simulations object is using the `parse_config_file` function which directly reads the toml file and creates the simulation object. Toml file specifications are discussed in the user guide. This object can still be a good way to set the initial network settings as it gives more control over the settings.
        
#         Args:
#             file (str, optional): The path to the network file, by default is None.
#             number_of_agents (int): The number of agents in the network, by default is 200.
#             connections_per_agent (int): The number of connections per agent, by default is 10."""

#         if file is not None:
#             self.settings.file = file
#         self.settings.number_of_agents = number_of_agents
#         self.settings.connections_per_agent = connections_per_agent

#     def __getattr__(self, name):
#         return getattr(self.settings, name)

#     def __setattr__(self, name, value):
#         if name == "settings":
#             super().__setattr__(name, value)
#         else:
#             setattr(self.settings, name, value)

#     def __delattr__(self, name):
#         delattr(self.settings, name)

#     @property
#     def print_settings(self):
#         print(f"file: {self.file}")
#         print(f"number_of_agents: {self.number_of_agents}")
#         print(f"connections_per_agent: {self.connections_per_agent}")


# class SimulationOptions:
#     def __init__(
#         self,
#         model_string: str = None,
#         model_settings: Union[
#             DeGrootSettings, DeffuantSettings, ActivityDrivenSettings, InertialSettings
#         ] = None,
#         rng_seed: int = None,
#         output_settings: OutputSettings = OutputSettings(),
#         network_settings: InitialNetworkSettings = InitialNetworkSettings(),
#     ):
#         self.options = seldoncore.SimulationOptions()

#         """Initialize the SimulationOptions object.
#         Makes the simulation options object which is used to run the simulation. Usually goes into the Simulation object. Which is then used to run the simulation.
        
#         Note: Another way to just pass in all the settings and make the simulations object is using the `parse_config_file` function which directly reads the toml file and creates the simulation object. Toml file specifications are discussed in the user guide. This object can still be a good way to set the simulation options as it gives more control over the settings.
        
#         Args:
#             model_string (str): The model string. The allowed models are DeGroot, Deffuant, DeffuantVector, ActivityDriven, and ActivityDrivenInertial.
#             model_settings (object): The model settings object. The allowed models are DeGrootSettings, DeffuantSettings, ActivityDrivenSettings, and InertialSettings.
#             rng_seed (int, optional): The random seed, if not set it will pick a random seed.
#             output_settings (object): The output settings object, by default is OutputSettings().
#             network_settings (object): The initial network settings object, by default is InitialNetworkSettings().
        
#         Returns:
#             The SimulationOptions object.
#         """

#         if not model_string:
#             raise ValueError("model_string must be provided")

#         valid_models = [
#             "DeGroot",
#             "Deffuant",
#             "DeffuantVector",
#             "ActivityDriven",
#             "ActivityDrivenInertial",
#         ]
#         if model_string not in valid_models:
#             raise ValueError(
#                 f"Invalid model_string: {model_string}. Must be one of {valid_models}"
#             )
#         self.model_string = model_string

#         if model_string == "DeGroot":
#             self.options.model = seldoncore.Model.DeGroot
#         elif model_string == "Deffuant" or model_string == "DeffuantVector":
#             self.options.model = seldoncore.Model.DeffuantModel
#         elif model_string == "ActivityDriven":
#             self.options.model = seldoncore.Model.ActivityDrivenModel
#         elif model_string == "ActivityDrivenInertial":
#             self.options.model = seldoncore.Model.ActivityDrivenInertial

#         if rng_seed is not None:
#             self.rng_seed = rng_seed

#         if isinstance(output_settings, OutputSettings):
#             self.options.output_settings = output_settings.settings

#         if isinstance(network_settings, InitialNetworkSettings):
#             self.options.network_settings = network_settings.settings

#         if isinstance(model_settings, DeGrootSettings):
#             self.options.model_settings = model_settings.settings
#         elif isinstance(model_settings, DeffuantSettings):
#             self.options.model_settings = model_settings.settings
#         elif isinstance(model_settings, ActivityDrivenSettings):
#             self.options.model_settings = model_settings.settings
#         elif isinstance(model_settings, InertialSettings):
#             self.options.model_settings = model_settings.settings
#         else:
#             TypeError(
#                 "Model settings not found, the allowed models are DeGroot, Deffuant, DeffuantVector, ActivityDriven, and ActivityDrivenInertial"
#             )

#         seldoncore.validate_settings(self.options)

#     def __getattr__(self, name):
#         return getattr(self.options, name)

#     def __setattr__(self, name, value):
#         if name == "options":
#             super().__setattr__(name, value)
#         else:
#             setattr(self.options, name, value)

#     def __delattr__(self, name):
#         delattr(self.options, name)

#     @property
#     def print_settings(self):
#         print(f"model_string: {self.options.model_string}")
#         print(f"rng_seed: {self.options.rng_seed}")
#         print(f"output_settings: {self.options.output_settings.print_settings}")
#         print(f"network_settings: {self.options.network_settings.print_settings}")
#         print(f"model_settings: {self.options.model_settings.print_settings}")


# def parse_config_file(file_path: str):
#     """Parse the toml file and create the simulation options object.

#     Args:
#         file_path (str): The path to the toml file.

#     Returns:
#         Simulation: The simulation object.
#     """
#     return seldoncore.parse_config_file(file_path)


# def generate_n_connections_network(
#     model_string: str,
#     n_agents: int,
#     n_connections: int,
#     self_iteraction: bool,
#     seed: int = None,
# ):
#     """Generate a network with n connections per agent.

#     Args:
#         model_string (str): The model string. The allowed models are DeGroot, Deffuant, DeffuantVector, ActivityDriven, and ActivityDrivenInertial.
#         n_agents (int): The number of agents in the network.
#         n_connections (int): The number of connections per agent.
#         self_iteraction (bool, optional): Whether to allow self-interaction, by default is False.
#         seed (int, optional): The random seed, by default is None.

#     Returns:
#         Network: The network object.
#     """
#     if model_string not in [
#         "DeGroot",
#         "Deffuant",
#         "DeffuantVector",
#         "ActivityDriven",
#         "ActivityDrivenInertial",
#     ]:
#         raise ValueError(
#             f"Invalid model_string: {model_string}. Must be one of ['DeGroot', 'Deffuant', 'DeffuantVector', 'ActivityDriven', 'ActivityDrivenInertial']"
#         )

#     if model_string == "DeGroot":
#         return seldoncore.generate_n_connections_network(
#             n_agents, n_connections, self_iteraction
#         )
