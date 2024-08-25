"""
In the study of Social Opinion Dynamics, it is essential to consider various factors that influence the formation and dissemination of opinions within networks of agents. The Activity Driven Inertial Model (or just Inertial Model) expands upon the traditional Activity Driven Model by introducing an additional parameter: the friction coefficient. This model simulates the evolution of opinions over time, capturing the complex interactions that lead to consensus formation, polarization, and the influence of external entities such as bots.

The friction coefficient in this model represents the resistance to change in an agent's opinion, introducing an inertial effect. This concept is akin to physical inertia, where an agent's opinion resists sudden changes, leading to smoother transitions in the opinion landscape. This feature adds realism by modeling how strongly held beliefs or social inertia can slow the pace of opinion change, even in the face of active social interactions.

Like the traditional Activity Driven Model, the Inertial Model is highly configurable, allowing users to explore a wide range of scenarios and behaviors by adjusting parameters related to agent activity, social influence, homophily, and now, opinion inertia. The model also considers the role of bots—agents with fixed opinions—who interact with humans on social media platforms, influencing human-machine interactions in opinion dynamics.

Key features as discussed in the Activity Driven Model:

Temporal Dynamics
-----------------
max_iterations:
    Limits the total number of simulation steps.
    If set to None, the model runs indefinitely, allowing for long-term analysis of opinion evolution.

dt:
    Defines the time step for each iteration, controlling the pace at which the simulation progresses.
    Smaller values lead to more granular updates, while larger values speed up the simulation but might miss finer details.

Agent Behavior and Interaction
------------------------------
m:
  Determines how many agents an active agent interacts with during each time step.
  Influences the rate of opinion spreading; higher values mean more interactions and potentially faster consensus or polarization.

eps:
    Sets the minimum activity level for agents, ensuring no agent is completely inactive.
    Helps prevent stagnation in the model by keeping all agents engaged at some level.

gamma:
    Controls the distribution of agent activity, typically following a power-law where few agents are very active, and many are less active.
    Affects how central or peripheral agents influence the overall opinion dynamics.

homophily:
    Measures the tendency of agents to interact with others who share similar opinions.
    High homophily can lead to echo chambers, while low homophily promotes diverse interactions.

reciprocity:
    Determines whether agents are likely to reciprocate interactions, creating more mutual or one-sided connections.
    High reciprocity strengthens bidirectional relationships, potentially stabilizing opinion clusters.

K:
    Represents the strength of social influence between agents.
    A higher K means opinions are more strongly influenced by interactions, which can accelerate consensus or deepen polarization.

Social Context and Controversialness
------------------------------------
alpha:
    Controls the degree of controversialness of the issue being simulated.
    A higher alpha can lead to more polarized opinions, as agents might have stronger reactions to the issue.

Bots and External Influence
---------------------------
n_bots:
    Specifies the number of bots in the simulation, which are fixed in their opinions.
    Bots influence the network without being influenced, potentially driving opinion shifts or reinforcing certain views.

bot_m, bot_activity, bot_opinion, bot_homophily:
    Define the specific behaviors and characteristics of bots, such as how often they interact or how similar they are to other agents.
    These parameters allow bots to mimic or diverge from regular agents, providing a controlled way to study external influence.

Reluctance and Activity Correlation
-----------------------------------
use_reluctances:
    Activates the feature where agents have a reluctance to change their opinions.
    Adds complexity by simulating resistance to change, affecting how quickly or slowly opinions evolve.

reluctance_mean, reluctance_sigma, reluctance_eps:
    Define the distribution of reluctance across agents, determining the average resistance and its variability.
    These parameters help model heterogeneous populations where some agents are more resistant to change than others.

covariance_factor:
    Introduces a correlation between an agent's activity level and its reluctance, meaning that activity might influence or be influenced by reluctance.
    Allows for more realistic scenarios where active agents may be more or less open to changing their opinions, depending on the sign of the covariance.

Friction Coefficient
--------------------
The friction coefficient in the Activity Driven Inertial Model represents the resistance to change in an agent's opinion, akin to physical inertia. It introduces an inertial effect that causes opinions to change more gradually, reflecting the persistence of strongly held beliefs and social inertia in opinion dynamics.

Example:
---------
>>> from pyseldonlib import Inertial_Model
>>> # Create the Inertial Model
>>> model = Inertial_Model(max_iterations=1000, convergence_tol=1e-6)
>>> # Run the simulation
>>> model.run("output_dir")
>>> # Access the network
>>> network = model.get_Network()
>>> # Access the opinions of the agents
>>> opinions = model.agents_opinions()
>>> activity = model.agents_activity()
>>> reluctance = model.agents_reluctance()
>>> velocity = model.agent_velocity()

Reference:
----------
.. bibliography::
   :style: plain

   Baumann_2020
   Baumann_2021

*************
"""

from bindings import seldoncore
import pathlib
from typing import Optional
from .ActivityDrivenModel import Activity_Driven_Model
from ._othersettings import Other_Settings


class Inertial_Model(Activity_Driven_Model):
    """
    Inertial Model base class for Simulation.

    Parameters
    -----------
    max_iterations : int, default=None
      The maximum number of iterations to run the simulation. If None, the simulation runs infinitely.

    dt : float, default=0.01
      The time step for the simulation.

    m : int, default=10
      Number of agents contacted, when the agent is active.

    eps : float, default=0.01
      The minimum activity epsilon.

    gamma : float, default=2.1
      Exponent of activity power law distribution of activities.

    alpha : float default=3.0
      Controversialness of the issue, must be greater than 0.

    homophily : float, default=0.5
      The extent to which similar agents interact with similar other.
      Example: If 0.0, agents pick their interaction partners at random.
               If 1.0, agents interact only with agents of the same opinion.

    reciprocity : float, default=0.5
      The extent to which agents reciprocate interactions.
      Example: If 0.0, agents do not reciprocate interactions.
               If 1.0, agents reciprocate all interactions.

    K : float, default=3.0
      Social interaction strength.

    mean_activities : bool, default=False
      Whether use the mean value of the powerlaw distribution for the activities of all agents.

    mean_weights : bool, default=False
      Whether use the meanfield approximation of the network edges, by default is False.

    n_bots : int, default=0
      Number of bots in the simulation.

    .. note::
      Bots are agents that are not influenced by the opinions of other agents, but they can influence the opinions of other agents. So they have fixed opinions and different parameters, the parameters are specified in the following lists.

    bot_m : list[int], default=[]
      Value of m for the bots, If not specified, defaults to `m`.

    bot_activity : list[float], default=[],
      The list of bot activities, If not specified, defaults to 0.

    bot_opinion : list[float], default=[]
      The fixed opinions of the bots.

    bot_homophily : list[float], default=[]
      The list of bot homophily, If not specified, defaults to `homophily`.

    use_reluctances : int, default=False
      Whether use reluctances, by default is False and every agent has a reluctance of 1.

    reluctance_mean : float, default=1.0
      Mean of distribution before drawing from a truncated normal distribution.

    reluctance_sigma : float, default=0.25
      Width of normal distribution (before truncating).

    reluctance_eps : float, default=0.01
      Minimum such that the normal distribution is truncated at this value.

    covariance_factor : float, default=0.0
      Covariance Factor, defines the correlation between reluctances and activities.

    rng_seed : int, default=None
      The seed for the random number generator. If not provided, a random seed is picked.

    agent_file : str, default=None
      The file to read the agents from. If None, the agents are generated randomly.

    network_file : str, default=None
      The file to read the network from. If None, the network is generated randomly

    other_settings : Other_Settings, default=None
      The other settings for the simulation. If None, the default settings are used.

    friction_coefficient : float, default=1.0
      The friction coefficient for the inertial model.

    Attributes
    -----------
    Network : Network (Object)
      The network generated by the simulation.

    Opinion : Float
      The opinions of the agents or nodes of the network.

    Activity : Float
      The activity of the agents or nodes of the network.

    Reluctance : Float
      The reluctance of the agents or nodes of the network.
    """

    def __init__(
        self,
        max_iterations: int = None,
        dt: float = 0.01,
        m: int = 10,
        eps: float = 0.01,
        gamma: float = 2.1,
        alpha: float = 3.0,
        homophily: float = 0.5,
        reciprocity: float = 0.5,
        K: float = 3.0,
        mean_activities: bool = False,
        mean_weights: bool = False,
        n_bots: int = 0,
        bot_m: list[int] = [],
        bot_activity: list[float] = [],
        bot_opinion: list[float] = [],
        bot_homophily: list[float] = [],
        use_reluctances: int = False,
        reluctance_mean: float = 1.0,
        reluctance_sigma: float = 0.25,
        reluctance_eps: float = 0.01,
        covariance_factor: float = 0.0,
        friction_coefficient: float = 1.0,
        rng_seed: Optional[int] = None,
        agent_file: Optional[str] = None,
        network_file: Optional[str] = None,
        other_settings: Other_Settings = None,
    ):
        # Other settings and Simulation Options are already intialised in super
        super().__init__()
        if other_settings is not None:
            self.other_settings = other_settings

        self._options.model_string = "ActivityDrivenInertial"
        self._options.model_settings = seldoncore.ActivityDrivenInertialSettings()
        self._options.output_settings = self.other_settings.output_settings
        self._options.network_settings = self.other_settings.network_settings
        self._options.model = seldoncore.Model.ActivityDrivenInertial

        if rng_seed is not None:
            self._options.rng_seed = rng_seed

        self._options.model_settings.max_iterations = max_iterations
        self._options.model_settings.dt = dt
        self._options.model_settings.m = m
        self._options.model_settings.eps = eps
        self._options.model_settings.gamma = gamma
        self._options.model_settings.alpha = alpha
        self._options.model_settings.homophily = homophily
        self._options.model_settings.reciprocity = reciprocity
        self._options.model_settings.K = K
        self._options.model_settings.mean_activities = mean_activities
        self._options.model_settings.mean_weights = mean_weights
        self._options.model_settings.n_bots = n_bots
        self._options.model_settings.bot_m = bot_m
        self._options.model_settings.bot_activity = bot_activity
        self._options.model_settings.bot_opinion = bot_opinion
        self._options.model_settings.bot_homophily = bot_homophily
        self._options.model_settings.use_reluctances = use_reluctances
        self._options.model_settings.reluctance_mean = reluctance_mean
        self._options.model_settings.reluctance_sigma = reluctance_sigma
        self._options.model_settings.reluctance_eps = reluctance_eps
        self._options.model_settings.covariance_factor = covariance_factor
        self._options.model_settings.friction_coefficient = friction_coefficient

        self._simulation = seldoncore.SimulationInertialAgent(
            options=self._options,
            cli_agent_file=agent_file,
            cli_network_file=network_file,
        )

        self._network = self._simulation.network

    def agent_velocity(self, index: int = None):
        """
        Access the agents reluctance data from the simulated network.

        Parameters
        -----------
        index : int
          The index of the agent to access. The index is 0-based. If not provided, all agents are returned.
        """
        if index is None:
            result = [agent.data.velocity for agent in self._simulation.network.agent]
            return result
        else:
            if index < 0 or index >= self.Network.n_agents():
                raise IndexError("Agent index is out of range.")
            return self._simulation.network.agent[index].data.velocity

    def set_agent_velocity(self, index: int, velocity: float):
        """
        Set the velocity of a specific agent.

        Parameters
        ----------
        index : int
            The index of the agent whose opinion is to be set.
        velocity : float
            The new velocity value for the agent.
        """
        if index < 0 or index >= self.Network.n_agents():
            raise IndexError("Agent index is out of range.")

        self._simulation.network.agent[index].data.velocity = velocity
