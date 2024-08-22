"""
The DeGroot Model is a fundamental framework in social influence theory, illustrating how agents in a network update their opinions based on the opinions of their neighbors. At its core, the model posits that each agent revises their opinion by taking the weighted average of their neighbors' opinions. This iterative process continues until the opinions converge to a consensus or a stable state.

In practical terms, consider a group of individuals—such as a committee or a team—each holding a subjective probability distribution for an unknown parameter. The DeGroot Model describes how these individuals might converge on a unified subjective probability distribution through iterative opinion updates, effectively pooling their individual insights.

Additionally, the model is versatile and can be applied to scenarios where opinions are represented as point estimates rather than probability distributions. In such cases, the DeGroot Model helps illustrate how a group can achieve consensus on a specific parameter estimate, reflecting the collective judgment of the group.

Key features
------------

Opinion Averaging:
  Agents update their opinions based on the average opinions of their neighbors, fostering convergence and consensus.

Iterative Process:
  The model operates through a series of iterations, with opinions being refined each step until stability is achieved.

Consensus Formation:
  Applicable to both probability distributions and point estimates, showing how diverse opinions can be aggregated into a common view.

The DeGroot Model provides a clear and elegant approach to understanding how social influence and information sharing lead to collective agreement within a network of agents.

Example:
---------
>>> from pyseldon import DeGroot_Model
>>> # Create the DeGroot Model
>>> model = DeGroot_Model(max_iterations=1000, convergence_tol=1e-6)
>>> # Run the simulation
>>> model.run("output_dir")
>>> # Access the network
>>> network = model.get_Network()
>>> # Access the opinions of the agents
>>> opinions = model.agents_opinions()

Reference:
----------
.. bibliography:: 
   :style: plain
   
   DeGroot_1974

*************
"""

from bindings import seldoncore
from typing import Optional
from ._basemodel import Base_Model

from ._othersettings import Other_Settings


class DeGroot_Model(Base_Model):
    """
    DeGroot Model base class for Simulation.

    Parameters
    -----------
    max_iterations : int, default=None
      The maximum number of iterations to run the simulation. If None, the simulation runs infinitely.

    convergence_tol : float, default=1e-6
      The tolerance for convergence of the simulation.

    rng_seed : int, default=None
      The seed for the random number generator. If not provided, a random seed is picked.

    agent_file : str, default=None
      The file to read the agents from. If None, the agents are generated randomly.

    network_file : str, default=None
      The file to read the network from. If None, the network is generated randomly

    other_settings : Other_Settings, default=None
      The other settings for the simulation. If None, the default settings are used.

    Attributes
    -----------
    Network : Network (Object)
      The network generated by the simulation.

    Opinion : Float
      The opinions of the agents or nodes of the network.

    see also: seldoncore.Network
    """

    def __init__(
        self,
        max_iterations: int = None,
        convergence_tol: float = 1e-6,
        rng_seed: Optional[int] = None,
        agent_file: Optional[str] = None,
        network_file: Optional[str] = None,
        other_settings: Other_Settings = None,
    ):
        super().__init__()
        if other_settings is not None:
            self.other_settings = other_settings

        self._options.model_string = "DeGroot"
        self._options.model_settings = seldoncore.DeGrootSettings()
        self._options.output_settings = self.other_settings.output_settings
        self._options.network_settings = self.other_settings.network_settings
        self._options.model = seldoncore.Model.DeGroot

        self._options.model_settings.max_iterations = max_iterations
        self._options.model_settings.convergence_tol = convergence_tol

        if rng_seed is not None:
            self._options.rng_seed = rng_seed

        self._simulation = seldoncore.SimulationSimpleAgent(
            options=self._options,
            cli_agent_file=agent_file,
            cli_network_file=network_file,
        )

        self._network = self._simulation.network
