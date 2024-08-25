from bindings import seldoncore
from typing import Optional


class Other_Settings:
    """
    All other settings for the simulation.

    Parameters
    -----------

    n_output_agents :  int, default=None
      Write out the agents every n iterations.

    n_output_network : int, default=None
      Write out the network every n iterations.

    print_progress : bool, default=False
      Print the progress of the simulation.

    output_initial : bool, default=True
      Output initial opinions and network.

    start_output : int, default=1
      Start printing opinion and/or network files from this iteration number.

    start_numbering_from : int, default=0
      The initial step number, before the simulation runs, is this value. The first step would be (1+start_numbering_from).

    number_of_agents : int, default=200
      The number of agents in the network.

    connections_per_agent : int, default=10
      The number of connections per agent.
    """

    def __init__(
        self,
        n_output_agents: Optional[int] = None,
        n_output_network: Optional[int] = None,
        print_progress: bool = False,
        output_initial: bool = True,
        start_output: int = 1,
        start_numbering_from: int = 0,
        number_of_agents: int = 200,
        connections_per_agent: int = 10,
    ):
        self.output_settings = seldoncore.OutputSettings()
        self.output_settings.n_output_agents = n_output_agents
        self.output_settings.n_output_network = n_output_network
        self.output_settings.print_progress = print_progress
        self.output_settings.output_initial = output_initial
        self.output_settings.start_output = start_output
        self.output_settings.start_numbering_from = start_numbering_from
        self.network_settings = seldoncore.InitialNetworkSettings()
        self.network_settings.number_of_agents = number_of_agents
        self.network_settings.connections_per_agent = connections_per_agent
