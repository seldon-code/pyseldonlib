from pydantic import BaseModel, Field
from typing import List, Optional

class Other_SettingsConfig(BaseModel):
  """All other settings for the simulation.
  """
  n_output_agents: Optional[int] = Field(None, description="Write out the agents every n iterations.")
  n_output_network: Optional[int] = Field(None, description="Write out the network every n iterations.")
  print_progress: bool = Field(False, description="Print the progress of the simulation.")
  output_initial: bool = Field(True, description="Output initial opinions and network.")
  start_output: int = Field(1, description="Start printing opinion and/or network files from this iteration number.")
  start_numbering_from: int = Field(0, description="The initial step number, before the simulation runs, is this value. The first step would be (1+start_numbering_from).")
  number_of_agents: int = Field(200, description="The number of agents in the network.")
  connections_per_agent: int = Field(10, description="The number of connections per agent.")


