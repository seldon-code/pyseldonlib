import pyseldon.seldoncore as pd
import pytest

network = pd.generate_fully_connected_activity_driven(n_agents=100, weight = None, seed = None)


pd.agents_to_file(network, "/home/parrot_user/Desktop/pyseldon/agents.txt")
print(type(network))