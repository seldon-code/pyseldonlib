import pyseldon.seldoncore as pd
from pyseldon.seldoncore import ActivityAgentNetwork
import pytest

network = pd.generate_fully_connected_activity_driven(
    n_agents=100, weight=None, seed=None
)

assert isinstance(network, ActivityAgentNetwork)

if __name__ == "__main__":
    pytest.main([__file__])
