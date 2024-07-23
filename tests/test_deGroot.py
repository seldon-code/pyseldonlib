import pyseldon.seldoncore as pd
import math
import pytest


def test_deGroot():
    n_agents = 2
    neighbour_list = [[1, 0], [0, 1]]
    weight_list = [[0.2, 0.8], [0.2, 0.8]]

    settings = pd.DeGrootSettings()
    settings.max_iterations = 100
    settings.convergence_tol = 1e-6

    network = pd.SimpleAgentNetwork(
        neighbour_list=neighbour_list, weight_list=weight_list, direction="Incoming"
    )

    simulation = pd.SimulationSimpleAgent(
        options=pd.SimulationOptions(model_string="DeGroot", model_settings=settings)
    )

    simulation.network.agent[0].data.opinion = 0.0
    simulation.network.agent[1].data.opinion = 1.0

    simulation.run()

    for i in range(n_agents):
        print(f"Opinion {i} = {network.agent[i].data.opinion}")
        assert math.isclose(
            network.agent[i].data.opinion, 0.5, abs_tol=settings.convergence_tol * 10.0
        )


test_deGroot()

# if __name__ == "__main__":
#   pytest.main([__file__])
