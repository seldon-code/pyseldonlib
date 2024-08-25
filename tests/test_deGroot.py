import pyseldonlib
import math
import pytest
import pathlib
import shutil


def test_deGroot():
    n_agents = 2
    neighbour_list = [[1, 0], [0, 1]]
    weight_list = [[0.2, 0.8], [0.2, 0.8]]

    network = pyseldonlib.seldoncore.SimpleAgentNetwork(
        neighbour_list=neighbour_list, weight_list=weight_list, direction="Incoming"
    )

    output_dir = str(pathlib.Path.cwd() / "tests/output")

    model = pyseldonlib.DeGroot_Model(max_iterations=100, convergence_tol=1e-6)
    model.Network = network
    model.set_agent_opinion(0, 0.0)
    model.set_agent_opinion(1, 1.0)
    if pathlib.Path(output_dir).exists():
        shutil.rmtree(output_dir)

    model.run(output_dir)

    for i in range(n_agents):
        print(f"Opinion {i} = {model.agent_opinion(i)}")
        assert math.isclose(model.agent_opinion(i), 0.5, rel_tol=0.1)
    shutil.rmtree(output_dir)


if __name__ == "__main__":
    pytest.main([__file__])
