import pyseldon
import math
import pytest
import pathlib
import shutil


def test_deGroot():
    n_agents = 2
    neighbour_list = [[1, 0], [0, 1]]
    weight_list = [[0.2, 0.8], [0.2, 0.8]]

    network = pyseldon.seldoncore.SimpleAgentNetwork(
        neighbour_list=neighbour_list, weight_list=weight_list, direction="Incoming"
    )

    network_file = str(pathlib.Path.cwd() / "tests/network/net.txt")
    output_dir = str(pathlib.Path.cwd() / "tests/output")
    pyseldon.seldoncore.network_to_dot_file_simple_agent(network, network_file)

    model = pyseldon.DeGroot_Model(
        max_iterations=100, convergence_tol=1e-6, network_file=network_file
    )
    model.set_agent_opinion(0, 0.0)
    model.set_agent_opinion(1, 1.0)
    model.run(output_dir)
    print(model.agent_opinion())

    for i in range(n_agents):
        print(f"Opinion {i} = {model.agent_opinion(i)}")
        assert math.isclose(
            model.agent_opinion(i), 0.5, abs_tol=model.convergence_tol * 10.0
        )
    shutil.rmtree(output_dir)


if __name__ == "__main__":
    pytest.main([__file__])
