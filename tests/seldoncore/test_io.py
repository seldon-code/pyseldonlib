import pyseldonlib
import pytest
import pathlib


# Test reading in the network from a file
def test_io_network():
    proj_root_path = pathlib.Path.cwd()
    network_file = str(proj_root_path / "tests" / "res" / "network.txt")
    network = pyseldonlib.seldoncore.generate_from_file_activity_agent(network_file)

    assert network.n_agents() == 3

    neighbours_expected = [[2, 1], [], [1]]
    weights_expected = [[0.1, -0.2], [], [1.2]]

    for i in range(0, network.n_agents()):
        assert neighbours_expected[i] == network.get_neighbours(i)
        assert weights_expected[i] == network.get_weights(i)


def test_io_agents():
    proj_root_path = pathlib.Path.cwd()
    agent_file = str(proj_root_path / "tests" / "res" / "opinions.txt")

    agents = pyseldonlib.seldoncore.agents_from_file_activity_agent(agent_file)
    opinions_expected = [2.1127107987061544, 0.8088982488089491, -0.8802809369462433]
    activities_expected = [
        0.044554683389757696,
        0.015813166022685163,
        0.015863953902810535,
    ]
    reluctances_expected = [1.0, 1.0, 2.3]

    assert len(agents) == 3

    for i in range(0, len(agents)):
        assert agents[i].data.opinion == pytest.approx(opinions_expected[i], abs=1e-16)
        assert agents[i].data.activity == pytest.approx(
            activities_expected[i], abs=1e-16
        )
        assert agents[i].data.reluctance == pytest.approx(
            reluctances_expected[i], abs=1e-16
        )


if __name__ == "__main__":
    pytest.main([__file__])
