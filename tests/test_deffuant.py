import pyseldon
import pytest
from pathlib import Path


def test_basic_deffuant_model_two_agents():
    proj_root = Path.cwd()
    input_file = str(proj_root / "tests" / "res" / "deffuant_2agents.toml")
    print(type(input_file))
    options = pyseldon.seldoncore.parse_config_file(input_file)

    simulation = pyseldon.seldoncore.SimulationSimpleAgent(options=options)

    output_dir_path = str(proj_root / "tests" / "output_deffuant")
    model_settings = options.model_settings
    mu = model_settings.mu
    homophily_threshold = model_settings.homophily_threshold

    # agents are too far apart, we dont expect any change with the iterations
    agent1_init = homophily_threshold * 1.1
    agent2_init = 0

    simulation.network.agent[0].data.opinion = agent1_init
    simulation.network.agent[1].data.opinion = agent2_init

    simulation.run(output_dir_path)

    assert simulation.network.agent[0].data.opinion == pytest.approx(agent1_init)
    assert simulation.network.agent[1].data.opinion == pytest.approx(agent2_init)

    # agents are too close, we expect them to converge to the same opinion
    agent1_init = homophily_threshold * 0.9
    agent2_init = 0

    simulation.network.agent[0].data.opinion = agent1_init
    simulation.network.agent[1].data.opinion = agent2_init

    simulation.run(output_dir_path)

    n_iterations = model_settings.max_iterations
    expected_diff = (1.0 - 2.0 * mu) ** (2 * n_iterations) * (agent1_init - agent2_init)
    assert simulation.network.agent[0].data.opinion - simulation.network.agent[
        1
    ].data.opinion == pytest.approx(expected_diff)


def test_lattice_deffuant_model_16X16_agents():
    proj_root = Path.cwd()
    input_file = str(proj_root / "tests" / "res" / "deffuant_16x16_agents.toml")
    options = pyseldon.seldoncore.parse_config_file(input_file)

    simulation = pyseldon.seldoncore.SimulationSimpleAgent(options=options)

    output_dir_path = str(proj_root / "tests" / "output_deffuant")
    model_settings = options.model_settings
    homophily_threshold = model_settings.homophily_threshold

    n_agents = simulation.network.n_agents()
    n_agents_half = int(n_agents / 2)
    avg_opinion = 0

    # first half with low opinions
    for i in range(0, n_agents_half):
        op = -homophily_threshold - 0.5 * i / n_agents * homophily_threshold
        avg_opinion += op / float(n_agents_half)
        simulation.network.agent[i].data.opinion = op

    # second half with low opinions
    for i in range(n_agents_half, n_agents):
        op = (
            homophily_threshold
            + 0.5 * (i - n_agents_half) / n_agents * homophily_threshold
        )
        simulation.network.agent[i].data.opinion = op

    # The two halves are so far apart that they should not interact an therefore form two stable clusters.
    simulation.run(output_dir_path)

    for i in range(0, n_agents_half):
        assert simulation.network.agent[i].data.opinion == pytest.approx(avg_opinion)

    for i in range(n_agents_half, n_agents):
        assert simulation.network.agent[i].data.opinion == pytest.approx(-avg_opinion)


# Test the multi-dimensional Deffuant vector model, with 3-dimensional binary opinions, for two agents
def test_deffuant_vector_model():
    proj_root = Path.cwd()
    input_file = str(proj_root / "tests" / "res" / "deffuant_vector_2agents.toml")
    options = pyseldon.seldoncore.parse_config_file(input_file)

    simulation = pyseldon.seldoncore.SimulationDiscreteVector(options=options)

    output_dir_path = str(proj_root / "tests" / "output_deffuant_vector")

    # agents are too far apart, we dont expect any change with the iterations
    agent1_init = [0, 1, 0]
    agent2_init = [1, 0, 1]

    simulation.network.agent[0].data.opinion = agent1_init
    simulation.network.agent[1].data.opinion = agent2_init

    simulation.run(output_dir_path)

    assert simulation.network.agent[0].data.opinion == agent1_init
    assert simulation.network.agent[1].data.opinion == agent2_init

    # agents are close enough, they should converge
    # dim-1 or 2 opinions should be the same
    agent1_init = [0, 1, 1]
    agent2_init = [1, 1, 1]

    simulation.network.agent[0].data.opinion = agent1_init
    simulation.network.agent[1].data.opinion = agent2_init

    simulation.run(output_dir_path)

    assert (
        simulation.network.agent[0].data.opinion
        == simulation.network.agent[1].data.opinion
    )


if __name__ == "__main__":
    pytest.main([__file__])
