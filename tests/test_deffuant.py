import pyseldonlib
import pytest
from pathlib import Path
import shutil


def test_basic_deffuant_model_two_agents():
    proj_root = Path.cwd()
    other_settings = pyseldonlib.Other_Settings(
        number_of_agents=2, connections_per_agent=0
    )
    model = pyseldonlib.Deffuant_Model(
        max_iterations=10,
        homophily_threshold=0.2,
        mu=0.5,
        other_settings=other_settings,
    )
    output_dir_path = str(proj_root / "tests" / "output_deffuant")
    mu = model.mu
    homophily_threshold = model.homophily_threshold

    # agents are too far apart, we dont expect any change with the iterations
    agent1_init = homophily_threshold * 1.1
    agent2_init = 0

    model.set_agent_opinion(0, agent1_init)
    model.set_agent_opinion(1, agent2_init)

    model.run(output_dir_path)

    assert model.agent_opinion(0) == pytest.approx(agent1_init)
    assert model.agent_opinion(1) == pytest.approx(agent2_init)

    # agents are too close, we expect them to converge to the same opinion
    agent1_init = homophily_threshold * 0.9
    agent2_init = 0

    model.set_agent_opinion(0, agent1_init)
    model.set_agent_opinion(1, agent2_init)

    shutil.rmtree(output_dir_path)
    model.run(output_dir_path)

    n_iterations = model.max_iterations
    expected_diff = (1.0 - 2.0 * mu) ** (2 * n_iterations) * (agent1_init - agent2_init)
    assert model.agent_opinion(0) - model.agent_opinion(1) == pytest.approx(
        expected_diff
    )
    shutil.rmtree(output_dir_path)


def test_lattice_deffuant_model_16X16_agents():
    proj_root = Path.cwd()
    other_settings = pyseldonlib.Other_Settings(
        number_of_agents=256, connections_per_agent=0
    )
    model = pyseldonlib.Deffuant_Model(
        max_iterations=10000,
        homophily_threshold=1.0,
        mu=0.5,
        use_network=True,
        other_settings=other_settings,
    )

    output_dir_path = str(proj_root / "tests" / "output_deffuant")
    homophily_threshold = model.homophily_threshold

    n_agents = model.Network.n_agents()
    n_agents_half = int(n_agents / 2)
    avg_opinion = 0

    # first half with low opinions
    for i in range(0, n_agents_half):
        op = -homophily_threshold - 0.5 * i / n_agents * homophily_threshold
        avg_opinion += op / float(n_agents_half)
        model.set_agent_opinion(i, op)

    # second half with low opinions
    for i in range(n_agents_half, n_agents):
        op = (
            homophily_threshold
            + 0.5 * (i - n_agents_half) / n_agents * homophily_threshold
        )
        model.set_agent_opinion(i, op)

    # The two halves are so far apart that they should not interact an therefore form two stable clusters.
    model.run(output_dir_path)

    for i in range(0, n_agents_half):
        assert model.agent_opinion(i) == pytest.approx(avg_opinion)

    for i in range(n_agents_half, n_agents):
        assert model.agent_opinion(i) == pytest.approx(-avg_opinion)

    shutil.rmtree(output_dir_path)


# Test the multi-dimensional Deffuant vector model, with 3-dimensional binary opinions, for two agents
def test_deffuant_vector_model():
    proj_root = Path.cwd()
    other_settings = pyseldonlib.Other_Settings(
        number_of_agents=2, connections_per_agent=0
    )
    model = pyseldonlib.Deffuant_Vector_Model(
        max_iterations=10,
        homophily_threshold=2,
        mu=0.5,
        use_network=False,
        dim=3,
        other_settings=other_settings,
    )

    output_dir_path = str(proj_root / "tests" / "output_deffuant_vector")

    # agents are too far apart, we dont expect any change with the iterations
    agent1_init = [0, 1, 0]
    agent2_init = [1, 0, 1]

    model.set_agent_opinion(0, agent1_init)
    model.set_agent_opinion(1, agent2_init)

    model.run(output_dir_path)

    assert model.agent_opinion(0) == agent1_init
    assert model.agent_opinion(1) == agent2_init

    # agents are close enough, they should converge
    # dim-1 or 2 opinions should be the same
    agent1_init = [0, 1, 1]
    agent2_init = [1, 1, 1]

    model.set_agent_opinion(0, agent1_init)
    model.set_agent_opinion(1, agent2_init)

    shutil.rmtree(output_dir_path)
    model.run(output_dir_path)

    assert model.agent_opinion(0) == model.agent_opinion(1)
    shutil.rmtree(output_dir_path)


if __name__ == "__main__":
    pytest.main([__file__])
