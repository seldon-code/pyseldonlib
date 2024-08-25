import pyseldonlib
import pathlib
import pytest
import shutil
import math


def test_activity_driven():
    # using ./tests/res/activity_probabilistic_conf.toml

    other_settings = pyseldonlib.Other_Settings(
        number_of_agents=1000,
        connections_per_agent=10,
        n_output_agents=1,
        print_progress=False,
    )

    model = pyseldonlib.Activity_Driven_Model(
        max_iterations=20,
        dt=0.01,  # Timestep for the integration of the coupled ODEs
        m=10,  # Number of agents contacted, when the agent is active
        eps=0.01,  # Minimum activity epsilon; a_i belongs to [epsilon,1]
        gamma=2.1,  # Exponent of activity power law distribution of activities
        reciprocity=0.5,  # Probability that when agent i contacts j via weighted reservoir sampling, j also sends feedback to i. So every agent can have more than m incoming connections
        homophily=0.5,  # aka beta. if zero, agents pick their interaction partners at random
        alpha=3.0,  # Controversialness of the issue, must be greater than 0.
        K=3.0,  # Social interaction strength
        mean_activities=False,  # Use the mean value of the powerlaw distribution for the activities of all agents
        mean_weights=False,  # Use the meanfield approximation of the network edges
        other_settings=other_settings,
        rng_seed=120,
    )

    base_dir = pathlib.Path(__file__).parent.resolve()
    print(base_dir)
    output_dir = str(base_dir / "outputs")

    if pathlib.Path(output_dir).exists():
        shutil.rmtree(output_dir)

    # By using the above settings
    model.run(output_dir=output_dir)
    print("Simulation completed!")
    assert pathlib.Path(output_dir).exists()
    shutil.rmtree(output_dir)

    # By using a config file
    config_file_path = str(base_dir / "res/activity_probabilistic_conf.toml")
    pyseldonlib.run_simulation_from_config_file(
        config_file_path=config_file_path, output_dir_path=output_dir
    )
    assert pathlib.Path(output_dir).exists()
    shutil.rmtree(output_dir)


# Test that you can produce output for the probabilistic acitivity driven model, from a conf file",
def test_activityProb():
    proj_root = pathlib.Path.cwd()
    input_file = str(proj_root / "tests" / "res" / "activity_probabilistic_conf.toml")
    options = pyseldonlib.parse_config_file(input_file)

    output_dir_path = proj_root / "tests" / "output"

    if output_dir_path.exists():
        shutil.rmtree(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    simulation = pyseldonlib.seldoncore.SimulationActivityAgent(options=options)
    simulation.run(str(output_dir_path))

    assert any(output_dir_path.iterdir()), "Output directory is empty after simulation."

    shutil.rmtree(output_dir_path)


# Test the probabilistic activity driven model for two agents
def test_activityProbTwoAgents():
    proj_root = pathlib.Path.cwd()

    output_dir_path = str(proj_root / "tests" / "output")

    other_settings = pyseldonlib.Other_Settings(
        number_of_agents=2, connections_per_agent=1, print_progress=False
    )

    model = pyseldonlib.Activity_Driven_Model(
        max_iterations=10000,
        dt=0.005,  # Timestep for the integration of the coupled ODEs
        m=1,  # Number of agents contacted, when the agent is active
        eps=1,  # Minimum activity epsilon; a_i belongs to [epsilon,1]
        gamma=2.1,  # Exponent of activity power law distribution of activities
        reciprocity=1,  # Probability that when agent i contacts j via weighted reservoir sampling, j also sends feedback to i. So every agent can have more than m incoming connections
        homophily=0.5,  # aka beta. if zero, agents pick their interaction partners at random
        alpha=1.01,  # Controversialness of the issue, must be greater than 0.
        K=2.0,  # Social interaction strength
        mean_activities=False,  # Use the mean value of the powerlaw distribution for the activities of all agents
        mean_weights=False,  # Use the meanfield approximation of the network edges
        other_settings=other_settings,
        rng_seed=120,
    )
    model.run(output_dir_path)

    K = model.K
    alpha = model.alpha

    assert K == pytest.approx(2.0, 1e-16)
    assert alpha == pytest.approx(1.01, 1e-16)

    # This is the solution of x = K tanh(alpha x)
    analytical_x = 1.9187384098662013
    assert analytical_x == 1.9187384098662013

    for idx in range(0, model.Network.n_agents()):
        assert model.agent_opinion(idx) == pytest.approx(analytical_x, abs=1e-4)

    shutil.rmtree(output_dir_path)


# Test the probabilistic activity driven model with one bot and one (reluctant) agent
def test_activity1Bot1AgentReluctance():
    proj_root = pathlib.Path.cwd()

    other_settings = pyseldonlib.Other_Settings(
        number_of_agents=2, connections_per_agent=1, print_progress=False
    )

    model = pyseldonlib.Activity_Driven_Model(
        max_iterations=1000,
        dt=0.001,  # Timestep for the integration of the coupled ODEs
        m=1,  # Number of agents contacted, when the agent is active
        eps=1,  # Minimum activity epsilon; a_i belongs to [epsilon,1]
        gamma=2.1,  # Exponent of activity power law distribution of activities
        reciprocity=1,  # Probability that when agent i contacts j via weighted reservoir sampling, j also sends feedback to i. So every agent can have more than m incoming connections
        homophily=0.5,  # aka beta. if zero, agents pick their interaction partners at random
        alpha=1.5,  # Controversialness of the issue, must be greater than 0.
        K=2.0,  # Social interaction strength
        mean_activities=False,  # Use the mean value of the powerlaw distribution for the activities of all agents
        mean_weights=False,  # Use the meanfield approximation of the network edges
        use_reluctances=True,  # Assigns a "reluctance" (m_i) to each agent. By default; false and every agent has a reluctance of 1
        reluctance_mean=1.5,  # Mean of distribution before drawing from a truncated normal distribution (default set to 1.0)
        reluctance_sigma=0.1,  # Width of normal distribution (before truncating)
        reluctance_eps=0.01,  # Minimum such that the normal distribution is truncated at this value
        n_bots=1,  # The number of bots to be used; if not specified defaults to 0 (which means bots are deactivated)
        # Bots are agents with fixed opinions and different parameters, the parameters are specified in the following lists
        # If n_bots is smaller than the length of any of the lists, the first n_bots entries are used. If n_bots is greater the code will throw an exception.
        bot_m=[1],  # If not specified, defaults to `m`
        bot_homophily=[0.7],  # If not specified, defaults to `homophily`
        bot_activity=[1.0],  # If not specified, defaults to 0
        bot_opinion=[2],  # The fixed opinions of the bots
        other_settings=other_settings,
        rng_seed=120,
    )

    output_dir_path = str(proj_root / "tests" / "output1bot")

    # Get the bot opinion (which won't change)
    x_bot = model.agent_opinion(0)

    # Get the initial agent opinion
    x_0 = model.agent_opinion(1)

    model.run(output_dir_path)

    K = model.K
    alpha = model.alpha
    iterations = model.max_iterations
    dt = model.dt
    time_elapsed = iterations * dt

    # final agent and bot opinion
    x_t = model.agent_opinion(1)
    x_t_bot = model.agent_opinion(0)
    reluctance = model.agent_reluctance(1)

    # The bot opinion should not change during the simulation
    assert x_t_bot == pytest.approx(x_bot, abs=1e-16)

    # Test that the agent opinion matches the analytical solution for an agent with a bot
    # Analytical solution is:
    # x_t = [x(0) - Ktanh(alpha*x_bot)]e^(-t) + Ktanh(alpha*x_bot)
    x_t_analytical = (x_0 - K / reluctance * math.tanh(alpha * x_bot)) * math.exp(
        -time_elapsed
    ) + K / reluctance * math.tanh(alpha * x_bot)

    assert x_t == pytest.approx(x_t_analytical, abs=1e-5)

    shutil.rmtree(output_dir_path)


# Test the meanfield activity driven model with 10 agents
@pytest.mark.xfail(reason="Test is not passing, future work needed")
def test_activityMeanfield10Agents():
    proj_root = pathlib.Path.cwd()

    other_settings = pyseldonlib.Other_Settings(
        number_of_agents=50, connections_per_agent=1, print_progress=False
    )

    model = pyseldonlib.Activity_Driven_Model(
        max_iterations=1000,
        dt=0.01,  # Timestep for the integration of the coupled ODEs
        m=10,  # Number of agents contacted, when the agent is active
        eps=0.01,  # Minimum activity epsilon; a_i belongs to [epsilon,1]
        gamma=2.1,  # Exponent of activity power law distribution of activities
        reciprocity=1.0,  # Probability that when agent i contacts j via weighted reservoir sampling, j also sends feedback to i. So every agent can have more than m incoming connections
        homophily=0.0,  # aka beta. if zero, agents pick their interaction partners at random
        alpha=1.01,  # Controversialness of the issue, must be greater than 0.
        K=2.0,  # Social interaction strength
        mean_activities=True,  # Use the mean value of the powerlaw distribution for the activities of all agents
        mean_weights=True,  # Use the meanfield approximation of the network edges
        other_settings=other_settings,
    )

    output_dir_path = str(proj_root / "tests" / "output")

    assert model.mean_weights == True
    assert model.mean_activities == True
    # We require zero homophily, since we only know the critical controversialness in that case
    assert model.homophily == pytest.approx(0, abs=1e-16)

    K = model.K
    n_agents = model.Network.n_agents()
    reciprocity = model.reciprocity
    m = model.m
    eps = model.eps
    gamma = model.gamma

    dist = pyseldonlib.seldoncore.Power_Law_Distribution(eps, gamma)
    mean_activity = dist.mean()
    print(mean_activity)
    print(type(mean_activity))

    def set_opinions_and_run(above_critical_controversialness):
        initial_opinion_delta = 0.1  # Set the initial opinion in the interval [-delta, delta]

        for i in range(0, n_agents):
            _model = model
            assert mean_activity == pytest.approx(_model.agent_activity(i), abs=1e-16)
            opinion = -initial_opinion_delta + 2.0 * i / (n_agents - 1) * (
                initial_opinion_delta
            )
            _model.set_agent_opinion(i, opinion)

        output_dir_p = str(proj_root / "tests" / "output_meanfield_test")
        if pathlib.Path(output_dir_p).exists():
            shutil.rmtree(output_dir_p)

        _model.run(output_dir_p)

        # Check the opinions after the run, if alpha is above the critical controversialness,
        # the opinions need to deviate from zero
        avg_deviation = 0.0
        for i in range(0, n_agents):
            if above_critical_controversialness is False:
                assert abs(_model.agent_opinion(i)) > abs(initial_opinion_delta)
            else:
                assert abs(_model.agent_opinion(i)) < abs(initial_opinion_delta)

            avg_deviation += abs(_model.agent_opinion(i))
        print(f"Average deviation of agents = { avg_deviation / n_agents}\n")
        shutil.rmtree(output_dir_p)

    alpha_critical = (
        float(n_agents)
        / (float(n_agents) - 1.0)
        * 1.0
        / ((1.0 + reciprocity) * K * m * mean_activity)
    )
    print(f"Critical controversialness = { alpha_critical}\n")
    delta_alpha = 0.1

    # Set the critical controversialness to a little above the critical alpha
    model.alpha = alpha_critical + delta_alpha
    set_opinions_and_run(True)

    # Set the critical controversialness to a little above the critical alpha
    model.alpha = alpha_critical - delta_alpha
    set_opinions_and_run(False)
    shutil.rmtree(output_dir_path)

if __name__ == "__main__":
    pytest.main([__file__])
