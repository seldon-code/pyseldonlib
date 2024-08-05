import pyseldon
import pathlib as ptlb
import pytest
import shutil
import math


def test_activity_driven():
    # using ../subprojects/seldon/test/res/activity_probabilistic_conf.toml
    settings = pyseldon.ActivityDrivenSettings(
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
    )

    initial_network_settings = pyseldon.InitialNetworkSettings(
        number_of_agents=1000, connections_per_agent=10
    )
    output_settings = pyseldon.OutputSettings(
        n_output_agents=1,
        print_progress=True,  # Print the iteration time ; if not set, then does not print
    )

    options = pyseldon.SimulationOptions(
        model_string="ActivityDriven",
        rng_seed=120,
        output_settings=output_settings,
        model_settings=settings,
        network_settings=initial_network_settings,
    )

    base_dir = ptlb.Path(__file__).parent.resolve()
    print(base_dir)
    output_dir = str(base_dir / "outputs")

    if ptlb.Path(output_dir).exists():
        shutil.rmtree(output_dir)

    # By using the above settings
    pyseldon.seldoncore.run_simulation(options=options, output_dir_path=output_dir)
    print("Simulation completed!")
    assert ptlb.Path(output_dir).exists()
    shutil.rmtree(output_dir)

    # By using a config file
    config_file_path = str(base_dir / "res/activity_probabilistic_conf.toml")
    pyseldon.seldoncore.run_simulation(config_file_path=config_file_path, output_dir_path=output_dir)
    assert ptlb.Path(output_dir).exists()
    shutil.rmtree(output_dir)


# Test that you can produce output for the probabilistic acitivity driven model, from a conf file",
def test_activityProb():
    proj_root = ptlb.Path.cwd()
    input_file = str(proj_root / "tests" / "res" / "activity_probabilistic_conf.toml")
    options = pyseldon.seldoncore.parse_config_file(input_file)

    output_dir_path = proj_root / "tests" / "output"

    if output_dir_path.exists():
        shutil.rmtree(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    simulation = pyseldon.seldoncore.SimulationActivityAgent(options=options)
    simulation.run(str(output_dir_path))

    assert any(output_dir_path.iterdir()), "Output directory is empty after simulation."

    shutil.rmtree(output_dir_path)


# Test the probabilistic activity driven model for two agents
def test_activityProbTwoAgents():
    proj_root = ptlb.Path.cwd()
    input_file = str(proj_root / "tests" / "res" / "2_agents_activity_prob.toml")
    options = pyseldon.seldoncore.parse_config_file(input_file)

    output_dir_path = str(proj_root / "tests" / "output")

    simulation = pyseldon.seldoncore.SimulationActivityAgent(options=options)
    simulation.run(output_dir_path)

    model_settings = options.model_settings
    K = model_settings.K
    alpha = model_settings.alpha

    assert K == pytest.approx(2.0, 1e-16)
    assert alpha == pytest.approx(1.01, 1e-16)

    # This is the solution of x = K tanh(alpha x)
    analytical_x = 1.9187384098662013
    assert analytical_x == 1.9187384098662013

    for idx in range(0, simulation.network.n_agents()):
        assert simulation.network.agent[idx].data.opinion == pytest.approx(
            analytical_x, abs=1e-4
        )


# Test the probabilistic activity driven model with one bot and one (reluctant) agent
def test_activity1Bot1AgentReluctance():
    proj_root = ptlb.Path.cwd()
    input_file = str(proj_root / "tests" / "res" / "1bot_1agent_activity_prob.toml")
    options = pyseldon.seldoncore.parse_config_file(input_file)

    simulation = pyseldon.seldoncore.SimulationActivityAgent(options)

    output_dir_path = str(proj_root / "tests" / "output")

    # Get the bot opinion (which won't change)
    bot = simulation.network.agent[0]
    x_bot = bot.data.opinion

    # Get the initial agent opinion
    agent = simulation.network.agent[1]
    x_0 = agent.data.opinion

    simulation.run(output_dir_path)

    model_settings = options.model_settings
    K = model_settings.K
    alpha = model_settings.alpha
    iterations = model_settings.max_iterations
    dt = model_settings.dt
    time_elapsed = iterations * dt

    # final agent and bot opinion
    x_t = agent.data.opinion
    x_t_bot = bot.data.opinion
    reluctance = agent.data.reluctance

    # The bot opinion should not change during the simulation
    assert x_t_bot == pytest.approx(x_bot, abs=1e-16)

    # Test that the agent opinion matches the analytical solution for an agent with a bot
    # Analytical solution is:
    # x_t = [x(0) - Ktanh(alpha*x_bot)]e^(-t) + Ktanh(alpha*x_bot)
    x_t_analytical = (x_0 - K / reluctance * math.tanh(alpha * x_bot)) * math.exp(
        -time_elapsed
    ) + K / reluctance * math.tanh(alpha * x_bot)

    assert x_t == pytest.approx(x_t_analytical, abs=1e-5)


# Test the meanfield activity driven model with 10 agents
def test_activityMeanfield10Agents():
    proj_root = ptlb.Path.cwd()
    input_file = str(proj_root / "tests" / "res" / "10_agents_meanfield_activity.toml")
    options = pyseldon.seldoncore.parse_config_file(input_file)

    model_settings = options.model_settings
    assert model_settings.mean_weights == True
    assert model_settings.mean_activities == True
    # We require zero homophily, since we only know the critical controversialness in that case
    assert model_settings.homophily == pytest.approx(0, abs=1e-16)

    K = model_settings.K
    n_agents = options.network_settings.number_of_agents
    reciprocity = model_settings.reciprocity
    m = model_settings.m
    eps = model_settings.eps
    gamma = model_settings.gamma

    dist = pyseldon.seldoncore.Power_Law_Distribution(eps, gamma)
    mean_activity = dist.mean()

    def set_opinions_and_run(above_critical_controversialness):
        simulation = pyseldon.seldoncore.SimulationActivityAgent(options=options)
        initial_opinion_delta = (
            0.1  # Set the initial opinion in the interval [-delta, delta]
        )

        for i in range(0, n_agents):
            agent = simulation.network.agent[i]
            assert mean_activity == pytest.approx(agent.data.activity, abs=1e-16)
            agent.data.opinion = -initial_opinion_delta + 2.0 * i / (n_agents - 1) * (
                initial_opinion_delta
            )

        output_dir_p = str(proj_root / "tests" / "output_meanfield_test")

        simulation.run(output_dir_p)

        # Check the opinions after the run, if alpha is above the critical controversialness,
        # the opinions need to deviate from zero
        avg_deviation = 0.0
        for i in range(0, n_agents):
            agent = simulation.network.agent[i]
            if above_critical_controversialness is True:
                assert abs(agent.data.opinion) > abs(initial_opinion_delta)
            else:
                assert abs(agent.data.opinion) < abs(initial_opinion_delta)

            avg_deviation += abs(agent.data.opinion)

    alpha_critical = (
        float(n_agents)
        / (float(n_agents) - 1.0)
        * 1.0
        / ((1.0 + reciprocity) * K * m * mean_activity)
    )
    print(f"Critical controversialness = { alpha_critical}\n")
    delta_alpha = 0.1

    # Set the critical controversialness to a little above the critical alpha
    model_settings.alpha = alpha_critical + delta_alpha
    set_opinions_and_run(True)

    # Set the critical controversialness to a little above the critical alpha
    model_settings.alpha = alpha_critical - delta_alpha
    set_opinions_and_run(False)


if __name__ == "__main__":
    pytest.main([__file__])
