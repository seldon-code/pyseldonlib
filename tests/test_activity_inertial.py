import pyseldon
import pathlib
import pytest
import cmath


# Test the probabilistic inertial activity driven model with one bot and one agent
def test_inertial1Bot1Agent():
    proj_root_path = pathlib.Path.cwd()
    input_file = str(proj_root_path / "tests" / "res" / "1bot_1agent_inertial.toml")

    options = pyseldon.seldoncore.parse_config_file(input_file)
    simulation = pyseldon.seldoncore.SimulationInertialAgent(options=options)

    output_dir_path = str(proj_root_path / "tests" / "output_inertial")

    # Get the bot opinion (which won't change)
    bot = simulation.network.agent[0]
    x_bot = bot.data.opinion  # bot opinion

    # Get the initial agent opinion
    agent = simulation.network.agent[1]
    x_0 = agent.data.opinion  # agent opinion

    simulation.run(output_dir_path)

    model_settings = options.model_settings
    K = model_settings.K
    alpha = model_settings.alpha
    iterations = model_settings.max_iterations
    dt = model_settings.dt
    mu = model_settings.friction_coefficient
    time_elapsed = iterations * dt

    # Final agent and bot opinions after the simulation run
    x_t = agent.data.opinion
    x_t_bot = bot.data.opinion
    reluctance = agent.data.reluctance

    # The bot opinion should not change during the simulation
    assert x_t_bot == pytest.approx(x_bot, abs=1e-16)

    # C = K/m tanh (alpha*x_bot)
    C = K / reluctance * cmath.tanh(alpha * x_bot)

    a1 = 0.5 * (-cmath.sqrt(mu * mu - 4.0) - mu)
    a2 = 0.5 * (cmath.sqrt(mu * mu - 4.0) - mu)
    c1 = (x_0 - C) / (1.0 - a1 / a2)
    c2 = -c1 * a1 / a2

    # Test that the agent opinion matches the analytical solution for an agent with a bot
    # Analytical solution is
    # x_t = c1 * exp(a1*t) + c2 *exp(a2*t) + C
    x_t_analytical = (
        c1 * cmath.exp(a1 * time_elapsed) + c2 * cmath.exp(a2 * time_elapsed) + C
    )

    assert x_t == pytest.approx(x_t_analytical, 1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
