import pyseldonlib
import pathlib
import pytest
import cmath
import shutil


# Test the probabilistic inertial activity driven model with one bot and one agent
def test_inertial1Bot1Agent():
    proj_root_path = pathlib.Path.cwd()
    other_settings = pyseldonlib.Other_Settings(
        print_progress=False, number_of_agents=2, connections_per_agent=1
    )
    model = pyseldonlib.Inertial_Model(
        max_iterations=1000,
        dt=0.001,
        m=1,
        eps=1,
        gamma=2.1,
        reciprocity=1,
        homophily=0.5,
        alpha=1.5,
        K=2.0,
        mean_activities=False,
        mean_weights=False,
        use_reluctances=True,
        reluctance_mean=1.5,
        reluctance_sigma=0.1,
        reluctance_eps=0.01,
        n_bots=1,
        bot_m=[1],
        bot_homophily=[0.7],
        bot_activity=[1.0],
        bot_opinion=[2],
        friction_coefficient=0.5,
        other_settings=other_settings,
        rng_seed=120,
    )
    output_dir_path = str(proj_root_path / "tests" / "output_inertial")

    # Get the bot opinion (which won't change)
    x_bot = model.agent_opinion(0)  # bot opinion

    # Get the initial agent opinion
    x_0 = model.agent_opinion(1)  # agent opinion

    model.run(output_dir_path)

    K = model.K
    alpha = model.alpha
    iterations = model.max_iterations
    dt = model.dt
    mu = model.friction_coefficient
    time_elapsed = iterations * dt

    # Final agent and bot opinions after the simulation run
    x_t = model.agent_opinion(1)
    x_t_bot = model.agent_opinion(0)
    reluctance = model.agent_reluctance(1)

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
    shutil.rmtree(output_dir_path)


if __name__ == "__main__":
    pytest.main([__file__])
