import pyseldon.seldoncore as pd
import pathlib as ptlb
import pytest
import shutil


def test_activity_driven():
    # using ../subprojects/seldon/test/res/activity_probabilistic_conf.toml
    settings = pd.ActivityDrivenSettings(
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

    initial_network_settings = pd.InitialNetworkSettings(
        number_of_agents=1000, connections_per_agent=10
    )
    output_settings = pd.OutputSettings(
        n_output_agents=1,
        print_progress=True,  # Print the iteration time ; if not set, then does not print
    )

    options = pd.SimulationOptions(
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
    pd.run_simulation(options=options, output_dir_path=output_dir)
    print("Simulation completed!")
    assert ptlb.Path(output_dir).exists()
    shutil.rmtree(output_dir)

    # By using a config file
    config_file_path = str(base_dir / "res/activity_probabilistic_conf.toml")
    pd.run_simulation(config_file_path=config_file_path, output_dir_path=output_dir)
    assert ptlb.Path(output_dir).exists()
    shutil.rmtree(output_dir)


if __name__ == "__main__":
    pytest.main([__file__])
