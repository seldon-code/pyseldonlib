import pathlib as ptlb
import pytest
import pyseldonlib
import shutil


def test_run_simulation(capsys):
    # Set up the paths
    base_dir = ptlb.Path(__file__).parent.parent.resolve()
    config_file = str(base_dir / "res/config/conf.toml")
    network_file = str(base_dir / "res/network/network.txt")
    invalid_config_file = str(base_dir / "res/config/inconf.txt")
    output_dir1 = str("outputs/outputfile")
    output_dir2 = str("outputs/opwithnetwork")
    output_dir = str("outputs/output")

    if ptlb.Path(output_dir1).exists():
        shutil.rmtree(output_dir1)
    if ptlb.Path(output_dir2).exists():
        shutil.rmtree(output_dir2)

    # Test with output directory and config
    with capsys.disabled():
        pyseldonlib.seldoncore.run_simulation(
            config_file_path=config_file, output_dir_path=output_dir1
        )
    assert ptlb.Path(output_dir1).exists()
    assert ptlb.Path(output_dir1).is_dir()

    # Test with network file
    with capsys.disabled():
        pyseldonlib.seldoncore.run_simulation(
            config_file_path=config_file,
            network_file_path=network_file,
            output_dir_path=output_dir2,
        )
    assert ptlb.Path(output_dir2).exists()
    assert ptlb.Path(output_dir2).is_dir()

    # Test with non-existent network file
    invalid_network_file = str(ptlb.Path(base_dir, "tests/res/network/net.txt"))
    with pytest.raises(RuntimeError):
        with capsys.disabled():
            pyseldonlib.seldoncore.run_simulation(
                config_file_path=config_file, network_file_path=invalid_network_file
            )

    # Test with invalid config file
    with pytest.raises(RuntimeError):
        with capsys.disabled():
            pyseldonlib.seldoncore.run_simulation(config_file_path=invalid_config_file)

    if ptlb.Path(output_dir).exists():
        shutil.rmtree(output_dir)
    if ptlb.Path(output_dir1).exists():
        shutil.rmtree(output_dir1)
    if ptlb.Path(output_dir2).exists():
        shutil.rmtree(output_dir2)


def test_settings():
    degroot_settings = pyseldonlib.seldoncore.DeGrootSettings()
    output_settings = pyseldonlib.seldoncore.OutputSettings()
    deffuant_settings = pyseldonlib.seldoncore.DeffuantSettings()
    activitydriven_settings = pyseldonlib.seldoncore.ActivityDrivenSettings()
    activitydriveninertial_settings = (
        pyseldonlib.seldoncore.ActivityDrivenInertialSettings()
    )
    initial_network_settings = pyseldonlib.seldoncore.InitialNetworkSettings()

    assert degroot_settings is not None
    assert output_settings is not None
    assert deffuant_settings is not None
    assert activitydriven_settings is not None
    assert activitydriveninertial_settings is not None
    assert initial_network_settings is not None
    assert activitydriveninertial_settings.covariance_factor == 0.0


def test_network():
    degroot_network = pyseldonlib.seldoncore.SimpleAgentNetwork()
    deffuant_network = pyseldonlib.seldoncore.SimpleAgentNetwork()
    activitydriven_network = pyseldonlib.seldoncore.ActivityAgentNetwork()
    activitydriveninertial_network = pyseldonlib.seldoncore.InertialAgentNetwork()

    assert degroot_network is not None
    assert deffuant_network is not None
    assert activitydriven_network is not None
    assert activitydriveninertial_network is not None


def test_simulation_with_simulationOptions():
    degroot_settings = pyseldonlib.seldoncore.DeGrootSettings()
    degroot_settings.max_iterations = 100
    output_settings = pyseldonlib.seldoncore.OutputSettings()
    initial_network_settings = pyseldonlib.seldoncore.InitialNetworkSettings()
    simulation_options = pyseldonlib.seldoncore.SimulationOptions()
    simulation_options.output_settings = output_settings
    simulation_options.model_settings = degroot_settings
    simulation_options.network_settings = initial_network_settings
    simulation_options.model_string = "DeGroot"

    base_dir = ptlb.Path(__file__).parent.resolve()
    output_dir = str(base_dir / "outputs/output")

    pyseldonlib.seldoncore.run_simulation(
        options=simulation_options, output_dir_path=output_dir
    )
    assert ptlb.Path(output_dir).exists()
    shutil.rmtree(output_dir)


if __name__ == "__main__":
    pytest.main([__file__])
