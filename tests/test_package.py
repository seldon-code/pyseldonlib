import pyseldon
import pathlib

# initial_agents = []
# final_agents = []

# pyseldon.run_simulation_from_config_file(config_file_path='/home/parrot_user/Desktop/pyseldon/subprojects/seldon/examples/ActivityDriven/conf.toml',initial_agents=initial_agents,final_agents=final_agents)

# print(initial_agents)

# print("------------------")

# print(final_agents)

sim_options = pyseldon.seldoncore.SimulationOptions(model_string = "ActivityDrivenInertial", model_settings = pyseldon.seldoncore.ActivityDrivenInertialSettings(max_iterations = 20), output_settings = pyseldon.seldoncore.OutputSettings(n_output_agents= 200,n_output_network = 5))
# simulation = pyseldon.seldoncore.SimulationDeGroot(sim_options)
# agent = simulation.network.agent[1].data.opinion

# print(agent)
# print(type(agent))

# print(dir(simulation))
# simulation.run(str(pathlib.Path(__file__).parent.resolve() / "output/"))

# print(type(pyseldon.seldoncore.create_output_settings()))
# print(type(pyseldon.seldoncore.create_output_settings(1,2, True, True,0,0)))

pyseldon.seldoncore.run_simulation(options = sim_options)

# agent = simulation.network.agent[5].data.opinion
# print(agent)

 
