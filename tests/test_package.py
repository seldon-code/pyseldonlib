import pyseldon
import pathlib

# initial_agents = []
# final_agents = []

# pyseldon.run_simulation_from_config_file(config_file_path='/home/parrot_user/Desktop/pyseldon/subprojects/seldon/examples/ActivityDriven/conf.toml',initial_agents=initial_agents,final_agents=final_agents)

# print(initial_agents)

# print("------------------")

# print(final_agents)

sim_options = pyseldon.seldoncore.SimulationOptions(model_string = "DeGroot", model_settings = pyseldon.seldoncore.DeGrootSettings(max_iterations = 20))
# simulation = pyseldon.seldoncore.SimulationDeGroot(sim_options)
# agent = simulation.network.agent[1].data.opinion

# print(agent)
# print(type(agent))

# print(dir(simulation))
# simulation.run(str(pathlib.Path(__file__).parent.resolve() / "output/"))

print(type(pyseldon.seldoncore.SimulationOptions()))


pyseldon.seldoncore.run_simulation(options = sim_options)

# agent = simulation.network.agent[5].data.opinion
# print(agent)

 
