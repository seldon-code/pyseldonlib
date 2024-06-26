import pyseldon.seldoncore as pd

opset = pd.OutputSettings(n_output_network = 20,n_output_agents = 1, print_progress = True, output_initial = True, start_output = 1, start_numbering_from = 0)
degroot_settings = pd.DeGrootSettings(max_iterations = 20, convergence_tol = 1e-3)
net_settings = pd.InitialNetworkSettings(n_agents = 300, n_connections = 10)
sim_options = pd.SimulationOptions(model_string = "DeGroot", model_settings = degroot_settings, output_settings = opset, network_settings = net_settings)

pd.run_simulation(options = sim_options)