
import pyseldonlib
other_settings = pyseldonlib.Other_Settings(n_output_agents=10,
                                         n_output_network= None, 
                                         print_progress= True, 
                                         output_initial=True, 
                                         start_output=1, 
                                         number_of_agents = 20, 
                                         connections_per_agent = 10)

model = pyseldonlib.DeGroot_Model(max_iterations=1000,
                               convergence_tol=1e-6,
                               rng_seed=120, 
                               other_settings=other_settings)

model.run('./ouput_20_agents')