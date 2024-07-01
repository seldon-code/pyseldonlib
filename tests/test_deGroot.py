import pyseldon.seldoncore as pd
import math
import pytest

def test_deGroot():
  n_agents = 2 
  neighbour_list = [[1,0], [0,1]]
  weight_list = [[0.2, 0.8], [0.2, 0.8]]

  settings = pd.DeGrootSettings()
  settings.max_iterations = 100
  settings.convergence_tol=1e-6

  network = pd.SimpleAgentNetwork(neighbour_list = neighbour_list, weight_list = weight_list, direction = "Incoming")

  model = pd.DeGrootModel(settings = settings, network = network)

  network.agents[0].data.opinion = 0.0
  network.agents[1].data.opinion = 1.0

  while not model.finished() :
    model.iteration()

  print(f"N_iterations = {model.n_iterations()} (with convergence_tol {settings.convergence_tol})")

  for i in range(n_agents):
    assert math.isclose(network.agents[i].data.opinion, 0.5, abs_tol=settings.convergence_tol * 10.0)


if __name__ == "__main__":
  pytest.main([__file__])