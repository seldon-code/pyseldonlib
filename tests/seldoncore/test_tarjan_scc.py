import pytest
import pyseldonlib

# Test data similar to the C++ version
neighbour_list = [[1], [2, 3], [0], [4], [5], [4], [4, 7], [5, 8], [9], [6, 7]]

# Expected SCCs
expected_scc = [{5, 4}, {3}, {2, 1, 0}, {9, 8, 7, 6}]


def test_tarjan_scc():
    # Run Tarjan's algorithm
    tarjan_scc = pyseldonlib.seldoncore.TarjanConnectivityAlgo(neighbour_list)

    # Convert each SCC list to a set for comparison
    scc_sets = [set(scc) for scc in tarjan_scc.scc_list]

    # Check if all expected SCCs are found
    for expected_set in expected_scc:
        assert expected_set in scc_sets

    # Check the total number of SCCs
    assert len(tarjan_scc.scc_list) == 4
