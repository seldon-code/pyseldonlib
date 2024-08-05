import pytest
import pyseldon


def test_hamming_distance():
    v1 = [1, 1, 1, 0, 1]
    v2 = [0, 1, 1, 0, 0]

    dist = pyseldon.seldoncore.hamming_distance(v1, v2)

    assert dist == 2
