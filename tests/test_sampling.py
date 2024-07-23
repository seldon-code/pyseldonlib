import math
import pytest
import random
import pyseldon.seldoncore as Seldon


def compute_p(k, n):
    if k == 0:
        return 0.0
    else:
        p = 1.0 / (float(n) - 1.0)
        return p + (1.0 - p) * compute_p(k - 1, n - 1)


def test_draw_unique_k_from_n():
    N_RUNS = 10000
    k = 6
    n = 100
    ignore_idx = 11

    histogram = [0] * n  # Count how often each element occurs amongst all samples

    for _ in range(N_RUNS):
        buffer = Seldon.draw_unique_k_from_n(ignore_idx, k, n)
        for num in buffer:
            histogram[num] += 1

    p = compute_p(k, n)
    mean = N_RUNS * p
    sigma = math.sqrt(N_RUNS * p * (1.0 - p))

    assert histogram[ignore_idx] == 0  # The ignore_idx should never be selected

    number_outside_three_sigma = sum(
        1 for count in histogram if 0 < abs(float(count) - mean) > 3.0 * sigma
    )

    for count in histogram:
        if count == 0:
            continue
        assert abs(count - mean) <= 5 * sigma

    if number_outside_three_sigma > 0.01 * N_RUNS:
        pytest.warns(
            UserWarning,
            f"Many deviations beyond the 3 sigma range. {number_outside_three_sigma} out of {N_RUNS}",
        )


def test_weighted_reservoir_sampling():
    N_RUNS = 10000
    k = 6
    n = 100
    ignore_idx = 11
    ignore_idx2 = 29

    histogram = [0] * n  # Count how often each element occurs amongst all samples

    def weight_callback(idx):
        if idx == ignore_idx or idx == ignore_idx2:
            return 0.0
        else:
            return abs(n / 2.0 - idx)

    for _ in range(N_RUNS):
        buffer = Seldon.reservoir_sampling_A_ExpJ(k, n, weight_callback)
        for num in buffer:
            histogram[num] += 1

    assert histogram[ignore_idx] == 0  # The ignore_idx should never be selected
    assert histogram[ignore_idx2] == 0  # The ignore_idx2 should never be selected
