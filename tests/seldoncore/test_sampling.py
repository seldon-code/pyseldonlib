import math
import pytest
import random
import pyseldonlib


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

    histogram = [0] * n
    buffer = []
    gen = pyseldonlib.seldoncore.RandomGenerator(random.randint(0, 2**32 - 1))
    for _i in range(0, N_RUNS):
        pyseldonlib.seldoncore.draw_unique_k_from_n(
            ignore_idx=ignore_idx, k=k, n=n, buffer=buffer, gen=gen
        )
        for num in buffer:
            histogram[num] += 1

    # In each run there is a probability of p for each element to be selected
    # That means for each histogram bin we have a binomial distribution with p
    p = compute_p(k, n)

    mean = N_RUNS * p
    # The variance of a binomial distribution is var = n*p*(1-p)
    sigma = math.sqrt(N_RUNS * p * (1.0 - p))

    assert histogram[ignore_idx] == 0  # The ignore_idx should never be selected

    number_outside_three_sigma = 0
    for n in histogram:
        if n == 0:
            continue

        if abs(float(n) - float(mean)) > 3.0 * sigma:
            number_outside_three_sigma += 1

        assert n == pytest.approx(mean, abs=5 * sigma)

    if number_outside_three_sigma > 0.01 * N_RUNS:
        pytest.warns(
            UserWarning,
            f"Many deviations beyond the 3 sigma range. {number_outside_three_sigma} out of {N_RUNS}",
        )


# to-do
# def test_weighted_reservoir_sampling():
#     N_RUNS = 10000
#     k = 6
#     n = 100
#     ignore_idx = 11
#     ignore_idx2 = 29

#     histogram = [0] * n  # Count how often each element occurs amongst all samples

#     def weight_callback(idx):
#         if idx == ignore_idx or idx == ignore_idx2:
#             return 0.0
#         else:
#             return abs(n / 2.0 - idx)
#     buffer = []
#     gen = pyseldon.seldoncore.RandomGenerator(random.randint(0, 2**32 - 1))

#     for _ in range(N_RUNS):
#         pyseldon.seldoncore.reservoir_sampling_A_ExpJ(k=k, n=n, weight_callback=weight_callback,buffer = buffer,gen = gen)
#         for num in buffer:
#             histogram[num] += 1

#     assert histogram[ignore_idx] == 0  # The ignore_idx should never be selected
#     assert histogram[ignore_idx2] == 0  # The ignore_idx2 should never be selected
