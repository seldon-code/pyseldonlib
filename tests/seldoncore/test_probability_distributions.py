import os
from pathlib import Path
import pyseldonlib
import pytest
import random


def write_results_to_file(n_samples, dist, filename):
    proj_root_path = Path.cwd()
    file_path = proj_root_path / f"tests/output_probability_distributions/{filename}"
    print(f"file = {file_path}")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    gen = pyseldonlib.seldoncore.RandomGenerator(random.randint(0, 2**32 - 1))
    results = [dist(gen) for _ in range(n_samples)]

    with open(file_path, "w") as file:
        for value in results:
            file.write(f"{value}\n")


def test_probability_distributions():
    write_results_to_file(
        10000,
        pyseldonlib.seldoncore.Truncated_Normal_Distribution(1.0, 0.5, 0.75),
        "truncated_normal.txt",
    )
    write_results_to_file(
        10000, pyseldonlib.seldoncore.Power_Law_Distribution(0.01, 2.1), "power_law.txt"
    )
    write_results_to_file(
        10000,
        pyseldonlib.seldoncore.Bivariate_Normal_Distribution(0.5),
        "bivariate_normal.txt",
    )


def test_bivariate_gaussian_copula():
    dist1 = pyseldonlib.seldoncore.Power_Law_Distribution(0.02, 2.5)
    dist2 = pyseldonlib.seldoncore.Truncated_Normal_Distribution(1.0, 0.75, 0.2)
    copula = pyseldonlib.seldoncore.Bivariate_Gaussian_Copula(0.5, dist1, dist2)
    write_results_to_file(10000, copula, "gaussian_copula.txt")
