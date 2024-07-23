import os
from pathlib import Path
import pyseldon.seldoncore as Seldon
import pytest

def write_results_to_file(n_samples, dist, filename):
    proj_root_path = Path.cwd()
    file_path = proj_root_path / f"tests/output_probability_distributions/{filename}"
    print(f"file = {file_path}")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Assuming dist function internally manages its random number generation
    results = dist(n_samples)

    with open(file_path, 'w') as filestream:
        for value in results:
            filestream.write(f"{value}\n")

def test_probability_distributions():
    write_results_to_file(10000, Seldon.Truncated_Normal_Distribution(1.0, 0.5, 0.75), "truncated_normal.txt")
    write_results_to_file(10000, Seldon.Power_Law_Distribution(0.01, 2.1), "power_law.txt")
    write_results_to_file(10000, Seldon.Bivariate_Bormal_Distribution(0.5), "bivariate_normal.txt")

def test_bivariate_gaussian_copula():
    dist1 = Seldon.Power_Law_Distribution(0.02, 2.5)
    dist2 = Seldon.Truncated_Normal_Distribution(1.0, 0.75, 0.2)
    copula = Seldon.Bivariate_Gaussian_Copula(0.5, dist1, dist2)