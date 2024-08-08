# pyseldon
<img src="./res/logotext.png">

## Python bindings for the Seldon framework

This work is a part of Google Summer of Code 2024 under the Python Software Organisation. The aim of this project is to create Python bindings for the Seldon framework. The Seldon framework is a C++ Engine for Opinion Dynamics. The bindings and the complete project as a python package will allow users to use the Seldon framework in Python and carry out Simulations.

Technologies/ Tools used:
- C++
- Python
- Meson
- Pybind11
- Pytest
- Sphinx
- Micromamba
- Github Actions

### Steps to Install and Compile the code

### Installation
Create a new `micromamba` environment and activate it:
```bash
micromamba create -f environment.yml
micromamba activate pyseldonenv
```
If you get problem with micromamba can look at the installation documentation [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

### Build
Set up the build directory with `meson`:
```bash
meson setup build
```

### Install Python package
Use `pip` to install the package:
```bash
pip install .
```

### Run the tests
Use `pytest` to run the tests:
```bash
pytest tests/
```
 or

 ```bash
pytest -s tests/
```
to see the output of the tests without capturing the std output