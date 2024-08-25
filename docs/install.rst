Installation
=============

Python Package Index (PYPI)
---------------------------
You can install the package from PYPI using `pip`:

.. code-block:: bash

    $ pip install pyseldonlib


From Source
--------------

Create and Activate the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new `micromamba` environment and activate it:

.. code-block:: bash

  $ micromamba create -f environment.yml

  $ micromamba activate pyseldonenv


If you get problem with micromamba can look at the installation documentation `here <https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>`_

Build
~~~~~
Set up the build directory with `meson`:

.. code-block:: bash

   $ meson setup build


Install Python package
~~~~~~~~~~~~~~~~~~~~~~~
Use `pip` to install the package:

.. code-block:: bash

  $ pip install .

Run the tests
~~~~~~~~~~~~~
Use `pytest` to run the tests:

.. code-block:: bash

  $ pytest tests

or

.. code-block:: bash
  
  $ python -s pytest tests