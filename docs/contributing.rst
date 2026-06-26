============
Contributing
============

Thank you for your interest in contributing to ocpy! This document provides guidelines
and information for contributors.

Getting Started
---------------

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/ocpy.git
      cd ocpy

3. **Create a virtual environment**:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # Linux/macOS
      # or: venv\Scripts\activate  # Windows

4. **Install in development mode**:

   .. code-block:: bash

      pip install -e .
      pip install pytest pytest-cov

5. **Create a branch** for your changes:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

Development Guidelines
----------------------

Code Style
^^^^^^^^^^

* Follow PEP 8 style guidelines
* Use meaningful variable and function names
* Keep functions focused and reasonably sized
* Add docstrings to all public functions and classes

Docstring Format
^^^^^^^^^^^^^^^^

Use NumPy-style docstrings:

.. code-block:: python

   def calculate_absorption(wavelengths, concentration, coefficient='IOCCG'):
       """Calculate absorption at given wavelengths.

       Parameters
       ----------
       wavelengths : array-like
           Wavelengths in nanometers.
       concentration : float
           Concentration in mg/m³.
       coefficient : str, optional
           Coefficient source ('IOCCG' or 'GSFC'). Default is 'IOCCG'.

       Returns
       -------
       absorption : ndarray
           Absorption coefficients in m⁻¹.

       Raises
       ------
       ValueError
           If wavelengths are outside valid range.

       Examples
       --------
       >>> wavelengths = np.array([443, 490, 555])
       >>> abs_coef = calculate_absorption(wavelengths, 1.0)

       Notes
       -----
       Based on the model from Smith et al. (2020).

       References
       ----------
       .. [1] Smith, J. et al. (2020). Journal of Ocean Optics.
       """
       pass

Testing
^^^^^^^

* Write tests for all new functionality
* Ensure existing tests pass before submitting
* Aim for good test coverage

.. code-block:: bash

   # Run all tests
   pytest ocpy/tests/

   # Run with coverage report
   pytest --cov=ocpy ocpy/tests/

   # Run specific test file
   pytest ocpy/tests/test_water.py

Submitting Changes
------------------

1. **Commit your changes** with clear messages:

   .. code-block:: bash

      git add .
      git commit -m "Add feature: description of changes"

2. **Push to your fork**:

   .. code-block:: bash

      git push origin feature/your-feature-name

3. **Open a Pull Request** on GitHub with:

   * Clear title describing the change
   * Description of what was changed and why
   * Reference to any related issues

Pull Request Checklist
^^^^^^^^^^^^^^^^^^^^^^

Before submitting, ensure:

* [ ] Code follows project style guidelines
* [ ] All tests pass
* [ ] New functionality has tests
* [ ] Documentation is updated if needed
* [ ] Docstrings are complete
* [ ] No debug code or print statements

Types of Contributions
----------------------

Bug Reports
^^^^^^^^^^^

When reporting bugs, please include:

* Python version
* ocpy version
* Operating system
* Minimal code example that reproduces the issue
* Full error traceback

Feature Requests
^^^^^^^^^^^^^^^^

For new features:

* Describe the use case
* Explain the expected behavior
* Consider potential implementation approaches

Code Contributions
^^^^^^^^^^^^^^^^^^

We welcome contributions including:

* Bug fixes
* New algorithms
* Additional datasets
* Documentation improvements
* Test coverage improvements

Adding New Algorithms
---------------------

When adding new ocean color algorithms:

1. **Research**: Ensure the algorithm is well-documented in literature
2. **Implementation**: Follow existing code patterns
3. **Validation**: Test against reference implementations if available
4. **Documentation**: Add docstrings with references

Example structure for a new module:

.. code-block:: python

   """
   Module description.

   This module implements the XYZ algorithm from Author et al. (Year).

   References
   ----------
   Author, A. et al. (Year). Title. Journal, Volume, Pages.
   """

   import numpy as np

   def main_function(wavelengths, Rrs):
       """Main algorithm function.

       Parameters
       ----------
       wavelengths : ndarray
           Wavelengths in nm.
       Rrs : ndarray
           Remote sensing reflectance in sr⁻¹.

       Returns
       -------
       result : ndarray
           Computed result.
       """
       # Implementation
       pass

Adding Reference Data
---------------------

When adding new reference datasets:

1. Place data files in ``ocpy/data/`` in appropriate subdirectory
2. Use ``importlib.resources`` for loading:

   .. code-block:: python

      from importlib import resources

      def load_data():
          with resources.files('ocpy.data.subdir').joinpath('file.csv').open() as f:
              data = pd.read_csv(f)
          return data

3. Document the data source and any preprocessing
4. Include citation information

Code of Conduct
---------------

* Be respectful and inclusive
* Provide constructive feedback
* Focus on the technical aspects of contributions
* Acknowledge the work of others

Questions?
----------

* Open an issue on GitHub for questions
* Email the maintainers for sensitive matters

Thank you for contributing to ocpy!
