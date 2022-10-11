.. include:: references.txt

.. _overload_numpy-install:

************
Installation
************

With ``pip`` (recommended)
==========================

To install the latest stable version using ``pip``, use

.. code-block:: bash

    python -m pip install overload_numpy

This is the recommended way to install ``overload_numpy``.

To install the development version

.. code-block:: bash

    python -m pip install git+https://github.com/nstarman/overload_numpy


With ``conda``
==============

Conda is not yet supported.


From Source: Cloning, Building, Installing
==========================================

The latest development version of overload_numpy can be cloned from `GitHub
<https://github.com/>`_ using ``git``

.. code-block:: bash

    git clone git://github.com/nstarman/overload_numpy.git

To build and install the project (from the root of the source tree, e.g., inside
the cloned ``overload_numpy`` directory)

.. code-block:: bash

    python -m pip install [-e] .


To ``c``-transpile and build wheels with ``mypyc``.

.. code-block:: bash

    python -m pip install [-e] . --install-option='--use-mypyc"


Python Dependencies
===================

This packages has the following dependencies:

* `Python`_ >= 3.8
* ``mypy_extensions`` >= 0.4.3  : for ``c``-transpilation

Explicit version requirements are specified in the project `pyproject.toml
<https://github.com/nstarman/overload_numpy/blob/main/pyproject.toml>`_. ``pip``
and ``conda`` should install and enforce these versions automatically.
