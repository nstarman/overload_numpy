.. include:: references.txt

############################
Overload-NumPy Documentation
############################

|NumPy| offers powerful methods to allow arguments of |NumPy| functions (and
|ufunc| objects) to define how a given function operates on them. The details
are specified in NEP13_ and NEP18_, but in summary: normally |NumPy| only works
on an |ndarray| but with NEP13_/NEP18_ for a custom object, users can register
overrides for a |NumPy| function and then use that function on that object (a
quick example is outlined below). Plugging into the |NumPy| framework is
convenient both for developers -- to let |NumPy| take care of the actual math --
and users -- who get many things, not least of which is a familiar API. If all
this sounds great, that's because it is. However, if you've read NEP13_/NEP18_
then you know that making the |NumPy| bridge to your custom object and
registering overrides is non-trivial. That's where |overload_numpy| comes in.
|overload_numpy| offers convenient base classes for the |NumPy| bridge and
powerful methods to register overrides for functions, |ufunc| objects, and even
|ufunc| methods (e.g. ``.accumulate()``). The library is fully typed and
(almost) fully ``c``-transpiled for speed.

.. code-block:: python

   from dataclasses import dataclass
   import numpy as np

   @dataclass
   class ArrayWrapper:
      x: np.ndarray

      ... # lot's of non-trivial implementation details

   aw = ArrayWrapper(np.arange(10))

   np.add(aw, aw)  # returns ArrayWrapper([0, 2, ...])


This package is being actively developed in a `public repository on GitHub
<https://github.com/nstarman/overload_numpy/>`_, and we are always looking for
new contributors! No contribution is too small, so if you have any trouble with
this code, find a typo, or have requests for new content (tutorials or
features), `open an issue on GitHub
<https://github.com/nstarman/overload_numpy/issues>`_.


.. ---------------------
.. Nav bar (top of docs)

.. toctree::
   :maxdepth: 1
   :titlesonly:

   install
   getting_started
   src/index
   contributing


Contributors
============

.. include:: ../AUTHORS.rst
