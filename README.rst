Overload ``NumPy`` ufuncs and functions
#######################################

.. container::

    |PyPI status| |coverage status| |RTD status| |black status| |pre-commit status|


``overload_numpy`` provides easy-to-use tools for working with ``NumPy``'s
``__array_(u)func(tion)__``. The library is fully typed and wheels are compiled
with mypyc.


Implementing an Overload
------------------------

First, some imports:

    >>> from dataclasses import dataclass, fields
    >>> from typing import ClassVar
    >>> import numpy as np
    >>> from overload_numpy import NumPyOverloader, NPArrayOverloadMixin

Now we can define a ``NumPyOverloader`` instance:

    >>> W_FUNCS = NumPyOverloader()

The overloads apply to an array wrapping class. Let's define one:

    >>> @dataclass
    ... class Wrap1D(NPArrayOverloadMixin):
    ...     '''A simple array wrapper.'''
    ...     x: np.ndarray
    ...     NP_OVERLOADS: ClassVar[NumPyOverloader] = W_FUNCS

    >>> w1d = Wrap1D(np.arange(3))

Now both ``numpy.ufunc`` (e.g. ``numpy.add``) and ``numpy`` functions (e.g.
``numpy.concatenate``) can be overloaded and registered for ``Wrap1D``.

    >>> @W_FUNCS.implements(np.add, Wrap1D)
    ... def add(w1, w2):
    ...     return Wrap1D(np.add(w1.x, w2.x))

    >>> @W_FUNCS.implements(np.concatenate, Wrap1D)
    ... def concatenate(w1ds):
    ...     return Wrap1D(np.concatenate(tuple(w.x for w in w1ds)))

Time to check these work:

    >>> np.add(w1d, w1d)
    Wrap1D(x=array([0, 2, 4]))

    >>> np.concatenate((w1d, w1d))
    Wrap1D(x=array([0, 1, 2, 0, 1, 2]))

``ufunc`` also have a number of methods: 'at', 'accumulate', etc. The function
dispatch mechanism in `NEP13
<https://numpy.org/neps/nep-0013-ufunc-overrides.html>`_ says that  "If one of
the input or output arguments implements __array_ufunc__, it is executed instead
of the ufunc." Currently the overloaded ``numpy.add`` does not work for any of
the ``ufunc`` methods.

    >>> try: np.add.accumulate(w1d)
    ... except Exception: print("failed")
    failed

``ufunc`` method overloads can be registered on the wrapped ``add``
implementation:

    >>> @add.register('accumulate')
    ... def add_accumulate(w1):
    ...     return Wrap1D(np.add.accumulate(w1.x))

    >>> np.add.accumulate(w1d)
    Wrap1D(x=array([0, 1, 3]))


Dispatching Overloads for Subclasses
------------------------------------
What if we defined a subclass of ``Wrap1D``?

    >>> @dataclass
    ... class Wrap2D(Wrap1D):
    ...     '''A simple 2-array wrapper.'''
    ...     y: np.ndarray

The overload for ``numpy.concatenate`` registered on ``Wrap1D`` will not work
correctly for ``Wrap2D``. However, ``NumPyOverloader`` supports single-dispatch
on the calling type for the overload, so overloads can be customized for
subclasses.

    >>> @W_FUNCS.implements(np.add, Wrap2D)
    ... def add(w1, w2):
    ...     print("using Wrap2D implementation...")
    ...     return Wrap2D(np.add(w1.x, w2.x),
    ...                   np.add(w1.y, w2.y))

    >>> @W_FUNCS.implements(np.concatenate, Wrap2D)
    ... def concatenate2(w2ds):
    ...     print("using Wrap2D implementation...")
    ...     return Wrap2D(np.concatenate(tuple(w.x for w in w2ds)),
    ...                   np.concatenate(tuple(w.y for w in w2ds)))

Checking these work:

    >>> w2d = Wrap2D(np.arange(3), np.arange(3, 6))
    >>> np.add(w2d, w2d)
    using Wrap2D implementation...
    Wrap2D(x=array([0, 2, 4]), y=array([ 6, 8, 10]))

    >>> np.concatenate((w2d, w2d))
    using Wrap2D implementation...
    Wrap2D(x=array([0, 1, 2, 0, 1, 2]), y=array([3, 4, 5, 3, 4, 5]))

Great! But rather than defining a new implementation for each subclass,
let's see how we could write a more broadly applicable overload:

    >>> @W_FUNCS.implements(np.add, Wrap1D)  # overriding both
    ... @W_FUNCS.implements(np.add, Wrap2D)  # overriding both
    ... def add_general(w1, w2):
    ...     WT = type(w1)
    ...     return WT(*(np.add(getattr(w1, f.name), getattr(w2, f.name))
    ...                 for f in fields(WT)))

    >>> @W_FUNCS.implements(np.concatenate, Wrap1D)  # overriding both
    ... @W_FUNCS.implements(np.concatenate, Wrap2D)  # overriding both
    ... def concatenate_general(ws):
    ...     WT = type(ws[0])
    ...     return WT(*(np.concatenate(tuple(getattr(w, f.name) for w in ws))
    ...                 for f in fields(WT)))

Checking these work:

    >>> np.add(w2d, w2d)
    Wrap2D(x=array([0, 2, 4]), y=array([ 6, 8, 10]))

    >>> np.concatenate((w2d, w2d))
    Wrap2D(x=array([0, 1, 2, 0, 1, 2]), y=array([3, 4, 5, 3, 4, 5]))

    >>> @dataclass
    ... class Wrap3D(Wrap2D):
    ...     '''A simple 3-array wrapper.'''
    ...     z: np.ndarray

    >>> w3d = Wrap3D(np.arange(2), np.arange(3, 5), np.arange(6, 8))
    >>> np.add(w3d, w3d)
    Wrap3D(x=array([0, 2]), y=array([6, 8]), z=array([12, 14]))
    >>> np.concatenate((w3d, w3d))
    Wrap3D(x=array([0, 1, 0, 1]), y=array([3, 4, 3, 4]), z=array([6, 7, 6, 7]))


Assisting Groups of Overloads
-----------------------------

In the previous examples we wrote implementations for a single NumPy
function. Overloading the full set of NumPy functions this way would take a
long time.

Wouldn't it be better if we could write many fewer, based on groups of NumPy
functions?

    >>> add_funcs = {np.add, np.subtract}
    >>> @W_FUNCS.assists(add_funcs, types=Wrap1D, dispatch_on=Wrap1D)
    ... def add_assists(cls, func, w1, w2, *args, **kwargs):
    ...     return cls(*(func(getattr(w1, f.name), getattr(w2, f.name), *args, **kwargs)
    ...                     for f in fields(cls)))

    >>> stack_funcs = {np.vstack, np.hstack, np.dstack, np.column_stack, np.row_stack}
    >>> @W_FUNCS.assists(stack_funcs, types=Wrap1D, dispatch_on=Wrap1D)
    ... def stack_assists(cls, func, ws, *args, **kwargs):
    ...     return cls(*(func(tuple(getattr(v, f.name) for v in ws), *args, **kwargs)
    ...                     for f in fields(cls)))

Checking these work:

    >>> np.subtract(w2d, w2d)
    Wrap2D(x=array([0, 0, 0]), y=array([0, 0, 0]))

    >>> np.vstack((w1d, w1d))
    Wrap1D(x=array([[0, 1, 2],
                        [0, 1, 2]]))

    >>> np.hstack((w1d, w1d))
    Wrap1D(x=array([0, 1, 2, 0, 1, 2]))

We would also like to implement the ``accumulate`` method for all the
``add_funcs`` overloads:

    >>> @add_assists.register("accumulate")
    ... def add_accumulate_assists(cls, func, w1, *args, **kwargs):
    ...     return cls(*(func(getattr(w1, f.name), *args, **kwargs)
    ...                  for f in fields(cls)))

    >>> np.subtract.accumulate(w2d)
    Wrap2D(x=array([ 0, -1, -3]), y=array([ 3, -1, -6]))


Details
-------

Want to see about type constraints and the API? Check out the docs!



.. |black status| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Codestyle Black

.. |coverage status| image:: https://codecov.io/gh/nstarman/overload_numpy/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/nstarman/overload_numpy
    :alt: overload_numpy's Coverage Status

.. |pre-commit status| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit

.. |PyPI status| image:: https://img.shields.io/pypi/v/overload_numpy.svg
    :target: https://pypi.org/project/overload_numpy
    :alt: overload_numpy's PyPI Status

.. |RTD status| image:: https://readthedocs.org/projects/overload-numpy/badge/?version=latest
    :target: https://overload-numpy.readthedocs.io/en/latest/?badge=latest
    :alt: overload_numpy's Documentation Status
