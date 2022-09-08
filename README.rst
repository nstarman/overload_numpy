Overload ``NumPy`` functions
############################

Tools for implementing ``__numpy_function__`` on custom functions.

Quick Worked Example
--------------------

This is a simple example.

First, some imports:

    >>> from dataclasses import dataclass
    >>> from typing import ClassVar
    >>> import numpy as np
    >>> from overload_numpy import NumPyOverloader, NDFunctionMixin

Now we can define a ``overload_numpy.NumPyOverloader`` instance:

    >>> VEC_FUNCS = NumPyOverloader()

The overloads apply to an array wrapping class. Let's define one:

    >>> @dataclass
    ... class Vector1D(NDFunctionMixin):
    ...     '''A simple array wrapper.'''
    ...     x: np.ndarray
    ...     NP_FUNC_OVERLOADS: ClassVar[NumPyOverloader] = VEC_FUNCS

Now ``numpy`` functions can be overloaded and registered for ``Vector1D``.

    >>> @VEC_FUNCS.implements(np.concatenate, Vector1D)
    ... def concatenate(vec1ds: tuple[Vector1D, ...]) -> Vector1D:
    ...     return Vector1D(np.concatenate(tuple(v.x for v in vec1ds)))

Time to check this works:

    >>> vec1d = Vector1D(np.arange(3))
    >>> newvec = np.concatenate((vec1d, vec1d))
    >>> newvec
    Vector1D(x=array([0, 1, 2, 0, 1, 2]))

Great. Your turn!


Details
-------

See the Docs.
