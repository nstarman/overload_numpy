Overload ``NumPy`` functions
############################

Tools for implementing ``__array_function__`` on custom functions.


Implementing an Overload
------------------------

First, some imports:

    >>> from dataclasses import dataclass, fields
    >>> from typing import ClassVar
    >>> import numpy as np
    >>> from overload_numpy import NumPyOverloader, NDFunctionMixin

Now we can define a |NumPyOverloader| instance:

    >>> VEC_FUNCS = NumPyOverloader()

The overloads apply to an array wrapping class. Let's define one:

    >>> @dataclass
    ... class Vector1D(NDFunctionMixin):
    ...     '''A simple array wrapper.'''
    ...     x: np.ndarray
    ...     NP_FUNC_OVERLOADS: ClassVar[NumPyOverloader] = VEC_FUNCS

Now ``numpy`` functions can be overloaded and registered for ``Vector1D``.

    >>> @VEC_FUNCS.implements(np.concatenate, Vector1D)
    ... def concatenate(vec1ds):
    ...     return Vector1D(np.concatenate(tuple(v.x for v in vec1ds)))

Time to check this works:

    >>> vec1d = Vector1D(np.arange(3))
    >>> newvec = np.concatenate((vec1d, vec1d))
    >>> newvec
    Vector1D(x=array([0, 1, 2, 0, 1, 2]))


Dispatching Overloads for Subclasses
------------------------------------

What if we defined a subclass of ``Vector1D``?

    >>> @dataclass
    ... class Vector2D(Vector1D):
    ...     '''A simple 2-array wrapper.'''
    ...     y: np.ndarray

The overload for :func:`numpy.concatenate` registered on ``Vector1D`` will not
work correctly for ``Vector2D``. However,
:class:`~overload_numpy.NumPyOverloader` supports single-dispatch on the calling
type for the overload, so overloads can be customized for subclasses.

    >>> @VEC_FUNCS.implements(np.concatenate, Vector2D)
    ... def concatenate(vec2ds):
    ...     print("using Vector2D implementation...")
    ...     return Vector2D(np.concatenate(tuple(v.x for v in vec2ds)),
    ...                     np.concatenate(tuple(v.y for v in vec2ds)))

Checking this works:

    >>> vec2d = Vector2D(np.arange(3), np.arange(3, 6))
    >>> newvec = np.concatenate((vec2d, vec2d))
    using Vector2D implementation...
    >>> newvec
    Vector2D(x=array([0, 1, 2, 0, 1, 2]), y=array([3, 4, 5, 3, 4, 5]))


Great! But rather than defining a new implementation for each
subclass, let's see how we could write a more broadly applicable overload:

    >>> @VEC_FUNCS.implements(np.concatenate, Vector2D)  # overriding
    ... def concatenate(vecs):
    ...     VT = type(vecs[0])
    ...     return VT(*(np.concatenate(tuple(getattr(v, f.name) for v in vecs))
    ...                 for f in fields(VT)))

Checking this works:

    >>> newvec = np.concatenate((vec2d, vec2d))
    >>> newvec
    Vector2D(x=array([0, 1, 2, 0, 1, 2]), y=array([3, 4, 5, 3, 4, 5]))


    >>> @dataclass
    ... class Vector3D(Vector2D):
    ...     '''A simple 3-array wrapper.'''
    ...     z: np.ndarray

    >>> vec3d = Vector3D(np.arange(2), np.arange(3, 5), np.arange(6, 8))
    >>> newvec = np.concatenate((vec3d, vec3d))
    >>> newvec
    Vector3D(x=array([0, 1, 0, 1]), y=array([3, 4, 3, 4]), z=array([6, 7, 6, 7]))


Assisting Groups of Overloads
-----------------------------

In the previous examples we wrote implementations for a single NumPy function. Overloading the full set of NumPy functions this way would take a long time.

Wouldn't it be better if we could write many fewer, based on groups of NumPy functions.

    >>> stack_funcs = {np.vstack, np.hstack, np.dstack, np.column_stack, np.row_stack}
    >>> @VEC_FUNCS.assists(stack_funcs, types=Vector1D, dispatch_on=Vector1D)
    ... def stack_assists(dispatch_on, func, vecs, *args, **kwargs):
    ...     cls = type(vecs[0])
    ...     return cls(*(func(tuple(getattr(v, f.name) for v in vecs), *args, **kwargs)
    ...                     for f in fields(cls)))

Checking this works:

    >>> np.vstack((vec1d, vec1d))
    Vector1D(x=array([[0, 1, 2],
                      [0, 1, 2]]))

    >>> np.hstack((vec1d, vec1d))
    Vector1D(x=array([0, 1, 2, 0, 1, 2]))


Details
-------

See the Docs.
