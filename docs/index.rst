##############
Overload NumPy
##############


Quick Worked Example
--------------------

.. testsetup::

    >>> from __future__ import annotations

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
    ... def concatenate(vec1ds: 'tuple[Vector1D, ...]') -> Vector1D:
    ...     return Vector1D(np.concatenate(tuple(v.x for v in vec1ds)))

Time to check this works:

    >>> vec1d = Vector1D(np.arange(3))
    >>> newvec = np.concatenate((vec1d, vec1d))
    >>> newvec
    Vector1D(x=array([0, 1, 2, 0, 1, 2]))


Overloading Subclasses
----------------------

What if we defined a subclass of ``Vector1D``?

    >>> @dataclass
    ... class Vector2D(Vector1D):
    ...     '''A simple 2-array wrapper.'''
    ...     y: np.ndarray

The overload for :func:`numpy.concatenate` registered on ``Vector1D`` will not
work correctly for ``Vector2D``. :class:`~overload_numpy.NumPyOverloader`
supports single-dispatch on the calling type for the overload, so overload can
be customized for subclasses.

    >>> @VEC_FUNCS.implements(np.concatenate, Vector2D)
    ... def concatenate(vec2ds: 'tuple[Vector2D, ...]') -> Vector2D:
    ...     print("using Vector2D implementation...")
    ...     return Vector2D(np.concatenate(tuple(v.x for v in vec2ds)),
    ...                     np.concatenate(tuple(v.y for v in vec2ds)))

Checking this works:

    >>> vec2d = Vector2D(np.arange(3), np.arange(3, 6))
    >>> newvec = np.concatenate((vec2d, vec2d))
    using Vector2D implementation...
    >>> newvec
    Vector2D(x=array([0, 1, 2, 0, 1, 2]), y=array([3, 4, 5, 3, 4, 5]))


That works great! But rather than defining a new implementation for each
subclass, let's see how we could write a more broadly applicable overload:

    >>> from dataclasses import fields
    >>> from typing import TypeVar
    >>> V = TypeVar("V", bound="Vector1D")

    >>> @VEC_FUNCS.implements(np.concatenate, Vector2D)  # overriding
    ... def concatenate(vecs: 'tuple[V, ...]') -> V:
    ...     VT = type(vecs[0])
    ...     if not all(isinstance(v, VT) for v in vecs):  # make sure all 1 type
    ...         return NotImplemented
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
