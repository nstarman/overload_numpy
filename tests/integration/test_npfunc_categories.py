##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import dataclass, fields
from typing import ClassVar

# THIRDPARTY
# THIRD PARTY
import numpy as np
import pytest

# LOCAL
import overload_numpy
from overload_numpy import NumPyOverloader, npfunc_categories

##############################################################################
# SETUP

VEC_FUNCS = NumPyOverloader()


@dataclass
class Vector1D(overload_numpy.NDFunctionMixin):
    """A simple array wrapper."""

    x: np.ndarray
    NP_FUNC_OVERLOADS: ClassVar[NumPyOverloader] = VEC_FUNCS


vec1d = Vector1D(np.arange(3, dtype=float))

##############################################################################
# TESTS
##############################################################################


@VEC_FUNCS.assists(npfunc_categories.has_like, types=Vector1D, dispatch_on=Vector1D)
def has_like(dispatch_on, func, *args, **kwargs):
    return dispatch_on(*(func(*args, **kwargs) for _ in fields(dispatch_on)))


@pytest.mark.parametrize(
    ("numpy_func", "input_args_kwargs"),
    [
        (np.empty, ((10,), {})),
        (np.eye, ((10,), {})),
        (np.identity, ((10,), {})),
        (np.ones, ((10,), {})),
        (np.zeros, ((10,), {})),
        (np.array, ((10,), {})),
        (np.asarray, ((10,), {})),
        (np.asanyarray, ((10,), {})),
        (np.ascontiguousarray, ((10,), {})),
        (np.frombuffer, ((np.arange(3).tobytes(),), {})),
        # np.fromfile,  # todo?
        # np.fromfunction,  # todo?
        (np.fromiter, ((range(10),), {"dtype": int})),
        (np.fromstring, (("1 2 3",), {"sep": " "})),
    ],
)
def test_has_like(numpy_func, input_args_kwargs):
    args, kwargs = input_args_kwargs
    got = numpy_func(*args, **kwargs, like=vec1d)
    expected = numpy_func(*args, **kwargs)

    assert isinstance(got, Vector1D)

    if "empty" in numpy_func.__name__:  # FIXME! for empty
        return

    assert np.all(np.isclose(got.x, expected))


# ===================================================================


@VEC_FUNCS.assists(npfunc_categories.as_like, types=Vector1D, dispatch_on=Vector1D)
def as_like(dispatch_on, func, prototype, *args, **kwargs):
    return dispatch_on(*(func(getattr(prototype, f.name), *args, **kwargs) for f in fields(prototype)))


@pytest.mark.parametrize(
    ("numpy_func", "input_args_kwargs"),
    [
        (np.empty_like, ((), {})),
        (np.ones_like, ((), {})),
    ],
)
def test_as_like(numpy_func, input_args_kwargs):
    args, kwargs = input_args_kwargs
    got = numpy_func(vec1d, *args, **kwargs)
    expected = numpy_func(vec1d.x, *args, **kwargs)

    assert isinstance(got, Vector1D)

    if "empty" in numpy_func.__name__:  # FIXME! for empty
        return

    assert np.all(np.isclose(got.x, expected))  # FIXME for empty_like
