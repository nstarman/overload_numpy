##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

# THIRD PARTY
import numpy as np
import pytest

# LOCAL
from overload_numpy import NumPyOverloader
from overload_numpy.constraints import Covariant
from overload_numpy.dispatch import _notdispatched_info

if TYPE_CHECKING:
    # THIRD PARTY
    from typing_extensions import Self


##############################################################################
# CODE

VEC_FUNCS = NumPyOverloader()


@dataclass(frozen=True)
class Vector1D:
    """A simple array wrapper."""

    x: np.ndarray

    def __array_function__(
        self: Self, func: Callable[..., Any], types: tuple[type, ...], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Self:
        """Interface with :mod:`numpy` functions."""
        if func not in VEC_FUNCS:
            return NotImplemented

        # Get _NumPyInfo on function, given type of self
        finfo = VEC_FUNCS[func](self)
        if not finfo.validate_types(types):
            return NotImplemented

        return finfo.func(*args, **kwargs)


@VEC_FUNCS.implements(np.concatenate, types=Vector1D, dispatch_on=Vector1D)
def concatenate(vec1ds: tuple[Vector1D, ...]) -> Vector1D:
    return Vector1D(np.concatenate(tuple(v.x for v in vec1ds)))


##############################################################################
# FIXTURES


@pytest.fixture
def overloader() -> NumPyOverloader:
    return VEC_FUNCS


@pytest.fixture
def array() -> np.ndarray:
    return np.arange(10)


@pytest.fixture
def vec1d(array: np.ndarray) -> Vector1D:
    return Vector1D(array)


##############################################################################
# TESTS


def test_entries(overloader, vec1d):
    """Test current entries."""

    # Only `numpy.add` is registered.
    assert np.concatenate in overloader
    assert {np.concatenate} == overloader.keys()

    # `numpy` function -> Dispatcher
    dispatcher = overloader[np.concatenate]

    # Dispatcher starts with `object`
    npinfo = dispatcher(object())
    assert npinfo is _notdispatched_info

    # :func:`numpy.concatenate` is also registered
    npinfo = dispatcher(vec1d)
    assert npinfo.func is concatenate
    assert npinfo.implements is np.concatenate
    assert npinfo.types == {
        Covariant(Vector1D),
    }


def test_calling(overloader, vec1d, array):
    """Test calling numpy functions."""
    # TODO! replace with pytest_arraydiff
    newvec = np.concatenate((vec1d, vec1d))
    newarr = np.concatenate((array, array))

    assert np.array_equal(newvec.x, newarr)
