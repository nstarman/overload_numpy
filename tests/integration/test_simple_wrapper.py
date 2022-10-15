##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import copy
import re
from dataclasses import dataclass, fields
from typing import ClassVar

# THIRDPARTY
import numpy as np
import pytest

# LOCAL
from overload_numpy import NPArrayOverloadMixin, NumPyOverloader
from overload_numpy.constraints import Covariant

##############################################################################
# CLASSES


NP_OVERLOADS = NumPyOverloader()


@dataclass
class Wrap1(NPArrayOverloadMixin):

    x: np.ndarray
    NP_OVERLOADS: ClassVar[NumPyOverloader] = NP_OVERLOADS


@dataclass
class Wrap2(Wrap1):
    y: np.ndarray


##############################################################################
# OVERLOADS

# ---------------------------------------------------------
# Implements function


@NP_OVERLOADS.implements(np.concatenate, Wrap1)
def concatenate(ws, *args, **kwargs):
    return Wrap1(np.concatenate(tuple(v.x for v in ws), *args, **kwargs))


# ---------------------------------------------------------
# Implements `ufunc`.


@NP_OVERLOADS.implements(np.add, Wrap1)
def add(w1, w2, *args, **kwargs):
    return Wrap1(np.add(w1.x, w2.x, *args, **kwargs))


# and the ufunc methods
@add.register("at")
def add_at(w1, indices, w2):
    np.add.at(w1.x, indices, w2.x)


@add.register("accumulate")
def add_accumulate(w1, axis=0, dtype=None, out=None):
    # if out is not None:  # TODO!
    #     np.add.accumulate(w1.x, axis=axis, dtype=dtype, out=out.x)
    #     return out
    return Wrap1(np.add.accumulate(w1.x, axis=axis, dtype=dtype))


@add.register("outer")
def add_outer(w1, w2, /, *, out=None, **kwargs):
    # if out is not None:  # TODO!
    #     np.add.outer(w1.x, w2.x, out=out.x, **kwargs)
    #     return out
    return Wrap1(np.add.outer(w1.x, w2.x, **kwargs))


@add.register("reduce")
def add_reduce(w, out=None, **kwargs):
    # if out is not None:  # TODO!
    #     np.add.reduce(w.x, out=out.x, **kwargs)
    #     return out
    return Wrap1(np.add.reduce(w.x, **kwargs))


@add.register("reduceat")
def add_reduceat(w, indices, axis=0, dtype=None, out=None):
    # if out is not None:   # TODO!
    #     np.add.reduceat(w.x, indices, axis=axis, dtype=dtype, out=out.x)
    #     return out
    return Wrap1(np.add.reduceat(w.x, indices, axis=axis, dtype=dtype))


# ---------------------------------------------------------
# Assists a function

stack_funcs = {np.vstack, np.hstack, np.dstack, np.column_stack, np.row_stack}


@NP_OVERLOADS.assists(stack_funcs, types=Wrap1, dispatch_on=Wrap1)
def stack_assists(cls, func, /, ws, *args, **kwargs):
    return cls(*(func(tuple(getattr(v, f.name) for v in ws), *args, **kwargs) for f in fields(cls)))


# ---------------------------------------------------------
# Assists a ufunc

mul_funcs = {np.multiply, np.divide}


@NP_OVERLOADS.assists(mul_funcs, types=Wrap1, dispatch_on=Wrap1)
def mul_assists(cls, func, /, w1, w2, *args, **kwargs):
    # TODO! handle "out"
    return cls(*(func(getattr(w1, f.name), getattr(w2, f.name), *args, **kwargs) for f in fields(cls)))


@mul_assists.register("at")
def mul_at_assists(cls, func, /, w1, index, w2):
    return cls(*(func(getattr(w1, f.name), index, w2.x) for f in fields(cls)))


@mul_assists.register("accumulate")
def mul_accumulate_assists(cls, func, /, w1, axis=0, dtype=None, out=None):
    # if out is not None:  # TODO!
    return cls(*(func(getattr(w1, f.name), axis, dtype, out) for f in fields(cls)))


@mul_assists.register("outer")
def mul_outer_assists(cls, func, /, w1, w2, **kwargs):
    # if out is not None:  # TODO!
    return cls(*(func(getattr(w1, f.name), getattr(w2, f.name), **kwargs) for f in fields(cls)))


@mul_assists.register("reduce")
def mul_reduce_assists(cls, func, /, w1, **kwargs):
    # if out is not None:  # TODO!
    return cls(*(func(getattr(w1, f.name), **kwargs) for f in fields(cls)))


@mul_assists.register("reduceat")
def mul_reduceat_assists(cls, func, /, w1, indices, axis=0, dtype=None, out=None):
    # if out is not None:   # TODO!
    #     np.add.reduceat(w.x, indices, axis=axis, dtype=dtype, out=out.x)
    #     return out
    return cls(*(func(getattr(w1, f.name), indices, axis=axis, dtype=dtype) for f in fields(cls)))


##############################################################################
# FIXTURES


@pytest.fixture
def overloader() -> NumPyOverloader:
    return NP_OVERLOADS


@pytest.fixture
def array() -> np.ndarray:
    return np.arange(1, 11, dtype=float)  # to avoid dividing by 0


@pytest.fixture
def array2() -> np.ndarray:
    return np.arange(11, 21, dtype=float)


@pytest.fixture
def w1(array: np.ndarray) -> Wrap1:
    return Wrap1(array)


@pytest.fixture
def w1b(array: np.ndarray) -> Wrap1:
    return Wrap1(array)


@pytest.fixture
def w2(array: np.ndarray, array2: np.ndarray) -> Wrap2:
    return Wrap2(array, array2)


@pytest.fixture
def w2b(array: np.ndarray, array2: np.ndarray) -> Wrap2:
    return Wrap2(array, array2)


# ##############################################################################
# # TESTS


@pytest.mark.parametrize("func", [np.concatenate, np.add, *stack_funcs, *mul_funcs])
def test_entry(overloader, w1, func):
    """Test current entries."""
    # what's registered
    assert func in overloader
    assert f"numpy.{func.__name__}" in overloader.keys()

    # `numpy` function -> Dispatcher
    dispatcher = overloader[np.concatenate]

    # Dispatcher starts with `object`
    with pytest.raises(NotImplementedError):
        dispatcher(object())

    # (u)func is also registered
    npinfo = dispatcher(w1)
    assert npinfo.func is concatenate
    assert npinfo.implements is np.concatenate

    assert npinfo.types == {Covariant(Wrap1)}


def test_not_dispatched(overloader, w1):
    # not dispatched
    assert np.subtract not in overloader

    with pytest.raises(TypeError, match=re.escape("operand type(s) all returned NotImplemented")):
        np.subtract(w1, w1)


class Test_Concatenate:
    def test_concatenate(self, w1, array):
        # TODO! replace with pytest_arraydiff
        nw = np.concatenate((w1, w1))
        na = np.concatenate((array, array))

        assert np.array_equal(nw.x, na)

    def test_types_must_be_compatible(self, w1):
        with pytest.raises(Exception):
            np.concatenate((w1, object()))

    def test_compatible_types_can_be_wrong(self, w1, w2):
        """Wrap2 has extra attributes over Wrap1 that are ignored."""
        nw = np.concatenate((w1, w2))
        assert not hasattr(nw, "y")


class Test_Add:
    def test_add(self, w1, w1b, array):
        nw = np.add(w1, w1b)
        na = np.add(array, array)
        assert np.array_equal(nw.x, na)

    def test_types_must_be_compatible(self, w1):
        with pytest.raises(Exception):
            np.add(w1, object())

    def test_compatible_types_can_be_wrong(self, w1, w2):
        """Wrap2 has extra attributes over Wrap1 that are ignored."""
        nw = np.add(w1, w2)
        assert not hasattr(nw, "y")

    def test_method_at(self, array):
        nw = Wrap1(copy.deepcopy(array))
        np.add.at(nw, 0, Wrap1(1.5))  # adding to itself
        assert np.equal(nw.x[0], array[0] + 1.5)

    def test_method_accumulate(self, w1, array):
        nw = np.add.accumulate(w1)
        na = np.add.accumulate(array)
        assert np.array_equal(nw.x, na)

    def test_method_outer(self, w1, w1b, array):
        nw = np.add.outer(w1, w1b)
        na = np.add.outer(array, array)
        assert np.array_equal(nw.x, na)

    def test_method_reduce(self, w1, array):
        nw = np.add.reduce(w1)
        na = np.add.reduce(array)
        assert np.array_equal(nw.x, na)

    def test_method_reduceat(self, w1, array):
        nw = np.add.reduceat(w1, [0, 1, 2])
        na = np.add.reduceat(array, [0, 1, 2])
        assert np.array_equal(nw.x, na)


class Test_stacks:
    @pytest.fixture(scope="class", params=stack_funcs)
    def func(self, request):
        return request.param

    def test_stack(self, func, w1, w1b, array):
        nw = func((w1, w1b))
        na = func((array, array))
        assert np.array_equal(nw.x, na)

    def test_types_must_be_compatible(self, func, w1):
        with pytest.raises(Exception):
            func((w1, object()))

    def test_compatible_types_fail(self, func, w1, w2):
        """Wrap2 has extra attributes over Wrap1."""
        # TODO! figure out why dispatches on Wrap2 before Wrap1
        with pytest.raises(AttributeError):
            func((w1, w2))

    def generalizes_to_subclass(self, func, w2, w2b, array, array2):
        nw = func((w2, w2b))
        na1 = func((array, array))
        na2 = func((array2, array2))
        assert np.array_equal(nw.x, na1)
        assert np.array_equal(nw.y, na2)


class Test_muls:
    @pytest.fixture(scope="class", params=mul_funcs)
    def func(self, request):
        return request.param

    def test_stack(self, func, w1, w1b, array):
        nw = func(w1, w1b)
        na = func(array, array)
        assert np.array_equal(nw.x, na)

    def test_types_must_be_compatible(self, func, w1):
        with pytest.raises(Exception):
            func(w1, object())

    def test_compatible_types_fail(self, func, w1, w2):
        """Wrap2 has extra attributes over Wrap1."""
        # TODO! figure out why dispatches on Wrap2 before Wrap1
        with pytest.raises(AttributeError):
            func(w1, w2)

    def test_method_at(self, func, array):
        nw = Wrap1(copy.deepcopy(array))
        func.at(nw, 0, Wrap1(1.5))
        na = func(array[0], 1.5)
        assert np.equal(nw.x[0], na)

    def test_method_accumulate(self, func, w1, array):
        nw = func.accumulate(w1)
        na = func.accumulate(array)
        assert np.array_equal(nw.x, na)

    def test_method_outer(self, func, w1, w1b, array):
        nw = func.outer(w1, w1b)
        na = func.outer(array, array)
        assert np.array_equal(nw.x, na)

    def test_method_reduce(self, func, w1, array):
        nw = func.reduce(w1)
        na = func.reduce(array)
        assert np.array_equal(nw.x, na)

    def test_method_reduceat(self, func, w1, array):
        nw = func.reduceat(w1, [0, 1, 2])
        na = func.reduceat(array, [0, 1, 2])
        assert np.array_equal(nw.x, na)
