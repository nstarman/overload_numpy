##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import pickle
from copy import copy, deepcopy
from numbers import Complex, Rational, Real
from typing import TYPE_CHECKING, Callable

# THIRDPARTY
import numpy as np
import pytest

# LOCALFOLDER
from .test_constraints import TypeConstraint_TestBase
from overload_numpy.constraints import Covariant, Invariant, TypeConstraint
from overload_numpy.npinfo import _NOT_DISPATCHED, _NotDispatched, _NumPyInfo

if TYPE_CHECKING:
    # STDLIB
    from types import FunctionType


##############################################################################


class Magnitude:
    def __init__(self, x) -> None:
        self._x = x


def add(obj1, obj2):
    return obj1._x + obj2


##############################################################################
# TESTS
##############################################################################


class Test__NotDispatched(TypeConstraint_TestBase):
    @pytest.fixture(scope="class")
    def constraint_cls(self) -> type:
        return _NotDispatched

    @pytest.fixture(scope="class")
    def constraint(self, constraint_cls) -> TypeConstraint:
        return _NOT_DISPATCHED

    def test_validate_type(self, constraint) -> None:
        assert constraint.validate_type(None) is False


##############################################################################


class Test__NumPyInfo:
    """Test :class:`overload_numpy.dispatch._NumPyInfo`."""

    @pytest.fixture(scope="class")
    def custom_cls(self):
        return Magnitude

    @pytest.fixture(scope="class")
    def implements_info(self) -> tuple[Callable, Callable]:
        return add, np.add

    @pytest.fixture
    def npinfo(self, custom_cls: type, implements_info: tuple[FunctionType, Callable]) -> _NumPyInfo:
        # TypeConstraint
        tinfo = (Covariant(custom_cls), Covariant(Real))

        return _NumPyInfo(*implements_info, tinfo, dispatch_on=object)

    # ===============================================================

    def test_init_error_func(self):
        with pytest.raises(TypeError, match="func must be callable"):
            _NumPyInfo(func=None, implements=np.add, types=Invariant(Real), dispatch_on=object)

    def test_init_error_implements(self):
        with pytest.raises(TypeError, match="implements must be callable"):
            _NumPyInfo(func=lambda x: x, implements=None, types=Invariant(Real), dispatch_on=object)

    def test_init_error_types(self):
        with pytest.raises(TypeError, match="types"):
            _NumPyInfo(func=lambda x: x, implements=np.add, types=None, dispatch_on=object)

    def test_init_error_dispatch_on(self):
        with pytest.raises(TypeError, match="dispatch"):
            _NumPyInfo(func=lambda x: x, implements=np.add, types=Invariant(Real), dispatch_on=None)

    def test_init(self, npinfo):
        # The pytest fixture proves it passes.
        pass

    # -------------------------------------------

    @pytest.mark.xfail  # limited by mypyc
    def test_validate_types_NotImplemented(self, implements_info):
        npinfo = _NumPyInfo(*implements_info, types=NotImplemented, dispatch_on=object)

        assert npinfo.validate_types(()) is False

    def test_validate_types_NotDispatched(self, implements_info):
        npinfo = _NumPyInfo(*implements_info, types=_NOT_DISPATCHED, dispatch_on=object)

        assert npinfo.validate_types(()) is False

    def test_validate_types_TypeConstraint(self, implements_info):
        # TODO! go through more TypeConstraint
        npinfo = _NumPyInfo(*implements_info, types=Invariant(Real), dispatch_on=object)

        assert npinfo.validate_types((Real,)) is True
        assert npinfo.validate_types((Real, Real)) is True

        assert npinfo.validate_types((Rational,)) is False
        assert npinfo.validate_types((Real, Rational)) is False
        assert npinfo.validate_types((Rational, Rational)) is False

        assert npinfo.validate_types((Complex,)) is False
        assert npinfo.validate_types((Complex, Real)) is False
        assert npinfo.validate_types((Complex, Rational)) is False

    def test_validate_types_Collection(self, implements_info):
        # TODO! go through more TypeConstraint
        npinfo = _NumPyInfo(*implements_info, types=(Invariant(Real), Invariant(Complex)), dispatch_on=object)

        assert npinfo.validate_types((Real,)) is True
        assert npinfo.validate_types((Real, Real)) is True

        assert npinfo.validate_types((Rational,)) is False
        assert npinfo.validate_types((Real, Rational)) is False
        assert npinfo.validate_types((Rational, Rational)) is False

        assert npinfo.validate_types((Complex,)) is True
        assert npinfo.validate_types((Complex, Real)) is True
        assert npinfo.validate_types((Complex, Rational)) is False

    # ===============================================================
    # Usage Tests

    @pytest.mark.incompatible_with_mypyc
    def test_serialization(self, implements_info) -> None:
        # copying
        assert copy(implements_info) == implements_info
        assert deepcopy(implements_info) == implements_info

        # pickling
        dumps = pickle.dumps(implements_info)
        assert pickle.loads(dumps) == implements_info
