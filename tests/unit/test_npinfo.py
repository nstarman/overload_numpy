##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from numbers import Complex, Rational, Real
from typing import TYPE_CHECKING, Callable

# THIRD PARTY
import numpy as np
import pytest

# LOCAL
from overload_numpy.constraints import Covariant, Invariant
from overload_numpy.npinfo import _NOT_DISPATCHED, _NumPyInfo

if TYPE_CHECKING:
    # STDLIB
    from types import FunctionType

##############################################################################
# TESTS
##############################################################################


class Test__NumPyInfo:
    """Test :class:`overload_numpy.dispatch._NumPyInfo`."""

    @pytest.fixture(scope="class")
    def custom_cls(self):
        class Magnitude:
            def __init__(self, x) -> None:
                self._x = x

        return Magnitude

    @pytest.fixture(scope="class")
    def implements_info(self) -> tuple[FunctionType, Callable]:
        def add(obj1, obj2):
            return obj1._x + obj2

        return add, np.add

    @pytest.fixture
    def npinfo(self, custom_cls: type, implements_info: tuple[FunctionType, Callable]) -> _NumPyInfo:
        # TypeConstraint
        tinfo = (Covariant(custom_cls), Covariant(Real))

        return _NumPyInfo(*implements_info, tinfo)

    # ===============================================================

    def test_init_error_func(self):
        with pytest.raises(TypeError, match="func must be callable"):
            _NumPyInfo(func=None, implements=np.add, types=None)

    def test_init_error_implements(self):
        with pytest.raises(TypeError, match="implements must be callable"):
            _NumPyInfo(func=lambda x: x, implements=None, types=None)

    def test_init_error_types(self):
        with pytest.raises(TypeError, match="types"):
            _NumPyInfo(func=lambda x: x, implements=np.add, types=None)

    def test_init(self, npinfo):
        # The pytest fixture proves it passes.
        pass

    # -------------------------------------------

    @pytest.mark.xfail  # limited by mypyc
    def test_validate_types_NotImplemented(self, implements_info):
        npinfo = _NumPyInfo(*implements_info, types=NotImplemented)

        assert npinfo.validate_types(()) is False

    def test_validate_types_NotDispatched(self, implements_info):
        npinfo = _NumPyInfo(*implements_info, types=_NOT_DISPATCHED)

        assert npinfo.validate_types(()) is False

    def test_validate_types_TypeConstraint(self, implements_info):
        # TODO! go through more TypeConstraint
        npinfo = _NumPyInfo(*implements_info, types=Invariant(Real))

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
        npinfo = _NumPyInfo(*implements_info, types=(Invariant(Real), Invariant(Complex)))

        assert npinfo.validate_types((Real,)) is True
        assert npinfo.validate_types((Real, Real)) is True

        assert npinfo.validate_types((Rational,)) is False
        assert npinfo.validate_types((Real, Rational)) is False
        assert npinfo.validate_types((Rational, Rational)) is False

        assert npinfo.validate_types((Complex,)) is True
        assert npinfo.validate_types((Complex, Real)) is True
        assert npinfo.validate_types((Complex, Rational)) is False
