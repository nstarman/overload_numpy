"""Test :mod:`overload_numpy.mixin`."""

##############################################################################
# IMPORTS

# STDLIB
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

# THIRDPARTY
import numpy as np
import pytest

# LOCAL
from .data import A
from overload_numpy import NumPyOverloader
from overload_numpy.mixin import (
    NPArrayFuncOverloadMixin,
    NPArrayOverloadMixin,
    NPArrayUFuncOverloadMixin,
)

if TYPE_CHECKING:
    # LOCAL
    from overload_numpy.constraints import TypeConstraint

##############################################################################
# TESTS
##############################################################################


class NPArrayOverloadMixinTestBase:
    @pytest.fixture(scope="class")
    def NP_OVERLOADS(self):
        return NumPyOverloader()

    @pytest.fixture(scope="class")
    def mixin_cls(self):
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def mixin_instance(self, mixin_cls):
        return mixin_cls()

    # ===============================================================


##############################################################################


class Test_NPArrayFuncOverloadMixin:
    @pytest.fixture(scope="class", params=[None, frozenset(), frozenset({A})])
    def NP_FUNC_TYPES(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def mixin_cls(self):
        return NPArrayFuncOverloadMixin

    @pytest.fixture(scope="class")
    def data_cls(self, mixin_cls, NP_OVERLOADS, NP_FUNC_TYPES):
        @dataclass
        class Wrap1D(mixin_cls):
            x: np.ndarray
            NP_OVERLOADS: ClassVar[NumPyOverloader] = NP_OVERLOADS
            NP_FUNC_TYPES: ClassVar[frozenset[type | TypeConstraint] | None] = NP_FUNC_TYPES

    # ===============================================================

    @pytest.mark.skip(reason="TODO")
    def test__array_function__(self, data_cls):
        pass


##############################################################################


class Test_NPArrayUFuncOverloadMixin:
    @pytest.fixture(scope="class")
    def mixin_cls(self):
        return NPArrayUFuncOverloadMixin

    @pytest.fixture(scope="class")
    def data_cls(self, mixin_cls, NP_OVERLOADS):
        @dataclass
        class Wrap1D(mixin_cls):
            x: np.ndarray
            NP_OVERLOADS: ClassVar[NumPyOverloader] = NP_OVERLOADS

    # ===============================================================

    @pytest.mark.skip(reason="TODO")
    def test__array_ufunc__(self, data_cls):
        pass


##############################################################################


class Test_NPArrayOverloadMixin(Test_NPArrayFuncOverloadMixin, Test_NPArrayUFuncOverloadMixin):
    @pytest.fixture(scope="class")
    def mixin_cls(self):
        return NPArrayOverloadMixin

    # ===============================================================
