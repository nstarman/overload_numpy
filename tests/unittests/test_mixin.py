"""Test :mod:`overload_numpy.mixin`."""

from dataclasses import dataclass, field, make_dataclass
from typing import ClassVar, FrozenSet, Optional, Union

import numpy as np
import pytest
from overload_numpy import NumPyOverloader
from overload_numpy.constraints import TypeConstraint
from overload_numpy.mixin import (
    NPArrayFuncOverloadMixin,
    NPArrayOverloadMixin,
    NPArrayUFuncOverloadMixin,
)

from .data import A

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


class Test_NPArrayFuncOverloadMixin(NPArrayOverloadMixinTestBase):
    @pytest.fixture(scope="class", params=[None, frozenset(), frozenset({A})])
    def NP_FUNC_TYPES(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def mixin_cls(self):
        return NPArrayFuncOverloadMixin

    @pytest.fixture(scope="class")
    def data_cls(self, mixin_cls, NP_OVERLOADS, NP_FUNC_TYPES):
        return make_dataclass(
            "Wrap1D",
            fields=[
                ("x", np.ndarray),
                ("NP_OVERLOADS", ClassVar[NumPyOverloader], field(default=NP_OVERLOADS)),
                (
                    "NP_FUNC_TYPES",
                    ClassVar[Optional[FrozenSet[Union[type, TypeConstraint]]]],
                    field(default=NP_FUNC_TYPES),
                ),
            ],
            bases=(mixin_cls,),
        )

    # ===============================================================

    def test_data_cls(self, data_cls, NP_OVERLOADS):
        assert isinstance(data_cls.NP_OVERLOADS, type(NP_OVERLOADS))

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

        return Wrap1D

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
