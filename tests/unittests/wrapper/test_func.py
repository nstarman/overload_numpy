##############################################################################
# IMPORTS

# THIRDPARTY
import numpy as np
import pytest

# LOCAL
from .test_base import OverloadDecoratorBase_Test, OverloadWrapperBase_Test
from overload_numpy.implementors.dispatch import DispatchWrapper
from overload_numpy.implementors.func import (
    AssistsFunc,
    ImplementsFunc,
    OverloadFuncDecorator,
)

##############################################################################
# TESTS
##############################################################################


class OverloadFuncDecorator_Test(OverloadDecoratorBase_Test):
    @pytest.fixture(scope="class")
    def numpy_func(self):
        return np.concatenate

    @pytest.fixture(scope="class")
    def custom_func(self):
        def concatenate(objs, *args, **kwargs):
            return "custom"

    @pytest.fixture(scope="class", params=[ImplementsFunc, AssistsFunc])
    def OverrideCls(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def decorator_cls(self):
        return OverloadFuncDecorator

    @pytest.fixture(scope="class")
    def types(self):
        return None  # TODO! type | TypeConstraint | Collection[type | TypeConstraint] | None

    @pytest.fixture(scope="class")
    def decorator(self, decorator_cls, dispatch_on, numpy_func, overloader, types):
        return decorator_cls(dispatch_on=dispatch_on, numpy_func=numpy_func, overloader=overloader, types=types)

    # ===============================================================
    # Method Tests

    # @pytest.mark.parametrize("test", [None, int, Covariant(int), (int, Covariant(int))])
    @pytest.mark.skip(reason="TODO")
    def test__parse_types(self, decorator, types):
        assert False

    def test___call__(self, decorator, custom_func, dispatch_on, OverrideCls):
        # calling the decorator registers ``custom_func``.
        out = decorator(custom_func)

        assert out is custom_func

        dispatch_wrapper = decorator.dispatcher._dispatcher.dispatch(dispatch_on)
        assert isinstance(dispatch_wrapper, DispatchWrapper)

        info = dispatch_wrapper()
        assert isinstance(info, OverrideCls)
        assert info.func is custom_func


##############################################################################


@pytest.mark.skip(reason="TODO")
class Test_ImplementsFunc(OverloadWrapperBase_Test):
    pass


##############################################################################


@pytest.mark.skip(reason="TODO")
class Test_AssistsFunc(OverloadWrapperBase_Test):
    pass


@pytest.mark.skip(reason="TODO")
class Test_OverloadFuncDecorator(OverloadFuncDecorator_Test):
    pass
