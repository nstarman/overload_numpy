##############################################################################
# IMPORTS

# STDLIB

# THIRDPARTY
import numpy as np
import pytest

# LOCAL
from overload_numpy import NumPyOverloader
from overload_numpy.implementors.dispatch import Dispatcher
from overload_numpy.utils import _get_key

##############################################################################
# TESTS
##############################################################################


class OverloadWrapperBase_Test:
    @pytest.fixture(scope="class")
    def wrapper_cls(self):
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def implements(self):
        return np.add

    @pytest.fixture(scope="class")
    def dispatch_on(self):
        return int

    @pytest.fixture(scope="class")
    def wrapper(self, wrapper_cls, implements, dispatch_on):
        return wrapper_cls(implements=implements, dispatch_on=dispatch_on)

    # ===============================================================
    # Method Tests

    def test___call__(self, wrapper):
        with pytest.raises(NotImplementedError, match="need to overwrite in subclass"):
            wrapper(int)


##############################################################################


class OverloadDecoratorBase_Test:
    @pytest.fixture(scope="class")
    def OverrideCls(self):
        return object

    @pytest.fixture(scope="class")
    def dispatch_on(self):
        return int

    @pytest.fixture(scope="class")
    def numpy_func(self):
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def custom_func(self):
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def overloader(self):
        return NumPyOverloader()

    @pytest.fixture(scope="class")
    def decorator_cls(self):
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def decorator(self, decorator_cls, dispatch_on, numpy_func, overloader):
        return decorator_cls(dispatch_on=dispatch_on, numpy_func=numpy_func, overloader=overloader)

    # ===============================================================
    # Method Tests

    def test_OverrideCls(self, decorator, OverrideCls):
        assert decorator.OverrideCls is OverrideCls

    def test_dispatch_on(self, decorator, dispatch_on):
        assert decorator.dispatch_on == dispatch_on

    def test_numpy_func(self, decorator, numpy_func):
        assert decorator.numpy_func == numpy_func

    def test_dispatcher(self, decorator, numpy_func, overloader):
        # Set in __post_init__
        assert isinstance(decorator.dispatcher, Dispatcher)

        key = _get_key(numpy_func)
        assert overloader._reg[key] is decorator.dispatcher

    def test___call__(self, decorator, custom_func):
        with pytest.raises(NotImplementedError):
            decorator(custom_func)
