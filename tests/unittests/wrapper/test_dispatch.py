##############################################################################
# IMPORTS

# STDLIB
import pickle
from copy import copy, deepcopy

# THIRDPARTY
import pytest

# LOCAL
from overload_numpy.implementors.dispatch import Dispatcher

##############################################################################
# TESTS


class Test_Dispatcher:
    """Test :class:`overload_numpy.overload.Dispatcher`."""

    @pytest.fixture(scope="class")
    def dispatcher_cls(self) -> type:
        return Dispatcher

    @pytest.fixture(scope="class")
    def dispatcher(self, dispatcher_cls) -> type:
        return dispatcher_cls()

    # ===============================================================
    # Method Tests

    def test___init__(self, dispatcher_cls):
        dispatcher = dispatcher_cls()

        assert isinstance(dispatcher, dispatcher_cls)
        assert hasattr(dispatcher, "_dspr")

    def test___call__(self, dispatcher_cls):
        # singledispatch is well tested, so only need to test that __call__ calls dispatcher.

        dispatcher = dispatcher_cls()

        # default implementation is to NotImplementedError
        obj = object()
        with pytest.raises(NotImplementedError):
            dispatcher(obj)

        obj = 1
        with pytest.raises(NotImplementedError):
            dispatcher(obj)

        # let's add something to the dispatcher
        dispatcher._dspr.register(int, lambda *args, **kwargs: "dispatched")
        assert dispatcher(obj) == "dispatched"

    # ===============================================================
    # Usage Tests

    @pytest.mark.incompatible_with_mypyc
    @pytest.mark.xfail
    def test_copy(self, dispatcher) -> None:
        # TODO! get copy working.
        # copying
        assert copy(dispatcher) == dispatcher
        assert deepcopy(dispatcher) == dispatcher

    @pytest.mark.incompatible_with_mypyc
    @pytest.mark.xfail
    def test_serialization(self, dispatcher) -> None:
        # TODO! get serialization working.
        # pickling
        dumps = pickle.dumps(dispatcher)
        assert pickle.loads(dumps) == dispatcher
