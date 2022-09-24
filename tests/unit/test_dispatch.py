##############################################################################
# IMPORTS

# STDLIB
import pickle
from copy import copy, deepcopy

# THIRDPARTY
import pytest

# LOCAL
from overload_numpy.dispatch import _Dispatcher, _notdispatched, _notdispatched_info
from overload_numpy.npinfo import _NOT_DISPATCHED, _NumPyFuncOverloadInfo

##############################################################################
# TESTS
##############################################################################


def test__notdispatched() -> None:
    """Test :func:`overload_numpy.overload._notdispatched`."""
    with pytest.raises(NotImplementedError, match="not dispatched"):
        _notdispatched()


def test__notdispatched_info() -> None:
    """Test :func:`overload_numpy.overload._notdispatched_info`."""
    assert isinstance(_notdispatched_info, _NumPyFuncOverloadInfo)
    assert _notdispatched_info.func is _notdispatched
    assert _notdispatched_info.implements is _notdispatched
    assert _notdispatched_info.types is _NOT_DISPATCHED


class Test__Dispatcher:
    """Test :class:`overload_numpy.overload._Dispatcher`."""

    @pytest.fixture(scope="class")
    def dispatcher_cls(self) -> type:
        return _Dispatcher

    @pytest.fixture(scope="class")
    def dispatcher(self, dispatcher_cls) -> type:
        return dispatcher_cls()

    # ===============================================================
    # Method Tests

    def test___init__(self, dispatcher_cls):
        dispatcher = dispatcher_cls()

        assert isinstance(dispatcher, dispatcher_cls)
        assert hasattr(dispatcher, "_dispatcher")

        npinfo = dispatcher._dispatcher(object())

        assert npinfo.func is _notdispatched
        assert npinfo.implements is _notdispatched
        assert npinfo.types is _NOT_DISPATCHED

    def test___call__(self, dispatcher_cls):
        # singledispatch is well tested, so only need to test that __call__ calls dispaatcher.

        dispatcher = dispatcher_cls()
        dispatcher._dispatcher.register(int, lambda *args: _notdispatched_info)

        obj = object()
        assert dispatcher(obj) is dispatcher._dispatcher(obj)
        assert dispatcher(obj) is _notdispatched_info

        obj = 1
        assert dispatcher(obj) is dispatcher._dispatcher(obj)
        assert dispatcher(obj) is _notdispatched_info

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
