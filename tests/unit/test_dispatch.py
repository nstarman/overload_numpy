##############################################################################
# IMPORTS

# THIRD PARTY
import pytest

# LOCAL
from overload_numpy.dispatch import _Dispatcher, _notdispatched, _notdispatched_info
from overload_numpy.npinfo import _NOT_DISPATCHED, _NumPyInfo

##############################################################################
# TESTS
##############################################################################


def test__notdispatched() -> None:
    """Test :func:`overload_numpy.overload._notdispatched`."""
    with pytest.raises(NotImplementedError, match="not dispatched"):
        _notdispatched()


def test__notdispatched_info() -> None:
    """Test :func:`overload_numpy.overload._notdispatched_info`."""
    assert isinstance(_notdispatched_info, _NumPyInfo)
    assert _notdispatched_info.func is _notdispatched
    assert _notdispatched_info.implements is _notdispatched
    assert _notdispatched_info.types is _NOT_DISPATCHED


class Test__Dispatcher:
    """Test :class:`overload_numpy.overload._Dispatcher`."""

    def test___init__(self):
        dispatcher = _Dispatcher()

        assert isinstance(dispatcher, _Dispatcher)
        assert hasattr(dispatcher, "_dispatcher")

        npinfo = dispatcher._dispatcher(object())

        assert npinfo.func is _notdispatched
        assert npinfo.implements is _notdispatched
        assert npinfo.types is _NOT_DISPATCHED

    def test___call__(self):
        # singledispatch is well tested, so only need to test that __call__ calls dispaatcher.

        dispatcher = _Dispatcher()
        dispatcher._dispatcher.register(int, lambda *args: _notdispatched_info)

        obj = object()
        assert dispatcher(obj) is dispatcher._dispatcher(obj)
        assert dispatcher(obj) is _notdispatched_info

        obj = 1
        assert dispatcher(obj) is dispatcher._dispatcher(obj)
        assert dispatcher(obj) is _notdispatched_info
