##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from functools import singledispatch
from typing import TYPE_CHECKING, Any, NoReturn, final

# LOCAL
from overload_numpy.npinfo import _NOT_DISPATCHED, _NumPyFuncOverloadInfo

if TYPE_CHECKING:
    # STDLIB
    import functools

__all__: list[str] = []


##############################################################################
# CODE
##############################################################################


# A stand-in function for a not-dispatched `_NumPyFuncOverloadInfo`. If
# `_NumPyFuncOverloadInfo.validate_types` is used in `__array_function__` then this function
# will never be called.
def _notdispatched(*args: Any, **kwarg: Any) -> NoReturn:
    raise NotImplementedError("not dispatched")


_notdispatched_info = _NumPyFuncOverloadInfo(
    func=_notdispatched, implements=_notdispatched, types=_NOT_DISPATCHED, dispatch_on=object
)
# The not-dispatched `_NumPyFuncOverloadInfo`. All `_Dispatcher` start with this as the base
# `_NumPyFuncOverloadInfo`.


@final
class _Dispatcher:
    """`~functools.singledispatch` instance."""

    def __init__(self) -> None:
        # Make default NotImplementedError
        # `~overload_numpy.npinfo._NumPyFuncOverloadInfo` This return value is
        # necessary until the functional form of singledispatch works with MyPy
        # for adding different return types. Until then this is restricted to
        # the one return type. This cannot be built in mypyc v0.971
        @singledispatch
        def dispatcher(obj: object, /) -> _NumPyFuncOverloadInfo:
            return _notdispatched_info

        self._dispatcher: functools._SingleDispatchCallable[_NumPyFuncOverloadInfo]
        self._dispatcher = dispatcher

    def __call__(self, obj: object, /) -> _NumPyFuncOverloadInfo:
        """Get correct `~overload_numpy.npinfo._NumPyFuncOverloadInfo` for the calling object's type."""
        npinfo: _NumPyFuncOverloadInfo = self._dispatcher(obj)
        return npinfo
