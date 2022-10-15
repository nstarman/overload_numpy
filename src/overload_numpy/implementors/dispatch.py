"""`~functools.singledispatch` wrapping for overrides.

.. todo::

    - mypyc compile when https://github.com/python/mypy/issues/13613 and
      https://github.com/python/mypy/issues/13304 are resolved.
"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Generic, TypeVar, final

if TYPE_CHECKING:
    # STDLIB
    import functools

    # LOCAL
    from overload_numpy.implementors.func import AssistsFunc, ImplementsFunc
    from overload_numpy.implementors.ufunc import AssistsUFunc, ImplementsUFunc

__all__: list[str] = []

##############################################################################
# TYPING

WT = TypeVar("WT", "ImplementsFunc", "AssistsFunc", "ImplementsUFunc", "AssistsUFunc")
# All_Dispatchers = Union[
#     "Dispatcher[ImplementsUFunc]", "Dispatcher[ImplementsFunc]", "Dispatcher[AssistsFunc]", "Dispatcher[AssistsUFunc]"
# ]  TODO! parametrization of Dispatcher.  See moved definition below


##############################################################################
# CODE
##############################################################################


@final
class Dispatcher(Generic[WT]):
    """`~functools.singledispatch` instance."""

    def __init__(self) -> None:
        @singledispatch
        def dispatcher(obj: object, /) -> WT:
            raise NotImplementedError  # See Mixin for handling.

        self._dspr: functools._SingleDispatchCallable[WT]
        self._dspr = dispatcher

    def __call__(self, obj: object, /) -> WT:
        """Get wrapper for the object's type.

        Get correct :mod:`~overload_numpy.wrapper` wrapper for the calling
        object's type.

        Parameters
        ----------
        obj : object, positional-only
            object for calling the `~functools.singledispatch` .

        Returns
        -------
        ``WT``
            One of `overload_numpy.wrapper.func.ImplementsFunc`,
            `overload_numpy.wrapper.func.AssistsFunc`,
            `overload_numpy.wrapper.ufunc.ImplementsUFunc`,
            `overload_numpy.wrapper.ufunc.AssistsUFunc`, depending on the type
            parameterization of this generic class.
        """
        return self._dspr(obj)

    def register(self, cls: type, func: WT) -> None:
        """Register ``func``, wrapped with `DispatchWrapper`.

        Parameters
        ----------
        cls : type
            The type on which to dispatch.
        func : ``WT``
            One of `overload_numpy.wrapper.func.ImplementsFunc`,
            `overload_numpy.wrapper.func.AssistsFunc`,
            `overload_numpy.wrapper.ufunc.ImplementsUFunc`,
            `overload_numpy.wrapper.ufunc.AssistsUFunc`, depending on the type
            parameterization of this generic class.
        """
        self._dspr.register(cls, DispatchWrapper(func))
        return None


@final
@dataclass(frozen=True)
class DispatchWrapper(Generic[WT]):
    """Wrap dispatche to return, not call, the function.

    :func:`~functools.singledispatch` calls the dispatched functions.
    This wraps that function so the single-dispatch instead returns the function.

    Parameters
    ----------
    __wrapped__ : `ImplementsFunc` or `AssistsFunc` or `ImplementsUFunc` or `AssistsUFunc`
        The result of calling ``Dispatch``.
    """

    __wrapped__: WT  # Dispatch wrapper

    def __call__(self, *_: Any, **__: Any) -> WT:
        """Return ``__wrapped__``, ignoring input."""
        return self.__wrapped__  # `Dispatch` wrapper


All_Dispatchers = Dispatcher[Any]  # TODO: parametrization of Dispatcher
