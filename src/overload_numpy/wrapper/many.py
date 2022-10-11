##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import itertools
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

# LOCAL
from overload_numpy.utils import UFMT, UFMsT, _parse_methods
from overload_numpy.wrapper.func import AssistsFuncDecorator
from overload_numpy.wrapper.ufunc import (
    AssistsUFunc,
    AssistsUFuncDecorator,
    ImplementsUFunc,
)

__all__: list[str] = []

##############################################################################
# TYPING

C = TypeVar("C", bound=Callable[..., Any])
Self = TypeVar("Self", bound="AssistsManyDecorator")


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class AssistsManyDecorator:
    """Class for registering `~overload_numpy.NumPyOverloader.assists` (u)funcs.

    .. warning::

        Only the ``__call__`` and ``register`` methods are currently public API.
        Instances of this class are created as-needed by |NumPyOverloader|
        whenever multiple functions and |ufunc| overrides are made with
        `~overload_numpy.NumPyOverloader.assists`.

    Parameters
    ----------
    _decorators : tuple[AssistsFuncDecorator | AssistsUFuncDecorator, ...]
        `tuple` of ``AssistsFuncDecorator | AssistsUFuncDecorator``.
    """

    _decorators: tuple[AssistsUFuncDecorator | AssistsFuncDecorator, ...]
    """`tuple` of ``AssistsFuncDecorator | AssistsUFuncDecorator``."""

    def __post_init__(self) -> None:
        self.ufunc_wrappers: tuple[ImplementsUFunc | AssistsUFunc, ...] | None
        object.__setattr__(self, "ufunc_wrappers", None)  # set in `__call__`

        self.__wrapped__: Callable[..., Any]
        object.__setattr__(self, "__wrapped__", None)  # set in `__call__`

        self._is_set: bool
        object.__setattr__(self, "_is_set", False)  # set in `__call__`

    def __call__(self: Self, assists_func: Callable[..., Any], /) -> Self:
        """Register ``assists_func`` with for all overloads.

        This function can only be called once.

        Parameters
        ----------
        assists_func : Callable[..., R]
            Assistance function.

        Returns
        -------
        `overload_numpy.wrapper.many.AssistsManyDecorator`
        """
        if self._is_set:
            raise ValueError("AssistsManyDecorator can only be called once")

        object.__setattr__(self, "__wrapped__", assists_func)

        # Iterate through, evaluating the contained decorator. Just evaluating
        # the decorator is enough to activate it. We separate funcs and ufuncs,
        # because the latter are kept in attr ``ufunc_wrappers``.
        # for dec in (d for d in self.decorators if isinstance(d, AssistsFuncDecorator)):  # NOTE: mypyc incompatible
        for dec in self._decorators:
            if not isinstance(dec, AssistsFuncDecorator):
                continue
            dec(assists_func)

        ufws = tuple(dec(assists_func) for dec in self._decorators if isinstance(dec, AssistsUFuncDecorator))
        object.__setattr__(self, "ufunc_wrappers", ufws)

        object.__setattr__(self, "_is_set", True)  # prevent re-calling
        return self

    def register(self, methods: UFMsT, /) -> RegisterManyUFuncMethodDecorator:
        """Register overload for |ufunc| methods.

        Parameters
        ----------
        methods : {'__call__', 'at', 'accumulate', 'outer', 'reduce',
        'reduceat'} or set thereof.
            The names of the methods to overload. Can be a set the names.

        Returns
        -------
        decorator : `RegisterManyUFuncMethodDecorator`
            Decorator to register a function as an overload for a |ufunc| method
            (or set thereof).
        """
        ufws = self.ufunc_wrappers
        if ufws is None:
            raise ValueError("need to call this decorator first")

        return RegisterManyUFuncMethodDecorator(ufws, _parse_methods(methods))


@dataclass(frozen=True)
class RegisterManyUFuncMethodDecorator:
    """Decorator to register a |ufunc| method implementation.

    Returned by by `~overload_numpy.wrapper.ufunc.OverrideUfuncBase.register`.

    .. warning::

        Only the ``__call__`` method is public API. Instances of this class are
        created as-needed by |NumPyOverloader| if many |ufunc| overrides are
        registered. Users should not make an instance of this class.
    """

    _ufunc_wrappers: tuple[ImplementsUFunc | AssistsUFunc, ...]
    """`tuple` of ``AssistsFuncDecorator | AssistsUFuncDecorator``."""

    _applicable_methods: frozenset[UFMT]
    """|ufunc| methods for which this decorator will register overrides."""

    def __call__(self, assist_ufunc_method: C, /) -> C:
        """Decorator to register an overload funcction for |ufunc| methods.

        Parameters
        ----------
        assist_ufunc_method : Callable[..., Any]
            The overload function for specified |ufunc| methods.

        Returns
        -------
        func : Callable[..., Any]
            Unchanged.
        """
        for ufw, m in itertools.product(self._ufunc_wrappers, self._applicable_methods):
            ufw._funcs[m] = assist_ufunc_method
        return assist_ufunc_method
