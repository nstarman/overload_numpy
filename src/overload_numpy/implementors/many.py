"""Implementation of overrides for many functions and |ufunc|."""


##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import itertools
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ItemsView,
    Iterator,
    KeysView,
    Mapping,
    TypeVar,
    Union,
    ValuesView,
)

# LOCAL
from overload_numpy.implementors.func import AssistsFunc, OverloadFuncDecorator
from overload_numpy.implementors.ufunc import AssistsUFunc, OverloadUFuncDecorator
from overload_numpy.utils import UFMT, UFMsT, _get_key, _parse_methods

__all__: list[str] = []

##############################################################################
# TYPING

C = TypeVar("C", bound=Callable[..., Any])
Self = TypeVar("Self", bound="AssistsManyDecorator")
V = Union[OverloadUFuncDecorator[AssistsUFunc], OverloadFuncDecorator[AssistsFunc]]


##############################################################################
# CODE
##############################################################################


# @dataclass(frozen=True)  # todo when mypyc happy
class AssistsManyDecorator(Mapping[str, V]):
    """Class for registering `~overload_numpy.NumPyOverloader.assists` (u)funcs.

    .. warning::

        Only the ``__call__`` and ``register`` methods are currently public API.
        Instances of this class are created as-needed by |NumPyOverloader|
        whenever multiple functions and |ufunc| overrides are made with
        `~overload_numpy.NumPyOverloader.assists`.

    Parameters
    ----------
    _decorators : dict[str, OverloadFuncDecorator | OverloadUFuncDecorator]
        :class:`dict` of ``OverloadFuncDecorator[AssistsFunc] |
        OverloadUFuncDecorator[AssistsUFunc]``.
    """

    def __init__(self, decorators: dict[str, V], /) -> None:
        self._decorators = decorators

        self._ufunc_wrappers: dict[str, AssistsUFunc] = {}  # set in `__call__`
        self.__wrapped__: Callable[..., Any] | None = None  # set in `__call__`
        self._is_set: bool = False  # set in `__call__`

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

        self.__wrapped__ = assists_func

        # Iterate through, evaluating the contained decorator. Just evaluating
        # the decorator is enough to activate it. We separate funcs and ufuncs,
        # because the latter are kept in attr ``ufunc_wrappers``.
        for dec in self._decorators.values():
            if not isinstance(dec, OverloadFuncDecorator):
                continue
            dec(assists_func)

        ufws = {k: dec(assists_func) for k, dec in self._decorators.items() if isinstance(dec, OverloadUFuncDecorator)}
        self._ufunc_wrappers = ufws

        self._is_set = True  # prevent re-calling
        return self

    def register(self, methods: UFMsT, /) -> RegisterManyUFuncMethodDecorator:
        """Register override for |ufunc| methods.

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
        if self._is_set is False:
            raise ValueError("need to call this decorator first")

        return RegisterManyUFuncMethodDecorator(self._ufunc_wrappers, _parse_methods(methods))

    # ===============================================================
    # Mapping

    def __getitem__(self, key: str | Callable[..., Any], /) -> V:
        return self._decorators[_get_key(key)]

    def __contains__(self, o: object, /) -> bool:
        return _get_key(o) in self._decorators

    def __iter__(self) -> Iterator[str]:
        return iter(self._decorators)

    def __len__(self) -> int:
        return len(self._decorators)

    def keys(self) -> KeysView[str]:
        """Return functions (str representations) which have overrides."""
        return self._decorators.keys()

    def values(self) -> ValuesView[OverloadUFuncDecorator[AssistsUFunc] | OverloadFuncDecorator[AssistsFunc]]:
        """Return override implementors for functions and |ufunc|."""
        return self._decorators.values()

    def items(self) -> ItemsView[str, OverloadUFuncDecorator[AssistsUFunc] | OverloadFuncDecorator[AssistsFunc]]:
        """Return view of function (str representations) and override implementors."""
        return self._decorators.items()


@dataclass(frozen=True)
class RegisterManyUFuncMethodDecorator:
    """Decorator to register a |ufunc| method implementation.

    Returned by by `~overload_numpy.wrapper.ufunc.OverrideUfuncBase.register`.

    .. warning::

        Only the ``__call__`` method is public API. Instances of this class are
        created as-needed by |NumPyOverloader| if many |ufunc| overrides are
        registered. Users should not make an instance of this class.
    """

    _ufunc_wrappers: dict[str, AssistsUFunc]
    _applicable_methods: frozenset[UFMT]

    def __call__(self, assist_ufunc_method: C, /) -> C:
        """Register an overload function for |ufunc| methods.

        Parameters
        ----------
        assist_ufunc_method : Callable[..., Any]
            The overload function for specified |ufunc| methods.

        Returns
        -------
        func : Callable[..., Any]
            Unchanged.
        """
        for ufw, m in itertools.product(self._ufunc_wrappers.values(), self._applicable_methods):
            ufw._funcs[m] = assist_ufunc_method
        return assist_ufunc_method
