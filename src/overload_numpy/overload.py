##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from abc import ABCMeta
from collections.abc import Collection
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Final,
    Iterator,
    KeysView,
    Mapping,
    TypeVar,
    ValuesView,
)

# LOCALFOLDER
from overload_numpy.assists import Assists
from overload_numpy.constraints import Covariant, TypeConstraint
from overload_numpy.dispatch import _Dispatcher
from overload_numpy.npinfo import _NumPyInfo

__all__: list[str] = []

##############################################################################
# TYPING

C = TypeVar("C", bound=Callable[..., Any])
R = TypeVar("R")


##############################################################################
# CODE
##############################################################################


class NumPyOverloader(Mapping[Callable[..., Any], _Dispatcher]):
    """Overload :mod:`numpy` functions in ``__array_function__``."""

    _reg: Final[dict[Callable[..., Any], _Dispatcher]] = {}

    # ===============================================================
    # Mapping

    def __getitem__(self, key: Callable[..., Any], /) -> _Dispatcher:
        return self._reg[key]

    def __contains__(self, o: object, /) -> bool:
        return o in self._reg

    def __iter__(self) -> Iterator[Callable[..., Any]]:
        return iter(self._reg)

    def __len__(self) -> int:
        return len(self._reg)

    def keys(self) -> KeysView[Callable[..., Any]]:
        return self._reg.keys()

    def values(self) -> ValuesView[_Dispatcher]:
        return self._reg.values()

    # ===============================================================

    def _parse_types(
        self,
        types: type | TypeConstraint | Collection[type | TypeConstraint] | None,
        dispatch_type: type,
        /,
    ) -> frozenset[TypeConstraint]:
        """Parse types argument ``.implements()``.

        Parameters
        ----------
        types : type or TypeConstraint or Collection[type | TypeConstraint] or
        None
            The types for argument ``types`` in ``__array_function__``. If
            `None`, then ``dispatch_type`` must have class-level attribute
            ``NP_FUNC_TYPES`` that is a `frozenset` of `type` or
            `overload_numpy.constraints.TypeConstraint`.

        Returns
        -------
        frozenset[TypeConstraint]
            The set of TypeConstraint containing the constraints ono the types.

        Raises
        ------
        ValueError
            If ``types`` is the wrong type.
        AttributeError
            If ``types`` is `None` and ``dispatch_type`` does not have an
            attribute ``NP_FUNC_TYPES``.
            If ``dispatch_type.NP_FUNC_TYPES`` is not a `frozenset` of `type` or
            `overload_numpy.constraints.TypeConstraint
        """
        if types is not None:
            ts = types
        elif not hasattr(dispatch_type, "NP_FUNC_TYPES"):
            raise AttributeError(f"if types is None {dispatch_type} must have class-attribute ``NP_FUNC_TYPES``")
        else:
            ts = getattr(dispatch_type, "NP_FUNC_TYPES")

            if not isinstance(ts, frozenset) or not all(isinstance(t, (type, TypeConstraint)) for t in ts):
                raise AttributeError(
                    f"if types is None ``{dispatch_type}.NP_FUNC_TYPES`` must be frozenset[types | TypeConstraint]"
                )

            ts = frozenset({dispatch_type} | ts)  # Add self type!

        # Turn `types` into only TypeConstraint
        parsed: frozenset[TypeConstraint]
        # if types == "self":
        #     parsed = frozenset((Covariant(dispatch_type),))
        if isinstance(ts, TypeConstraint):
            parsed = frozenset((ts,))
        elif isinstance(ts, type):
            parsed = frozenset((Covariant(ts),))
        elif isinstance(ts, Collection):
            parsed = frozenset(t if isinstance(t, TypeConstraint) else Covariant(t) for t in ts)
        else:
            raise ValueError(f"types must be a {self.implements.__annotations__['types']}")

        return parsed

    def implements(
        self,
        numpy_func: Callable[..., Any],
        /,
        dispatch_on: type,
        *,
        types: type | TypeConstraint | Collection[type | TypeConstraint] | None = None,
    ) -> ImplementsDecorator:
        """Register an __array_function__ implementation object.

        This is a decorator factory, returning ``decorator``, which registers
        the decorated function as an overload method for :mod:`numpy` function
        ``numpy_func`` for a class of type ``dispatch_on``.

        Parameters
        ----------
        numpy_func : callable[..., Any], positional-only
            The :mod:`numpy` function that is being overloaded.
        dispatch_on : type
            The class type for which the overload implementation is being
            registered.
        types : type or TypeConstraint or Collection thereof or None,
        keyword-only
            The types of the arguments of `numpy_func`. See
            https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_function__
            If `None` then ``dispatch_on`` must have class-level attribute
            ``NP_FUNC_TYPES`` specifying the types.

        Returns
        -------
        decorator : callable[[callable], callable]
            Function to register a wrapped function.
        """
        return ImplementsDecorator(self, numpy_func=numpy_func, types=types, dispatch_on=dispatch_on)

    def assists(
        self,
        numpy_func: Callable[..., Any],
        /,
        dispatch_on: type,
        *,
        types: type | TypeConstraint | Collection[type | TypeConstraint] | None = None,
    ) -> AssistsDecorator:
        # - TODO! make it work on many numpy funcs
        return AssistsDecorator(self, numpy_func=numpy_func, types=types, dispatch_on=dispatch_on)


##############################################################################
# Decorators for implementations


@dataclass(frozen=True)
class OverloadDecoratorBase(metaclass=ABCMeta):

    overloader: NumPyOverloader
    numpy_func: Callable[..., Any]
    types: type | TypeConstraint | Collection[type | TypeConstraint] | None
    dispatch_on: type

    def __post_init__(self) -> None:
        # Make single-dispatcher for numpy function
        if self.numpy_func not in self.overloader._reg:
            self.overloader._reg[self.numpy_func] = _Dispatcher()

    # @abstractmethod  # TODO: fix when https://github.com/python/mypy/issues/5374 released
    def _func_hook(self, func: Callable[..., Any]) -> Callable[..., Any]:
        raise NotImplementedError

    def __call__(self, func: C) -> C:
        # TODO? infer dispatch_on from 1st argument
        # if dispatch_on is None:  # Get from 1st argument
        #     argname, dispatch_type = next(iter(get_type_hints(func).items()))
        #     if not _is_valid_dispatch_type(dispatch_type):
        #         raise TypeError(
        #             f"Invalid annotation for {argname!r}. "
        #             f"{dispatch_type!r} is not a class."
        #         )
        # else:
        #     dispatch_type = dispatch_on

        # Turn ``types`` into only TypeConstraint
        tinfo = self.overloader._parse_types(self.types, self.dispatch_on)
        # Note: I don't think types cannot be inferred from the function
        # because NumPy uses a dispatcher object to get the
        # ``relevant_args``.

        # Adding a new numpy function
        info = _NumPyInfo(
            func=self._func_hook(func), types=tinfo, implements=self.numpy_func, dispatch_on=self.dispatch_on
        )
        # Register the function
        self.overloader._reg[self.numpy_func]._dispatcher.register(self.dispatch_on, info)
        return func


@dataclass(frozen=True)
class ImplementsDecorator(OverloadDecoratorBase):
    def _func_hook(self, func: C) -> C:
        return func


@dataclass(frozen=True)
class AssistsDecorator(OverloadDecoratorBase):
    def _func_hook(self, func: Callable[..., R]) -> Assists[R]:
        return Assists(func=func, implements=self.numpy_func, dispatch_on=self.dispatch_on)
