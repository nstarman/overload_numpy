##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from collections.abc import Collection
from typing import Any, Callable, Final, Iterator, KeysView, Mapping, ValuesView

# LOCAL
from overload_numpy.constraints import Covariant, TypeConstraint
from overload_numpy.dispatch import _Dispatcher
from overload_numpy.npinfo import _NumPyInfo

# if TYPE_CHECKING:
#     # THIRD PARTY
#     from typing_extensions import TypeGuard

__all__: list[str] = []


##############################################################################
# CODE
##############################################################################


# def _is_valid_dispatch_type(cls: type) -> TypeGuard[type]:
#     return isinstance(cls, type) and not isinstance(cls, GenericAlias)


class NumPyOverloader(Mapping[Callable[..., Any], _Dispatcher]):
    """Overload :mod:`numpy` functions in ``__array_function__``."""

    _reg: Final[dict[Callable[..., Any], _Dispatcher]] = {}

    # ===============================================================
    # Mapping

    def __getitem__(self, key: Callable[..., Any]) -> _Dispatcher:
        return self._reg[key]

    def __contains__(self, o: object) -> bool:
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
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
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
        # Make single-dispatcher for numpy function
        if numpy_func not in self._reg:
            self._reg[numpy_func] = _Dispatcher()

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            """Add function to numpy overloading.

            Parameters
            ----------
            func : Callable[..., Any]
                The function.

            Returns
            -------
            func : Callable[..., Any]
                Same as ``func``.
            """
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
            dispatch_type = dispatch_on

            # Turn ``types`` into only TypeConstraint
            tinfo = self._parse_types(types, dispatch_type)
            # Note: I don't think types cannot be inferred from the function
            # because NumPy uses a dispatcher object to get the
            # ``relevant_args``.

            # Adding a new numpy function
            info = _NumPyInfo(func=func, types=tinfo, implements=numpy_func)
            # Register the function
            self._reg[numpy_func]._dispatcher.register(dispatch_type, info)
            return func

        return decorator
