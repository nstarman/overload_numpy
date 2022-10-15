"""Implementations for function overrides."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Final, Generic, Mapping, TypeVar

# THIRDPARTY
from mypy_extensions import trait

# LOCAL
from overload_numpy.constraints import Covariant, TypeConstraint
from overload_numpy.implementors.dispatch import Dispatcher
from overload_numpy.utils import _get_key

if TYPE_CHECKING:
    # LOCAL
    from overload_numpy.overload import NumPyOverloader

__all__: list[str] = []


##############################################################################
# TYPING

FT = TypeVar("FT", "ImplementsFunc", "AssistsFunc")
C = TypeVar("C", bound="Callable[..., Any]")


##############################################################################
# CODE
##############################################################################


@trait
class ValidatesType:
    """Mixin to add method ``validate_types`` for numpy functions..

    Parameters
    ----------
    types: `overload_numpy.constraints.TypeConstraint` or collection thereof
        The argument types for the overloaded function. Used to check if the
        overload is valid.
    """

    # TODO! when py3.10+ add NotImplemented
    types: TypeConstraint | Collection[TypeConstraint]
    """
    The argument types for the overloaded function.
    Used to check if the overload is valid.
    """

    def validate_types(self, types: Collection[type], /) -> bool:
        """Check the types of the arguments.

        Parameters
        ----------
        types : collection[type], positional-only
            Tuple of the types of the arguments.

        Returns
        -------
        bool
            Whether the argument types work for the ``func``.
        """
        # Construct types to check.
        valid_types: Collection[TypeConstraint]
        if isinstance(self.types, TypeConstraint):
            valid_types = (self.types,)
        else:  # isinstance(self.types, Collection)
            valid_types = self.types

        # Check that each type is considered valid. e.g. `types` is (ndarray,
        # bool) and valid_types are (int, ndarray). It passes b/c ndarray <-
        # ndarray and bool <- int.
        for t in types:
            if not any(vt.validate_type(t) for vt in valid_types):
                return False
        else:
            return True


# @dataclass(frozen=True)  # TODO: when https://github.com/python/mypy/issues/13304 fixed
class OverloadFuncDecorator(Generic[FT]):
    """Decorator base class for registering an |array_function|_ overload.

    Instances of this class should not be used directly.

    Parameters
    ----------
    dispatch_on : type, keyword-only
        The class type for which the overload implementation is being
        registered.
    implements : Callable[..., Any], keyword-only
        The :mod:`numpy` function that is being overloaded.
    types : type or TypeConstraint or Collection thereof or None, keyword-only
        The types of the arguments of ``implements``. See |array_function|_ If
        `None` then ``dispatch_on`` must have class-level attribute
        ``NP_FUNC_TYPES`` specifying the types.
    overloader : |NumPyOverloader|, keyword-only
        Overloader instance.
    """

    _override_cls: Final[type[ImplementsFunc] | type[AssistsFunc]]

    # TODO: rm when https://github.com/python/mypy/issues/13304 fixed
    def __init__(
        self,
        override_cls: type[ImplementsFunc] | type[AssistsFunc],
        *,
        dispatch_on: type,
        implements: Callable[..., Any],
        types: type | TypeConstraint | Collection[type | TypeConstraint] | None,
        overloader: NumPyOverloader,
    ) -> None:
        self._override_cls = override_cls
        self._types = types
        self._dispatch_on = dispatch_on
        self._implements = implements
        self._overloader = overloader
        self.__post_init__()

    def __post_init__(self) -> None:
        # Make single-dispatcher for numpy function
        key = _get_key(self.implements)
        if key not in self.overloader._reg:
            self.overloader._reg[key] = Dispatcher[FT]()

    @property
    def types(self) -> type | TypeConstraint | Collection[type | TypeConstraint] | None:
        """`type` | `TypeConstraint` | Collection[`type` | `TypeConstraint`] | None.

        The types of the arguments of `implements`. See |array_function|_ If
        `None` then ``dispatch_on`` must have class-level attribute
        ``NP_FUNC_TYPES`` specifying the types.
        """
        return self._types

    @property
    def dispatch_on(self) -> type:
        """Class type of registrant for overload implementation."""
        return self._dispatch_on

    @property
    def implements(self) -> Callable[..., Any]:
        """Return the :mod:`numpy` function that is being overloaded."""
        return self._implements

    @property
    def overloader(self) -> NumPyOverloader:
        """Instance of |NumPyOverloader|."""
        return self._overloader

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
            The types for argument ``types`` in |array_function|_. If
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

            ts = frozenset({dispatch_type} | ts)  # Adds self type!

        # Turn `types` into only TypeConstraint
        parsed: frozenset[TypeConstraint]
        if isinstance(ts, TypeConstraint):
            parsed = frozenset((ts,))
        elif isinstance(ts, type):
            parsed = frozenset((Covariant(ts),))
        elif isinstance(ts, Collection):
            parsed = frozenset(t if isinstance(t, TypeConstraint) else Covariant(t) for t in ts)
        else:
            raise ValueError(f"types must be a {self.types}")

        return parsed

    def __call__(self, func: C, /) -> C:
        """Register an |array_function|_ overload.

        Parameters
        ----------
        func : Callable[..., Any]
            The overloading function for ``implements``.

        Returns
        -------
        func : Callable[..., Any]
            The same as the input``func``.

        Raises
        ------
        ValueError
            If ``self.types`` is the wrong type.
        AttributeError
            If ``types`` is `None` and ``dispatch_type`` does not have an
            attribute ``NP_FUNC_TYPES``. If ``dispatch_type.NP_FUNC_TYPES`` is
            not a `frozenset` of `type` or
            `overload_numpy.constraints.TypeConstraint
        """
        # Turn ``types`` into only TypeConstraint
        types = self._parse_types(self.types, self.dispatch_on)

        # Adding a new numpy function
        info = self._override_cls(func=func, types=types, implements=self.implements, dispatch_on=self.dispatch_on)

        # Register the function
        self.overloader[self.implements].register(self.dispatch_on, info)
        return func


##############################################################################


@dataclass(frozen=True)
class ImplementsFunc(ValidatesType):
    """Info to overload a :mod:`numpy` function.

    Parameters
    ----------
    func : Callable[..., Any]
        The overloading function.
    implements
        The overloaded :mod:`numpy` function.
    types: TypeConstraint or Collection[TypeConstraint]
        The argument types for the overloaded function.
        Used to check if the overload is valid.
    dispatch_on: type
        The type dispatched on. See `~overload_numpy.dispatch.Dispatcher`.

    Methods
    -------
    validate_types
        Check the types of the arguments.
    """

    implements: Callable[..., Any]
    """The overloaded :mod:`numpy` function."""

    dispatch_on: type
    """The type dispatched on. See :class:`~overload_numpy.wrapper.dispatch.Dispatcher`."""

    func: Callable[..., Any]
    """The overloading function.

    Has signature::

        func(*args, **kwargs)
    """

    # TODO! when py3.10+ add NotImplemented
    types: TypeConstraint | Collection[TypeConstraint]
    """
    The argument types for the overloaded function.
    Used to check if the overload is valid.
    """

    # TODO: parametrize return type?
    def __call__(self, _: type, /, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> object:
        """Evaluate a |numpy| function with given arguments.

        Parameters
        ----------
        calling_type : type, positional-only
            The calling type of the class for which this is the override.
        args : tuple[Any, ...]
            Tuple of arguments to pass to the override.
        kwargs : Mapping[str, Any]
            Mapping of keyword arguments to pass to the override.

        Returns
        -------
        object
            The result of evaluating the |numpy| function method.
        """
        return self.func(*args, **kwargs)


# ============================================================================


@dataclass(frozen=True)
class AssistsFunc(ValidatesType):
    """Info to overload a :mod:`numpy` function.

    Parameters
    ----------
    func : Callable[..., Any]
        The overloading function.
    implements
        The overloaded :mod:`numpy` function.
    types: TypeConstraint or Collection[TypeConstraint]
        The argument types for the overloaded function.
        Used to check if the overload is valid.
    dispatch_on: type
        The type dispatched on. See `~overload_numpy.dispatch.Dispatcher`.

    Methods
    -------
    validate_types
        Check the types of the arguments.
    """

    implements: Callable[..., Any]
    """The overloaded :mod:`numpy` function."""

    dispatch_on: type
    """The type dispatched on.

    See :class:`~overload_numpy.wrapper.dispatch.Dispatcher`.
    """

    func: Callable[..., Any]
    """The overloading function.

    Has signature::

        func(calling_type: type, implements: FunctionType, *args, **kwargs)
    """

    # TODO! when py3.10+ add NotImplemented
    types: TypeConstraint | Collection[TypeConstraint]
    """
    The argument types for the overloaded function.
    Used to check if the overload is valid.
    """

    # TODO: parametrize return type?
    def __call__(self, calling_type: type, /, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        """Evaluate a |numpy| function with given arguments.

        Parameters
        ----------
        calling_type : type, positional-only
            The calling type of the class for which this is the override.
        args : tuple[Any, ...]
            Tuple of arguments to pass to the override.
        kwargs : Mapping[str, Any]
            Mapping of keyword arguments to pass to the override.

        Returns
        -------
        object
            The result of evaluating the |numpy| function method.
        """
        return self.func(calling_type, self.implements, *args, **kwargs)
