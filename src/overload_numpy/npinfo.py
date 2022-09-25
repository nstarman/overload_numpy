##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from collections.abc import Collection
from dataclasses import dataclass
from inspect import isclass
from typing import Any, Callable, TypeVar, final

# THIRDPARTY
from mypy_extensions import mypyc_attr

# LOCAL
from overload_numpy.constraints import TypeConstraint

__all__: list[str] = []


##############################################################################
# TYPING

Self = TypeVar("Self")


##############################################################################
# CODE
##############################################################################


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass(frozen=True)
class _NotDispatched(TypeConstraint):
    """A TypeConstraint that always validates to `False`.

    .. todo::

        This TypeConstraint is necessary since python<3.10 does not support
        ``NotImplementedType``. When the minimum version is py3.10 then this class
        should be deprecated in favor of `NotImplemented` as the special flag.
    """

    def validate_type(self, arg_type: type, /) -> bool:
        return False  # never true


_NOT_DISPATCHED = _NotDispatched()


##############################################################################


@final
@dataclass(frozen=True)
class _NumPyFuncOverloadInfo:
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
        The type dispatched on. See `~overload_numpy.dispatch._Dispatcher`.

    Methods
    -------
    validate_types
        Check the types of the arguments.
    """

    func: Callable[..., Any]
    """The overloading function."""

    implements: Callable[..., Any]
    """The overloaded :mod:`numpy` function."""

    # TODO! when py3.10+ add NotImplemented
    types: TypeConstraint | Collection[TypeConstraint]
    """
    The argument types for the overloaded function.
    Used to check if the overload is valid.
    """

    dispatch_on: type
    """The type dispatched on. See ``_Dispatcher``."""

    def __post_init__(self) -> None:
        # Validate
        if not callable(self.func):
            raise TypeError(f"func must be callable, not {self.func!r}")
        elif not callable(self.implements):
            raise TypeError(f"implements must be callable, not {self.implements!r}")
        elif not isinstance(self.types, TypeConstraint) and not (
            isinstance(self.types, Collection) and all(isinstance(t, TypeConstraint) for t in self.types)
        ):
            raise TypeError(f"types must be a TypeConstraint, or collection thereof, not {self.types}")
        elif not isclass(self.dispatch_on):
            raise TypeError(f"dispatch_on must be a type, not {self.dispatch_on}")

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
        # special cased for not dispatched.
        # TODO! py3.10+ -> `self.types is NotImplemented`
        if isinstance(self.types, _NotDispatched):
            return False

        # Construct types to check.
        valid_types: Collection[TypeConstraint]
        if isinstance(self.types, TypeConstraint):
            valid_types = (self.types,)
        else:  # isinstance(self.types, Collection)
            valid_types = self.types

        # Check that each type is considred valid. e.g. `types` is (ndarray,
        # bool) and valid_types are (int, ndarray). It passes b/c ndarray <-
        # ndarray and bool <- int.
        for t in types:
            if not any(vt.validate_type(t) for vt in valid_types):
                return False
        else:
            return True

    def __call__(self: Self, *args: Any) -> Self:
        """Return self.

        Used for `~functools.singledispatch` in
        `~overload_numpy.dispatch._Dispatcher`.
        """
        return self
