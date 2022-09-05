##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, TypeVar, final

# THIRD PARTY
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
class _NotDispatched(TypeConstraint):
    """A TypeConstraint that always validates to `False`.

    This TypeConstraint is necessary since python<3.10 does not support
    ``NotImplementedType``. When the minimum version is py3.10 then this class
    should be deprecated in favor of `NotImplemented` as the special flag.
    """

    _instance: ClassVar[_NotDispatched]

    def __init_subclass__(cls) -> None:
        raise NotImplementedError("`_NotDispatched` cannot be subclassed")

    def __new__(cls: type[_NotDispatched]) -> _NotDispatched:
        # Make a singleton
        if not hasattr(cls, "_instance"):
            self = super().__new__(cls)
            cls._instance = self

        return cls._instance

    def validate_type(self, arg_type: type) -> bool:
        return False  # never true


_NOT_DISPATCHED = _NotDispatched()


##############################################################################


@final
@dataclass(frozen=True)
class _NumPyInfo:
    """Info to overload a :mod:`numpy` function."""

    func: Callable[..., Any]
    """The overloading function."""

    implements: Callable[..., Any]
    """The overloaded :mod:`numpy` function."""

    types: TypeConstraint | Collection[TypeConstraint]
    """
    The argument types for the overloaded function.
    Used to check if the overload is valid.

    :todo: py3.10+ add NotImplemented
    """

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

    def validate_types(self, types: Collection[type]) -> bool:
        """Check the types of the arguments.

        Parameters
        ----------
        types : collection[type]
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
        """Return self. Used for singledispatch in _Dispatcher."""
        return self
