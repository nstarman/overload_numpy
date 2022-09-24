"""Assistance class for ``_NumPyInfo.func`` created by ``NumPyOverloads``.

This is in a separate file because it cannot be mypc compiled:
TODO! refactr when https://github.com/python/mypy/issues/13304 is fixed
"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, final

__all__: list[str] = []


##############################################################################
# TYPING

R = TypeVar("R")


##############################################################################
# CODE
##############################################################################


@final
@dataclass(frozen=True)
class Assists(Generic[R]):
    """Info to overload a :mod:`numpy` function."""

    func: Callable[..., R]
    """The overloading function."""

    implements: Callable[..., Any]
    """The overloaded :mod:`numpy` function."""

    dispatch_on: type[R]

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        return self.func(self.dispatch_on, self.implements, *args, **kwargs)
