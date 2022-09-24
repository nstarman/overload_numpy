"""Assistance class for ``_NumPyFuncOverloadInfo.func`` created by ``NumPyOverloads``.

This is in a separate file because it cannot be mypc compiled:
TODO! refactor when https://github.com/python/mypy/issues/13304 is fixed.
"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, final

__all__: list[str] = []  # nothing is public API


##############################################################################
# TYPING

R = TypeVar("R")


##############################################################################
# CODE
##############################################################################


@final
@dataclass(frozen=True)
class _Assists(Generic[R]):
    """Info to overload a :mod:`numpy` function.

    Parameters
    ----------
    func : Callable[..., R]
        The assisting function. Must have signature:

        .. code-block:: python

            def func(dispatch_on, numpy_func, *args, **kwargs):
                ...

    implements : Callable[..., R]
        The overloaded :mod:`numpy` function.
    dispatch_on : type[R]
        The type ``func`` returns.
    """

    func: Callable[..., R]
    """The overloading function."""

    dispatch_on: type[R]
    """The type ``func`` returns."""

    implements: Callable[..., Any]
    """The overloaded :mod:`numpy` function."""

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        """
        Call the wrapped function, providing the dispatch type, :mod:`numpy`
        function, and calling args and kwargs.

        Parameters
        ----------
        *args : Any
            Arguments into ``self.func``.
        **kwargs : Any
            Keyword arguments into ``self.func``.

        Returns
        -------
        ``R``
            The return output type of ``func``.
        """
        return self.func(self.dispatch_on, self.implements, *args, **kwargs)
