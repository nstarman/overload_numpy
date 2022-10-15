"""Utilities."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import Any, Callable, Final, Literal, Set, Union

# LOCAL
from overload_numpy._typeutils import UFuncLike

__all__: list[str] = []


##############################################################################
# TYPING

UFMT = Literal["__call__", "at", "accumulate", "outer", "reduce", "reduceat"]
UFMsT = Union[UFMT, Set[UFMT]]


##############################################################################
# PARAMETERS

VALID_UFUNC_METHODS: Final[set[UFMT]] = {
    "__call__",
    "at",
    "accumulate",
    "outer",
    "reduce",
    "reduceat",
}


##############################################################################
# CODE
##############################################################################


def _parse_methods(methods: UFMsT) -> frozenset[UFMT]:
    """Parse |ufunc| method.

    Parameters
    ----------
    methods : {'__call__', 'at', 'accumulate', 'outer', 'reduce'}
        The |ufunc| method name.

    Returns
    -------
    set[str]
        Set of parsed ``methods``.

    Raises
    ------
    ValueError
        If any of ``methods`` is not one of the allowedd types.
    """
    ms: set[UFMT] = {methods} if isinstance(methods, str) else methods
    # validation that each elt is a UFUNC method type
    if any(m not in VALID_UFUNC_METHODS for m in ms):
        raise ValueError(f"methods must be an element or subset of {VALID_UFUNC_METHODS}, not {ms}")
    return frozenset(ms)


def _get_key(key: str | UFuncLike | Callable[..., Any] | Any) -> str:
    """Get the key.

    Parameters
    ----------
    key : str or Callable[..., Any]
        The key.

    Returns
    -------
    str
        The name of the module and function.

    Raises
    ------
    ValueError
        If the key is not one of the known types.
    """
    if isinstance(key, str):
        return key
    elif isinstance(key, UFuncLike):
        return f"{key.__class__.__module__}.{key.__name__}"
    elif callable(key):
        return f"{key.__module__}.{key.__name__}"
    else:
        raise ValueError(f"the key {key} is not a str or Callable[..., Any]")
