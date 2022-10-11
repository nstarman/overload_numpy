##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import Any, Protocol, runtime_checkable

__all__: list[str] = []


##############################################################################
# TYPING

# TODO! mypyc when https://github.com/mypyc/mypyc/issues/909 fixed
@runtime_checkable
class UFuncLike(Protocol):
    """Protocol for checking if an object is |ufunc|-like.

    See https://numpy.org/doc/stable/reference/ufuncs.html.
    """

    @property
    def __name__(self) -> str:
        ...

    def __call__(self, /, *args: Any, **kwargs: Any) -> Any:
        ...

    def at(self, a: Any, indices: Any, b: Any, /) -> Any:
        ...

    def accumulate(self, array: Any, axis: Any, dtype: Any, out: Any) -> Any:
        ...

    def outer(self, A: Any, B: Any, /, **kwargs: Any) -> Any:
        ...

    def reduce(self, array: Any, axis: Any, dtype: Any, out: Any, keepdims: Any, initial: Any, where: Any) -> Any:
        ...

    def reduceat(self, array: Any, indices: Any, axis: Any, dtype: Any, out: Any) -> Any:
        ...
