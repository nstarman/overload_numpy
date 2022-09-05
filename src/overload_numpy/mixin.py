##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Collection

# THIRD PARTY
from mypy_extensions import mypyc_attr

if TYPE_CHECKING:
    # LOCAL
    from overload_numpy.constraints import TypeConstraint
    from overload_numpy.overload import NumPyOverloader


##############################################################################
# CODE
##############################################################################


@mypyc_attr(allow_interpreted_subclasses=True)
class NDFunctionMixin:
    """Mixin for adding ``__array_function__``.

    Attributes
    ----------
    NP_FUNC_OVERLOADS : `overload_numpy.NumPyOverloader`
        A class-attribute of an instance of ``NumPyOverloader``.
    NP_FUNC_TYPES : frozenset[type | TypeConstraint] | None, optional
        A class-attribute of `None` or a `frozenset` of `type` or
        `overload_numpy.constraints.TypeConstraint`.

    Examples
    --------
    First, some imports:

        >>> from dataclasses import dataclass
        >>> from typing import ClassVar
        >>> import numpy as np
        >>> from overload_numpy import NumPyOverloader, NDFunctionMixin

    Now we can define a `overload_numpy.NumPyOverloader` instance:

        >>> VEC_FUNCS = NumPyOverloader()

    The overloads apply to an array wrapping class. Let's define one:

        >>> @dataclass
        ... class Vector1D(NDFunctionMixin):
        ...     '''A simple array wrapper.'''
        ...     x: np.ndarray
        ...     NP_FUNC_OVERLOADS: ClassVar[NumPyOverloader] = VEC_FUNCS

    Now :mod:`numpy` functions can be overloaded and registered for ``Vector1D``.

        >>> @VEC_FUNCS.implements(np.concatenate, Vector1D)
        ... def concatenate(vec1ds: tuple[Vector1D, ...]) -> Vector1D:
        ...     return Vector1D(np.concatenate(tuple(v.x for v in vec1ds)))

    Time to check this works:

        >>> vec1d = Vector1D(np.arange(3))
        >>> newvec = np.concatenate((vec1d, vec1d))
        >>> newvec
        Vector1D(x=array([0, 1, 2, 0, 1, 2]))

    Great!
    """

    NP_FUNC_OVERLOADS: ClassVar[NumPyOverloader]
    NP_FUNC_TYPES: ClassVar[frozenset[type | TypeConstraint] | None] = frozenset()
    # empty frozenset -> self type.

    def __array_function__(
        self, func: Callable[..., Any], types: Collection[type], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        # Check if can be dispatched
        if func not in self.NP_FUNC_OVERLOADS:
            return NotImplemented

        # Get _NumPyInfo on function, given type of self
        finfo = self.NP_FUNC_OVERLOADS[func](self)

        # Note: this allows subclasses that don't override `__array_function__`
        # to handle this object type.
        #
        # From https://numpy.org/devdocs/reference/arrays.classes.html:
        # "Implementations should not rely on the iteration order of types."
        if not finfo.validate_types(types):
            return NotImplemented

        # TODO! validate args and kwargs.

        return finfo.func(*args, **kwargs)
