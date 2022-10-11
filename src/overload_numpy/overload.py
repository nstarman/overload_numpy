##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import Container  # noqa: F401
from typing import Iterable  # noqa: F401
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Iterator,
    KeysView,
    Mapping,
    ValuesView,
    overload,
)

# LOCAL
from overload_numpy._typeutils import UFuncLike
from overload_numpy.utils import UFMsT, _get_key, _parse_methods
from overload_numpy.wrapper.dispatch import Dispatcher

if TYPE_CHECKING:
    # STDLIB
    from collections.abc import Callable
    from types import FunctionType

    # LOCAL
    from overload_numpy.constraints import TypeConstraint
    from overload_numpy.wrapper.dispatch import All_Dispatchers
    from overload_numpy.wrapper.func import (
        AssistsFuncDecorator,
        ImplementsFuncDecorator,
    )
    from overload_numpy.wrapper.many import AssistsManyDecorator
    from overload_numpy.wrapper.ufunc import (
        AssistsUFuncDecorator,
        ImplementsUFuncDecorator,
    )


__all__: list[str] = []


##############################################################################
# CODE
##############################################################################


# @dataclass(frozen=True)  # TODO: when
# https://github.com/python/mypy/issues/13304 fixed
#
# Dispatcher[ImplementsUFunc] | Dispatcher[ImplementsFunc] |
# Dispatcher[AssistsFunc] | Dispatcher[AssistsUFunc]  # TODO: when py3.10+
# https://bugs.python.org/issue42233
class NumPyOverloader(Mapping[str, Dispatcher[Any]]):
    """
    Register :mod:`numpy` function overrides.

    This mapping works in conjunction with a mixin
    (:class:`~overload_numpy.NPArrayFuncOverloadMixin`,
    :class:`~overload_numpy.NPArrayUFuncOverloadMixin`,
    or :class:`~overload_numpy.NPArrayOverloadMixin`) to register and implement
    overrides with |array_function|_ and |array_ufunc|_.

    Examples
    --------
    First, some imports:

        >>> from dataclasses import dataclass, fields
        >>> from typing import ClassVar
        >>> import numpy as np
        >>> from overload_numpy import NumPyOverloader, NPArrayFuncOverloadMixin

    Now we can define a |NumPyOverloader| instance:

        >>> W_FUNCS = NumPyOverloader()

    The overloads apply to an array wrapping class. Let's define one:

        >>> @dataclass
        ... class Wrap1D(NPArrayFuncOverloadMixin):
        ...     '''A simple array wrapper.'''
        ...     x: np.ndarray
        ...     NP_OVERLOADS: ClassVar[NumPyOverloader] = W_FUNCS

    Now ``numpy`` functions can be overloaded and registered for ``Wrap1D``.

        >>> @W_FUNCS.implements(np.concatenate, Wrap1D)
        ... def concatenate(vecs):
        ...     VT = type(vecs[0])
        ...     return VT(*(np.concatenate(tuple(getattr(v, f.name) for v in vecs))
        ...                 for f in fields(VT)))

    Time to check this works:

        >>> vec1d = Wrap1D(np.arange(3))
        >>> np.concatenate((vec1d, vec1d))
        Wrap1D(x=array([0, 1, 2, 0, 1, 2]))
    """

    def __init__(self) -> None:
        self.__post_init__()  # initialize this way for `dataclasses.dataclass` subclasses.

    def __post_init__(self) -> None:
        # `_reg` is initialized here for `dataclasses.dataclass` subclasses.
        self._reg: dict[str, All_Dispatchers]
        object.__setattr__(self, "_reg", {})  # compatible with frozen dataclass
        # TODO  parametrization of Dispatcher.  Use All_Dispatchers

    # ===============================================================
    # Mapping

    def __getitem__(self, key: str | Callable[..., Any], /) -> All_Dispatchers:
        return self._reg[_get_key(key)]

    def __contains__(self, o: object, /) -> bool:
        return _get_key(o) in self._reg

    def __iter__(self) -> Iterator[str]:
        return iter(self._reg)

    def __len__(self) -> int:
        return len(self._reg)

    def keys(self) -> KeysView[str]:
        return self._reg.keys()

    def values(self) -> ValuesView[All_Dispatchers]:
        return self._reg.values()

    # ===============================================================

    @overload
    def implements(
        self,
        numpy_func: UFuncLike,
        /,
        dispatch_on: type,
        *,
        types: type | TypeConstraint | Collection[type | TypeConstraint] | None = None,
        methods: UFMsT = "__call__",
    ) -> ImplementsUFuncDecorator:
        ...

    @overload
    def implements(
        self,
        numpy_func: FunctionType,
        /,
        dispatch_on: type,
        *,
        types: type | TypeConstraint | Collection[type | TypeConstraint] | None = None,
        methods: UFMsT = "__call__",
    ) -> ImplementsFuncDecorator:
        ...

    def implements(
        self,
        numpy_func: UFuncLike | Callable[..., Any],
        /,
        dispatch_on: type,
        *,
        types: type | TypeConstraint | Collection[type | TypeConstraint] | None = None,
        methods: UFMsT = "__call__",
    ) -> ImplementsUFuncDecorator | ImplementsFuncDecorator:
        """
        Register an |array_function|_ implementation object.

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

        types : type or TypeConstraint or Collection thereof or None, keyword-only
            The types of the arguments of `numpy_func`. See
            |array_function|_. If `None` then ``dispatch_on`` must have
            class-level attribute ``NP_FUNC_TYPES`` specifying the types.
        methods :  {'__call__', 'accumulate', 'outer', 'reduce'} or None, keyword-only
            `numpy.ufunc` methods.

        Returns
        -------
        `~overload_numpy.overload.ImplementsFuncDecorator`
            Decorator to register the wrapped function.

        Examples
        --------
        There's a fair bit of setup required:

            >>> from dataclasses import dataclass, fields
            >>> from typing import ClassVar
            >>> import numpy as np
            >>> from overload_numpy import NumPyOverloader, NPArrayFuncOverloadMixin

            >>> W_FUNCS = NumPyOverloader()

            >>> @dataclass
            ... class Wrap1D(NPArrayFuncOverloadMixin):
            ...     '''A simple array wrapper.'''
            ...     x: np.ndarray
            ...     NP_OVERLOADS: ClassVar[NumPyOverloader] = W_FUNCS

            >>> w1d = Wrap1D(np.arange(3))

        Now we can register an ``implements`` functions.

            >>> @W_FUNCS.implements(np.concatenate, Wrap1D)  # overriding
            ... def concatenate(vecs):
            ...     VT = type(vecs[0])
            ...     return VT(*(np.concatenate(tuple(getattr(v, f.name) for v in vecs))
            ...                 for f in fields(VT)))

        Checking it works:

            >>> vec1d = Wrap1D(np.arange(3))

            >>> newvec = np.concatenate((vec1d, vec1d))
            >>> newvec
            Vector2D(x=array([0, 1, 2, 0, 1, 2]), y=array([3, 4, 5, 3, 4, 5]))

        ``implements`` also works for |ufunc|:

            >>> @W_FUNCS.implements(np.add, Wrap1D)
            ... def add(w1, w2):
            ...     return Wrap1D(np.add(w1.x, w2.x))

            >>> np.add(w1d, w1d)
            Wrap1D(x=array([0, 2, 4]))
        """
        if isinstance(numpy_func, UFuncLike):
            # LOCAL
            from overload_numpy.wrapper.ufunc import ImplementsUFuncDecorator

            if types is not None:
                raise ValueError(f"when implementing a ufunc, types must be None, not {types}")

            ms = _parse_methods(methods)
            return ImplementsUFuncDecorator(overloader=self, dispatch_on=dispatch_on, numpy_func=numpy_func, methods=ms)

        else:
            # LOCAL
            from overload_numpy.wrapper.func import ImplementsFuncDecorator

            return ImplementsFuncDecorator(overloader=self, dispatch_on=dispatch_on, numpy_func=numpy_func, types=types)

    # ---------------------------------------------------------------

    @overload
    def assists(
        self,
        numpy_funcs: UFuncLike,
        /,
        dispatch_on: type,
        *,
        types: type | TypeConstraint | Collection[type | TypeConstraint] | None,
        methods: UFMsT,
    ) -> AssistsUFuncDecorator:
        ...

    @overload
    def assists(
        self,
        numpy_funcs: FunctionType,
        /,
        dispatch_on: type,
        *,
        types: type | TypeConstraint | Collection[type | TypeConstraint] | None,
        methods: UFMsT,
    ) -> AssistsFuncDecorator:
        ...

    @overload
    def assists(
        self,
        numpy_funcs: set[Callable[..., Any] | UFuncLike],
        /,
        dispatch_on: type,
        *,
        types: type | TypeConstraint | Collection[type | TypeConstraint] | None = None,
        methods: UFMsT = "__call__",
    ) -> AssistsManyDecorator:
        ...

    def assists(
        self,
        numpy_funcs: UFuncLike | Callable[..., Any] | set[Callable[..., Any] | UFuncLike],
        /,
        dispatch_on: type,
        *,
        types: type | TypeConstraint | Collection[type | TypeConstraint] | None = None,
        methods: UFMsT = "__call__",
    ) -> AssistsUFuncDecorator | AssistsFuncDecorator | AssistsManyDecorator:
        """
        Register an |array_function|_ assistance function.

        This is a decorator factory, returning ``decorator``, which registers
        the decorated function as an overload method for :mod:`numpy` function
        ``numpy_func`` for a class of type ``dispatch_on``.

        Parameters
        ----------
        numpy_funcs : callable[..., Any], positional-only
            The :mod:`numpy` function that is being overloaded.
        dispatch_on : type
            The class type for which the overload implementation is being
            registered.
        types : type or TypeConstraint or Collection thereof or None, keyword-only
            The types of the arguments of ``numpy_func``.
            See |array_function|_ for details.
            Only used if a function (not |ufunc|) is being overridden.
            If `None` then ``dispatch_on`` must have class-level attribute
            ``NP_FUNC_TYPES`` specifying the types.
        methods : {'__call__', 'at', 'accumulate', 'outer', 'reduce', 'reduceat'} or set thereof, keyword-only
            The |ufunc| methods for which this override applies.
            Default is just "__call__".
            Only used if a |ufunc| (not function) is being overridden.

        Returns
        -------
        `~overload_numpy.overload.AssistsManyDecorator`
            Decorator to register the wrapped function(s). This decorator should
            never be called by the user.

        Examples
        --------
        There's a fair bit of setup required:

            >>> from dataclasses import dataclass, fields
            >>> from typing import ClassVar
            >>> import numpy as np
            >>> from overload_numpy import NumPyOverloader, NPArrayFuncOverloadMixin

            >>> W_FUNCS = NumPyOverloader()

            >>> @dataclass
            ... class Wrap1D(NPArrayFuncOverloadMixin):
            ...     '''A simple array wrapper.'''
            ...     x: np.ndarray
            ...     NP_OVERLOADS: ClassVar[NumPyOverloader] = W_FUNCS

            >>> w1d = Wrap1D(np.arange(3))

        Now we can register ``assists`` functions.

            >>> stack_funcs = {np.vstack, np.hstack, np.dstack, np.column_stack, np.row_stack}
            >>> @W_FUNCS.assists(stack_funcs, types=Wrap1D, dispatch_on=Wrap1D)
            ... def stack_assists(dispatch_on, func, vecs, *args, **kwargs):
            ...     cls = type(vecs[0])
            ...     return cls(*(func(tuple(getattr(v, f.name) for v in vecs), *args, **kwargs)
            ...                     for f in fields(cls)))

        Checking this works:

            >>> np.vstack((w1d, w1d))
            Wrap1D(x=array([[0, 1, 2],
                            [0, 1, 2]]))

            >>> np.hstack((w1d, w1d))
            Wrap1D(x=array([0, 1, 2, 0, 1, 2]))

        ``assists`` also works for |ufunc|:

            >>> add_funcs = {np.add, np.subtract}
            >>> @W_FUNCS.assists(add_funcs, types=Wrap1D, dispatch_on=Wrap1D)
            ... def add_assists(cls, func, w1, w2, *args, **kwargs):
            ...     return cls(*(func(getattr(w1, f.name), getattr(w2, f.name), *args, **kwargs)
            ...                  for f in fields(cls)))

        Checking this works:

            >>> np.subtract(w1d, w1d)
            Wrap1D(x=array([0, 0, 0]))

        We can also to implement the |ufunc| methods, like``accumulate``, for
        all the ``add_funcs`` overloads:

            >>> @add_assists.register("accumulate")
            ... def add_accumulate_assists(cls, func, w1, *args, **kwargs):
            ...     return cls(*(func(getattr(w1, f.name), *args, **kwargs)
            ...                  for f in fields(cls)))

            >>> np.subtract.accumulate(w1d)
            Wrap1D(x=array([ 0, -1, -3]))
        """
        if isinstance(numpy_funcs, UFuncLike):
            # LOCAL
            from overload_numpy.wrapper.ufunc import AssistsUFuncDecorator

            # `types` is ignored for ufuncs
            ms = _parse_methods(methods)
            return AssistsUFuncDecorator(overloader=self, dispatch_on=dispatch_on, numpy_func=numpy_funcs, methods=ms)

        elif callable(numpy_funcs):
            # LOCAL
            from overload_numpy.wrapper.func import AssistsFuncDecorator

            # `methods` is ignored for funcs
            return AssistsFuncDecorator(overloader=self, numpy_func=numpy_funcs, types=types, dispatch_on=dispatch_on)

        else:
            # LOCAL
            from overload_numpy.wrapper.func import AssistsFuncDecorator
            from overload_numpy.wrapper.many import AssistsManyDecorator
            from overload_numpy.wrapper.ufunc import AssistsUFuncDecorator

            ms = _parse_methods(methods)
            return AssistsManyDecorator(
                tuple(
                    (
                        AssistsUFuncDecorator(overloader=self, dispatch_on=dispatch_on, numpy_func=npf, methods=ms)
                        if isinstance(npf, UFuncLike)
                        else AssistsFuncDecorator(overloader=self, dispatch_on=dispatch_on, numpy_func=npf, types=types)
                    )
                    for npf in numpy_funcs
                )
            )
