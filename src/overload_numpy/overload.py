##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from abc import ABCMeta
from collections.abc import Collection
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Final,
    Iterator,
    KeysView,
    Mapping,
    TypeVar,
    ValuesView,
)

# LOCAL
from overload_numpy.assists import _Assists
from overload_numpy.constraints import Covariant, TypeConstraint
from overload_numpy.dispatch import _Dispatcher
from overload_numpy.npinfo import _NumPyFuncOverloadInfo

__all__: list[str] = []

##############################################################################
# TYPING

C = TypeVar("C", bound=Callable[..., Any])
R = TypeVar("R")


##############################################################################
# CODE
##############################################################################


class NumPyOverloader(Mapping[Callable[..., Any], _Dispatcher]):
    """Overload :mod:`numpy` functions with |__array_function__|.

    Parameters
    ----------
    _reg : dict[Callable, |`~overload_numpy.dispatch._Dispatcher`|], optional
        Registry of overloaded functions. You probably don't want to pass this
        as a parameter.

    Examples
    --------
    First, some imports:

        >>> from __future__ import annotations
        >>> from dataclasses import dataclass, fields
        >>> from typing import ClassVar
        >>> import numpy as np
        >>> from overload_numpy import NumPyOverloader, NDFunctionMixin

    Now we can define a |NumPyOverloader| instance:

        >>> VEC_FUNCS = NumPyOverloader()

    The overloads apply to an array wrapping class. Let's define one:

        >>> @dataclass
        ... class Vector1D(NDFunctionMixin):
        ...     '''A simple array wrapper.'''
        ...     x: np.ndarray
        ...     NP_FUNC_OVERLOADS: ClassVar[NumPyOverloader] = VEC_FUNCS

    Now ``numpy`` functions can be overloaded and registered for ``Vector1D``.

        >>> @VEC_FUNCS.implements(np.concatenate, Vector1D)
        ... def concatenate(vecs):
        ...     VT = type(vecs[0])
        ...     return VT(*(np.concatenate(tuple(getattr(v, f.name) for v in vecs))
        ...                 for f in fields(VT)))

    Time to check this works:

        >>> vec1d = Vector1D(np.arange(3))
        >>> np.concatenate((vec1d, vec1d))
        Vector1D(x=array([0, 1, 2, 0, 1, 2]))
    """

    _reg: Final[dict[Callable[..., Any], _Dispatcher]] = {}
    """Registry of overloaded functions.

    You probably don't want to pass this as a parameter.
    """

    # ===============================================================
    # Mapping

    def __getitem__(self, key: Callable[..., Any], /) -> _Dispatcher:
        return self._reg[key]

    def __contains__(self, o: object, /) -> bool:
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
            The types for argument ``types`` in |__array_function__|. If
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
            raise ValueError(f"types must be a {self.implements.__annotations__['types']}")

        return parsed

    def implements(
        self,
        numpy_func: Callable[..., Any],
        /,
        dispatch_on: type,
        *,
        types: type | TypeConstraint | Collection[type | TypeConstraint] | None = None,
    ) -> _ImplementsDecorator:
        """Register an |__array_function__| implementation object.

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
            |__array_function__|
            If `None` then ``dispatch_on`` must have class-level attribute
            ``NP_FUNC_TYPES`` specifying the types.

        Returns
        -------
        `~overload_numpy.overload._ImplementsDecorator`
            Decorator to register the wrapped function.

        Examples
        --------
        There's a fair bit of setup required:

            >>> from dataclasses import dataclass, fields
            >>> from typing import ClassVar
            >>> import numpy as np
            >>> from overload_numpy import NumPyOverloader, NDFunctionMixin

            >>> VEC_FUNCS = NumPyOverloader()

            >>> @dataclass
            ... class Vector1D(NDFunctionMixin):
            ...     '''A simple array wrapper.'''
            ...     x: np.ndarray
            ...     NP_FUNC_OVERLOADS: ClassVar[NumPyOverloader] = VEC_FUNCS

        Now we can register an ``implements`` functions.

            >>> @VEC_FUNCS.implements(np.concatenate, Vector1D)  # overriding
            ... def concatenate(vecs):
            ...     VT = type(vecs[0])
            ...     return VT(*(np.concatenate(tuple(getattr(v, f.name) for v in vecs))
            ...                 for f in fields(VT)))

        Checking it works:

            >>> vec1d = Vector1D(np.arange(3))

            >>> newvec = np.concatenate((vec1d, vec1d))
            >>> newvec
            Vector2D(x=array([0, 1, 2, 0, 1, 2]), y=array([3, 4, 5, 3, 4, 5]))
        """
        return _ImplementsDecorator(self, numpy_func=numpy_func, types=types, dispatch_on=dispatch_on)

    def assists(
        self,
        numpy_funcs: Callable[..., Any] | set[Callable[..., Any]],
        /,
        dispatch_on: type,
        *,
        types: type | TypeConstraint | Collection[type | TypeConstraint] | None = None,
    ) -> _AssistsManyDecorator:
        """Register an |__array_function__| assistance function.

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
            |__array_function__|
            If `None` then ``dispatch_on`` must have class-level attribute
            ``NP_FUNC_TYPES`` specifying the types.

        Returns
        -------
        `~overload_numpy.overload._AssistsManyDecorator`
            Decorator to register the wrapped function(s).
            This decorator should never be called by the user.

        Examples
        --------
        There's a fair bit of setup required:

            >>> from dataclasses import dataclass, fields
            >>> from typing import ClassVar
            >>> import numpy as np
            >>> from overload_numpy import NumPyOverloader, NDFunctionMixin

            >>> VEC_FUNCS = NumPyOverloader()

            >>> @dataclass
            ... class Vector1D(NDFunctionMixin):
            ...     '''A simple array wrapper.'''
            ...     x: np.ndarray
            ...     NP_FUNC_OVERLOADS: ClassVar[NumPyOverloader] = VEC_FUNCS

        Now we can register ``assists`` functions.

            >>> stack_funcs = {np.vstack, np.hstack, np.dstack, np.column_stack, np.row_stack}
            >>> @VEC_FUNCS.assists(stack_funcs, types=Vector1D, dispatch_on=Vector1D)
            ... def stack_assists(dispatch_on, func, vecs, *args, **kwargs):
            ...     cls = type(vecs[0])
            ...     return cls(*(func(tuple(getattr(v, f.name) for v in vecs), *args, **kwargs)
            ...                     for f in fields(cls)))

        Checking it worked:

            >>> assert np.vstack in VEC_FUNCS
            True
        """
        # Make ``numpy_funcs`` into a set[``numpy_func``]
        setnpfs = numpy_funcs if isinstance(numpy_funcs, set) else {numpy_funcs}

        # Return
        return _AssistsManyDecorator(
            _AssistsDecorator(self, numpy_func=npf, types=types, dispatch_on=dispatch_on) for npf in setnpfs
        )


##############################################################################
# Decorators for implementations


@dataclass(frozen=True)
class _OverloadDecoratorBase(metaclass=ABCMeta):
    """Decorator base class for registering an |__array_function__| overload.

    Instances of this class should not be used directly.

    Parameters
    ----------
    overloader : `~overload_numpy.overload.NumPyOverloader`
        Overloader instance.
    numpy_func : Callable[..., Any]
        The :mod:`numpy` function that is being overloaded.
    types : type or TypeConstraint or Collection thereof or None
        The types of the arguments of `numpy_func`. See |__array_function__| If
        `None` then ``dispatch_on`` must have class-level attribute
        ``NP_FUNC_TYPES`` specifying the types.
    dispatch_on : type
            The class type for which the overload implementation is being
            registered.
    """

    overloader: NumPyOverloader
    numpy_func: Callable[..., Any]
    types: type | TypeConstraint | Collection[type | TypeConstraint] | None
    dispatch_on: type

    def __post_init__(self) -> None:
        # Make single-dispatcher for numpy function
        if self.numpy_func not in self.overloader._reg:
            self.overloader._reg[self.numpy_func] = _Dispatcher()

        # Turn ``types`` into only TypeConstraint
        self.overloader._parse_types(self.types, self.dispatch_on)

    # @abstractmethod  # TODO: fix when https://github.com/python/mypy/issues/5374 released
    def func_hook(self, func: Callable[..., Any], /) -> Callable[..., Any]:
        """Function hook.

        Parameters
        ----------
        func : Callable[..., Any], positional-only
            The overloading function.

        Returns
        -------
        Callable[..., Any]
            How the function is overloaded. In ``_ImplementsDecorator`` this just
            returns ``func``. In ``_AssistsDecorator`` this creates an
            `~overload_numpy.assists._Assists`.

        Raises
        ------
        NotImplementedError
            This function must be overwritten in child classes.
        """
        raise NotImplementedError

    def __call__(self, func: C, /) -> C:
        """Register an |__array_function__| overload.

        Parameters
        ----------
        func : Callable[..., Any]
            The :mod:`numpy` function to overload.

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
            attribute ``NP_FUNC_TYPES``.
            If ``dispatch_type.NP_FUNC_TYPES`` is not a `frozenset` of `type` or
            `overload_numpy.constraints.TypeConstraint
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

        # Turn ``types`` into only TypeConstraint
        tinfo = self.overloader._parse_types(self.types, self.dispatch_on)
        # Note: I don't think types cannot be inferred from the function
        # because NumPy uses a dispatcher object to get the
        # ``relevant_args``.

        # Adding a new numpy function
        info = _NumPyFuncOverloadInfo(
            func=self.func_hook(func), types=tinfo, implements=self.numpy_func, dispatch_on=self.dispatch_on
        )
        # Register the function
        self.overloader._reg[self.numpy_func]._dispatcher.register(self.dispatch_on, info)
        return func


@dataclass(frozen=True)
class _ImplementsDecorator(_OverloadDecoratorBase):
    """Decorator for registering with `~overload_numpy.NumPyOverloader`.

    Instances of this class should not be used directly.

    Parameters
    ----------
    overloader : `~overload_numpy.overload.NumPyOverloader`
        Overloader instance.
    numpy_func : Callable[..., Any]
        The :mod:`numpy` function that is being overloaded.
    types : type or TypeConstraint or Collection thereof or None
        The types of the arguments of `numpy_func`. See |__array_function__| If
        `None` then ``dispatch_on`` must have class-level attribute
        ``NP_FUNC_TYPES`` specifying the types.
    dispatch_on : type
            The class type for which the overload implementation is being
            registered.

    Methods
    -------
    func_hook
        Returns the input ``func`` unchanged.
    """

    def func_hook(self, func: C, /) -> C:
        return func


@dataclass(frozen=True)
class _AssistsDecorator(_OverloadDecoratorBase):
    """Decorator for registering with `~overload_numpy.NumPyOverloader`.

    Instances of this class should not be used directly.

    Parameters
    ----------
    overloader : `~overload_numpy.overload.NumPyOverloader`
        Overloader instance.
    numpy_func : Callable[..., Any]
        The :mod:`numpy` function that is being overloaded.
    types : type or TypeConstraint or Collection thereof or None
        The types of the arguments of `numpy_func`. See |__array_function__| If
        `None` then ``dispatch_on`` must have class-level attribute
        ``NP_FUNC_TYPES`` specifying the types.
    dispatch_on : type
            The class type for which the overload implementation is being
            registered.

    Methods
    -------
    func_hook
        Returns the input ``func`` as an `~overload_numpy.assists._Assists` object.
    """

    def func_hook(self, func: Callable[..., R], /) -> _Assists[R]:
        return _Assists(func=func, implements=self.numpy_func, dispatch_on=self.dispatch_on)


@dataclass(frozen=True)
class _AssistsManyDecorator:
    """Decorator for registering many `~overload_numpy.NumPyOverloader.assists` functions.

    Parameters
    ----------
    _iterator : Iterator[_AssistsDecorator]
        Iterator of ``_AssistsDecorator``.
    """

    _iterator: Iterator[_AssistsDecorator]
    """Iterator of ``_AssistsDecorator``."""

    def __call__(self, func: C, /) -> C:
        """Decorate and return ``func``.

        Parameters
        ----------
        func : Callable[..., R]
            Assistance function.

        Returns
        -------
        func : Callable[..., R]
            Same as ``func``.
        """
        # Iterate through ``self._iterator``, calling the contained
        # ``_AssistsDecorator``.
        for obj in self._iterator:
            obj(func)

        return func
