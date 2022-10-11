##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    KeysView,
    Mapping,
    TypeVar,
    final,
)

# LOCAL
from overload_numpy.utils import (
    UFMT,
    UFMsT,
    UFuncMethodOverloadMap,
    _get_key,
    _parse_methods,
)
from overload_numpy.wrapper.dispatch import Dispatcher

if TYPE_CHECKING:
    # LOCAL
    from overload_numpy._typeutils import UFuncLike
    from overload_numpy.overload import NumPyOverloader

__all__: list[str] = []


##############################################################################
# TYPING

C = TypeVar("C", bound="Callable[..., Any]")
UT = TypeVar("UT", "ImplementsUFunc", "AssistsUFunc")


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class OverrideUFuncBase:
    """Base class to overloading a |func|.

    Parameters
    ----------
    funcs : dict[str, Callable], keyword-only
        The overloading function for each |ufunc| method.
    implements: |ufunc|, keyword-only
        The overloaded |ufunc|.
    dispatch_on : type, keyword-only
        The type dispatched on. See
        `~overload_numpy.wrapper.dispatch.Dispatcher`.
    """

    _funcs: UFuncMethodOverloadMap
    """The overloading function for each :class:`numpy.ufunc` method."""

    implements: UFuncLike
    """The overloaded |ufunc|."""

    dispatch_on: type
    """The type dispatched on. See
    `~overload_numpy.wrapper.dispatch.Dispatcher`."""

    @property
    def funcs(self) -> MappingProxyType[str, object]:
        return MappingProxyType(self._funcs)

    @property
    def __wrapped__(self) -> Callable[..., Any]:
        return self._funcs["__call__"]

    @property
    def methods(self) -> KeysView[str]:
        """All the |ufunc| methods."""
        return self._funcs.keys()

    def validate_method(self, method: UFMT, /) -> bool:
        """Validates that the method has an overload.

        Parameters
        ----------
        method : {'__call__', 'at', 'accumulate', 'outer', 'reduce'}
            Name of a method on a |ufunc|.

        Returns
        -------
        bool
            If the ``method`` is in the attr ``funcs``.
        """
        return method in self._funcs

    def register(self, methods: UFMsT, /) -> RegisterUFuncMethodDecorator:
        """Register overload for |ufunc| methods.

        Parameters
        ----------
        methods : {'__call__', 'at', 'accumulate', 'outer', 'reduce',
        'reduceat'}
            The names of the methods to overload.

        Returns
        -------
        decorator : `RegisterUFuncMethodDecorator`
            Decorator to register a function as an overload for a |ufunc|
            method.

        Examples
        --------
        We are going to work towards overriding the ``accumulate`` method of

        .. code-block:: python

            @OverrideUFuncBase.register("accumulate") def add_accumulate(w1,
            indices, w2):
                np.add.at(w1.x, indices, w2.x)

        >>> from dataclasses import dataclass
        >>> from typing import ClassVar
        >>> import numpy as np
        >>> from overload_numpy import NumPyOverloader, NPArrayOverloadMixin

        >>> NP_OVERLOADS = NumPyOverloader()

        >>> @dataclass
        ... class Wrap1(NPArrayOverloadMixin):
        ...     x: np.ndarray
        ...     NP_OVERLOADS: ClassVar[NumPyOverloader] = NP_OVERLOADS

        >>> @NP_OVERLOADS.implements(np.add, Wrap1)
        ... def add(w1, w2, *args, **kwargs):
        ...     return Wrap1(np.add(w1.x, w2.x, *args, **kwargs))

        >>> @add.register("at")
        ... def add_at(w1, indices, w2):
        ...     np.add.at(w1.x, indices, w2.x)

        Notes
        -----
        This is a decorator factory.
        """
        # TODO! validation that the function has the right signature.
        return RegisterUFuncMethodDecorator(self._funcs, _parse_methods(methods))


@final
@dataclass(frozen=True)
class RegisterUFuncMethodDecorator:
    """Decorator to register a |ufunc| method implementation.

    Returned by `~overload_numpy.wrapper.ufunc.OverrideUfuncBase.register`.

    .. warning::

        Only the ``__call__`` method is public API. Instances of this class are
        created as-needed by |NumPyOverloader| if a |ufunc| override is
        registered. Users should not make an instance of this class.

    Methods
    -------
    __call__
    """

    _funcs_dict: UFuncMethodOverloadMap
    """`dict` of the |ufunc| method overloads.

    This is linked to the map on a `OverrideUFuncBase` instance.
    """

    _applicable_methods: frozenset[UFMT]
    """|ufunc| methods for which this decorator will register overrides."""

    def __call__(self, func: C, /) -> C:
        """Decorator to register an overload funcction for |ufunc| methods.

        Parameters
        ----------
        func : Callable[..., Any]
            The overload function for specified |ufunc| methods.

        Returns
        -------
        func : Callable[..., Any]
            Unchanged.
        """
        # Iterate through the methods, adding as overloads for specified
        # methods.
        for m in self._applicable_methods:
            self._funcs_dict[m] = func
        return func


@dataclass(frozen=True)
class OverloadUFuncDecoratorBase(Generic[UT]):
    """Base class for |ufunc| overload decorator.

    Parameters
    ----------
    dispatch_on : type, keyword-only
        The class type for which the overload implementation is being
        registered.
    numpy_func : |ufunc|, keyword-only
        The :mod:`numpy` |ufunc| that is being overloaded.
    methods : set[{'__call__', 'at', 'accumulate', 'outer', 'reduce'}], keyword-only
        Set of names of |ufunc| methods.
    overloader : |NumPyOverloader|, keyword-only
        Overloader instance.

    Attributes
    ----------
    OverrideCls : type
        Class attribute.
    """

    OverrideCls: ClassVar[type]

    dispatch_on: type
    """The type dispatched on.

    See :class:`~overload_numpy.wrapper.dispatch.Dispatcher`.
    """

    numpy_func: UFuncLike
    """The :mod:`numpy` function that is being overloaded."""

    methods: frozenset[UFMT]
    """Set of names of |ufunc| methods."""

    overloader: NumPyOverloader
    """|NumPyOverloader| instance."""

    def __post_init__(self) -> None:
        # Make single-dispatcher for numpy function
        key = _get_key(self.numpy_func)
        if key not in self.overloader._reg:
            self.overloader._reg[key] = Dispatcher[UT]()

    def __call__(self, func: Callable[..., Any], /) -> UT:
        """Register overload on Dispatcher.

        Parameters
        ----------
        func : Callable[..., Any], positional-only
            Overloading function for specified ``methods``, ``numpy_func``, and
            ``dispatch_on``.

        Returns
        -------
        ``UT``
            `overload_numpy.wrapper.ufunc.ImplementsUFunc` or
            `overload_numpy.wrapper.ufunc.AssistsUFunc`.
        """
        # Adding a new numpy function
        info: UT = self.OverrideCls(
            _funcs={m: func for m in self.methods},  # includes __call__
            implements=self.numpy_func,
            dispatch_on=self.dispatch_on,
        )
        # Register the function. The ``info`` is wrapped by ``DispatchWrapper``
        # so `~functools.singledispatch` returns the ``info``.
        # self.dispatcher.register(self.dispatch_on, info)
        self.overloader[self.numpy_func].register(self.dispatch_on, info)
        return info


##############################################################################

# ============================================================================
# Implements


@dataclass(frozen=True)
class ImplementsUFunc(OverrideUFuncBase):
    """Implements a |ufunc| override.

    .. warning::

        Only the ``register`` method is currently public API. Instances of this
        class are created as-needed by |NumPyOverloader| whenever a |ufunc|
        override is made with `~overload_numpy.NumPyOverloader.implements`.

    Parameters
    ----------
    funcs : dict[str, Callable], keyword-only
        The overloading function for each |ufunc| method, including
        ``__call__``.
    implements: |ufunc|, keyword-only
        The overloaded |ufunc|.
    dispatch_on : type, keyword-only
        The type dispatched on. See
        `~overload_numpy.wrapper.dispatch.Dispatcher`.

    Methods
    -------
    register
    """

    def __call__(
        self, method: UFMT, _: type, /, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> object:  # TODO: parametrize return type?
        """Evaluate a |ufunc| method with given arguments.

        Parameters
        ----------
        method : str, positional-only
            The |ufunc| method to evaluate. One of {'__call__', 'at',
            'accumulate', 'outer', 'reduce'}. This is validated with
            `~overload_numpy.wrapper.ufunc.validate_method`.
        calling_type : type, positional-only
            The calling type of the class for which this is the override.
        args : tuple[Any, ...]
            Tuple of arguments to pass to the override.
        kwargs : Mapping[str, Any]
            Mapping of keyword arguments to pass to the override.

        Returns
        -------
        object
            The result of evaluating the |ufunc| method.
        """
        if not self.validate_method(method):
            return NotImplemented
        return self._funcs[method](*args, **kwargs)


@dataclass(frozen=True)
class ImplementsUFuncDecorator(OverloadUFuncDecoratorBase[ImplementsUFunc]):
    """Decorator for registering with |NumPyOverloader|.

    .. warning::

        Only the ``__call__`` methods is currently public API. Instances of this
        class are created as-needed by |NumPyOverloader| whenever a |ufunc|
        override is made with `~overload_numpy.NumPyOverloader.implements`.

    Parameters
    ----------
    numpy_func : Callable[..., Any]
        The :mod:`numpy` function that is being overloaded.
    dispatch_on : type
            The class type for which the overload implementation is being
            registered.
    overloader : |NumPyOverloader|
        Overloader instance.

    """

    OverrideCls: ClassVar[type[ImplementsUFunc]] = ImplementsUFunc


# ============================================================================
# Assists


@dataclass(frozen=True)
class AssistsUFunc(OverrideUFuncBase):
    """Assists a |ufunc| override.

    .. warning::

        Only the ``register`` methods is currently public API. Instances of this
        class are created as-needed by |NumPyOverloader| whenever a |ufunc|
        override is made with `~overload_numpy.NumPyOverloader.assists`.

    Parameters
    ----------
    funcs : dict[str, Callable], keyword-only
        The overloading function for each |ufunc| method.
    implements: |ufunc|, keyword-only
        The overloaded |ufunc|.
    dispatch_on : type, keyword-only
        The type dispatched on. See
        `~overload_numpy.wrapper.dispatch.Dispatcher`.
    """

    # TODO: parametrize return type?
    def __call__(self, method: UFMT, calling_type: type, /, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> object:
        """Evaluate a |ufunc| method with given arguments.

        Parameters
        ----------
        method : str, positional-only
            The |ufunc| method to evaluate. One of {'__call__', 'at',
            'accumulate', 'outer', 'reduce'}. This is validated with
            `~overload_numpy.wrapper.ufunc.validate_method`.
        calling_type : type, positional-only
            The calling type of the class for which this is the override.
        args : tuple[Any, ...]
            Tuple of arguments to pass to the override.
        kwargs : Mapping[str, Any]
            Mapping of keyword arguments to pass to the override.

        Returns
        -------
        object
            The result of evaluating the |ufunc| method.
        """
        if not self.validate_method(method):
            return NotImplemented
        return self._funcs[method](calling_type, getattr(self.implements, method), *args, **kwargs)


@dataclass(frozen=True)
class AssistsUFuncDecorator(OverloadUFuncDecoratorBase[AssistsUFunc]):
    """Decorator to register a |ufunc| override assist with |NumPyOverloader|.

    .. warning::

        Only the ``__call__`` methods is currently public API. Instances of this
        class are created as-needed by |NumPyOverloader| whenever a |ufunc|
        override is made with `~overload_numpy.NumPyOverloader.assists`.

    Parameters
    ----------
    overloader : |NumPyOverloader|, keyword-only
        Overloader instance.
    numpy_func : Callable[..., Any], keyword-only
        The :mod:`numpy` function that is being overloaded.
    methods : set[str], keyword-only
        Set of names of |ufunc| methods -- {'__call__', 'at', 'accumulate',
        'outer', 'reduce'}.
    dispatch_on : type, keyword-only
        The class type for which the overload implementation is being
        registered.
    """

    OverrideCls: ClassVar[type[AssistsUFunc]] = AssistsUFunc