r"""Mixins for adding |array_ufunc|_ &/or |array_function|_ methods.

Examples
--------
First, some imports:

    >>> from dataclasses import dataclass, fields
    >>> from typing import ClassVar
    >>> import numpy as np
    >>> from overload_numpy import NumPyOverloader, NPArrayOverloadMixin

Now we can define a |NumPyOverloader| instance:

    >>> W_FUNCS = NumPyOverloader()

The overloads apply to an array wrapping class. Let's define one:

    >>> @dataclass
    ... class Wrap1D(NPArrayOverloadMixin):
    ...     '''A simple array wrapper.'''
    ...     x: np.ndarray
    ...     NP_OVERLOADS: ClassVar[NumPyOverloader] = W_FUNCS

    >>> w1d = Wrap1D(np.arange(3))

Implementing an Overload
^^^^^^^^^^^^^^^^^^^^^^^^
Now both :class:`numpy.ufunc` (e.g. :obj:`numpy.add`) and :mod:`numpy` functions
(e.g. :func:`numpy.concatenate`) can be overloaded and registered for
``Wrap1D``.

    >>> @W_FUNCS.implements(np.add, Wrap1D)
    ... def add(w1, w2):
    ...     return Wrap1D(np.add(w1.x, w2.x))

    >>> @W_FUNCS.implements(np.concatenate, Wrap1D)
    ... def concatenate(w1ds):
    ...     return Wrap1D(np.concatenate(tuple(w.x for w in w1ds)))

Time to check these work:

    >>> np.add(w1d, w1d)
    Wrap1D(x=array([0, 2, 4]))

    >>> np.concatenate((w1d, w1d))
    Wrap1D(x=array([0, 1, 2, 0, 1, 2]))

|ufunc| also have a number of methods: 'at', 'accumulate', 'outer', etc. The
function dispatch mechanism in `NEP13
<https://numpy.org/neps/nep-0013-ufunc-overrides.html>`_ says that  "If one of
the input or output arguments implements __array_ufunc__, it is executed instead
of the ufunc." Currently the overloaded `numpy.add` does not work for any of the
|ufunc| methods.

    >>> try: np.add.accumulate(w1d)
    ... except Exception: print("failed")
    failed

|ufunc| method overloads can be registered on the wrapped ``add``
implementation:

    >>> @add.register('accumulate')
    ... def add_accumulate(w1):
    ...     return Wrap1D(np.add.accumulate(w1.x))

    >>> np.add.accumulate(w1d)
    Wrap1D(x=array([0, 1, 3]))

Dispatching Overloads for Subclasses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
What if we defined a subclass of ``Wrap1D``?

    >>> @dataclass
    ... class Wrap2D(Wrap1D):
    ...     '''A simple 2-array wrapper.'''
    ...     y: np.ndarray

The overload for :func:`numpy.concatenate` registered on ``Wrap1D`` will not
work correctly for ``Wrap2D``. However, |NumPyOverloader| supports
single-dispatch on the calling type for the overload, so overloads can be
customized for subclasses.

    >>> @W_FUNCS.implements(np.add, Wrap2D)
    ... def add(w1, w2):
    ...     print("using Wrap2D implementation...")
    ...     return Wrap2D(np.add(w1.x, w2.x),
    ...                   np.add(w1.y, w2.y))

    >>> @W_FUNCS.implements(np.concatenate, Wrap2D)
    ... def concatenate2(w2ds):
    ...     print("using Wrap2D implementation...")
    ...     return Wrap2D(np.concatenate(tuple(w.x for w in w2ds)),
    ...                   np.concatenate(tuple(w.y for w in w2ds)))

Checking these work:

    >>> w2d = Wrap2D(np.arange(3), np.arange(3, 6))
    >>> np.add(w2d, w2d)
    using Wrap2D implementation...
    Wrap2D(x=array([0, 2, 4]), y=array([ 6, 8, 10]))

    >>> np.concatenate((w2d, w2d))
    using Wrap2D implementation...
    Wrap2D(x=array([0, 1, 2, 0, 1, 2]), y=array([3, 4, 5, 3, 4, 5]))

Great! But rather than defining a new implementation for each subclass, let's
see how we could write a more broadly applicable overload:

    >>> @W_FUNCS.implements(np.add, Wrap1D)  # overriding both
    ... @W_FUNCS.implements(np.add, Wrap2D)  # overriding both
    ... def add_general(w1, w2):
    ...     WT = type(w1)
    ...     return WT(*(np.add(getattr(w1, f.name), getattr(w2, f.name))
    ...                 for f in fields(WT)))

    >>> @W_FUNCS.implements(np.concatenate, Wrap1D)  # overriding both
    ... @W_FUNCS.implements(np.concatenate, Wrap2D)  # overriding both
    ... def concatenate_general(ws):
    ...     WT = type(ws[0])
    ...     return WT(*(np.concatenate(tuple(getattr(w, f.name) for w in ws))
    ...                 for f in fields(WT)))

Checking these work:

    >>> np.add(w2d, w2d)
    Wrap2D(x=array([0, 2, 4]), y=array([ 6, 8, 10]))

    >>> np.concatenate((w2d, w2d))
    Wrap2D(x=array([0, 1, 2, 0, 1, 2]), y=array([3, 4, 5, 3, 4, 5]))

    >>> @dataclass
    ... class Wrap3D(Wrap2D):
    ...     '''A simple 3-array wrapper.'''
    ...     z: np.ndarray

    >>> w3d = Wrap3D(np.arange(2), np.arange(3, 5), np.arange(6, 8))
    >>> np.add(w3d, w3d)
    Wrap3D(x=array([0, 2]), y=array([6, 8]), z=array([12, 14]))
    >>> np.concatenate((w3d, w3d))
    Wrap3D(x=array([0, 1, 0, 1]), y=array([3, 4, 3, 4]), z=array([6, 7, 6, 7]))

Assisting Groups of Overloads
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the previous examples we wrote implementations for a single NumPy function.
Overloading the full set of NumPy functions this way would take a long time.

Wouldn't it be better if we could write many fewer, based on groups of NumPy
functions?

    >>> add_funcs = {np.add, np.subtract}
    >>> @W_FUNCS.assists(add_funcs, types=Wrap1D, dispatch_on=Wrap1D)
    ... def add_assists(cls, func, w1, w2, *args, **kwargs):
    ...     return cls(*(func(getattr(w1, f.name), getattr(w2, f.name), *args, **kwargs)
    ...                     for f in fields(cls)))

    >>> stack_funcs = {np.vstack, np.hstack, np.dstack, np.column_stack, np.row_stack}
    >>> @W_FUNCS.assists(stack_funcs, types=Wrap1D, dispatch_on=Wrap1D)
    ... def stack_assists(cls, func, ws, *args, **kwargs):
    ...     return cls(*(func(tuple(getattr(v, f.name) for v in ws), *args, **kwargs)
    ...                     for f in fields(cls)))

Checking these work:

    >>> np.subtract(w2d, w2d)
    Wrap2D(x=array([0, 0, 0]), y=array([0, 0, 0]))

    >>> np.vstack((w1d, w1d))
    Wrap1D(x=array([[0, 1, 2],
                        [0, 1, 2]]))

    >>> np.hstack((w1d, w1d))
    Wrap1D(x=array([0, 1, 2, 0, 1, 2]))

We would also like to implement the ``accumulate`` method for all the
``add_funcs`` overloads:

    >>> @add_assists.register("accumulate")
    ... def add_accumulate_assists(cls, func, w1, *args, **kwargs):
    ...     return cls(*(func(getattr(w1, f.name), *args, **kwargs)
    ...                  for f in fields(cls)))

    >>> np.subtract.accumulate(w2d)
    Wrap2D(x=array([ 0, -1, -3]), y=array([ 3, -1, -6]))
"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Collection

# THIRDPARTY
from mypy_extensions import mypyc_attr, trait

# LOCAL
from overload_numpy.implementors.func import AssistsFunc, ImplementsFunc
from overload_numpy.implementors.ufunc import AssistsUFunc, ImplementsUFunc

if TYPE_CHECKING:
    # LOCAL
    from overload_numpy._typeutils import UFuncLike
    from overload_numpy.constraints import TypeConstraint
    from overload_numpy.overload import NumPyOverloader
    from overload_numpy.utils import UFMT


__all__: list[str] = []


##############################################################################
# CODE
##############################################################################


@mypyc_attr(allow_interpreted_subclasses=True)
@trait
class NPArrayFuncOverloadMixin:
    """Mixin for adding |array_function|_ to a class.

    This mixin adds the method ``__array_function__``. Subclasses must define a
    class variable ``NP_OVERLOADS`` and optionally ``NP_FUNC_TYPES``.

    Attributes
    ----------
    NP_OVERLOADS : |NumPyOverloader|
        How the overrides are registered.
        A class-attribute of an instance of |NumPyOverloader|.
    NP_FUNC_TYPES : frozenset[type | TypeConstraint] | None, optional
        Default type constraints.
        A class-attribute of `None` or a `frozenset` of `type` or
        |TypeConstraint|. If `None`, then the ``types`` argument for overloading
        functions becomes mandatory. If a `frozenset` (including blank) then the
        self-type + the contents are used in the types check.
        See |array_function|_ for details of the ``types`` argument.

    Notes
    -----
    When compiled this class is a ``mypyc`` :func:`~mypy_extensions.trait` and
    permits interpreted subclasses (see
    https://mypyc.readthedocs.io/en/latest/native_classes.html#inheritance).

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

    Now :mod:`numpy` functions can be overloaded and registered for ``Wrap1D``.

        >>> @W_FUNCS.implements(np.concatenate, Wrap1D)
        ... def concatenate(w1ds):
        ...     return Wrap1D(np.concatenate(tuple(w.x for w in w1ds)))

    Time to check this works:

        >>> w1d = Wrap1D(np.arange(3))
        >>> np.concatenate((w1d, w1d))
        Wrap1D(x=array([0, 1, 2, 0, 1, 2]))

    What if we defined a subclass of ``Wrap1D``?

        >>> @dataclass
        ... class Wrap2D(Wrap1D):
        ...     '''A simple 2-array wrapper.'''
        ...     y: np.ndarray

    The overload for :func:`numpy.concatenate` registered on ``Wrap1D`` will not
    work correctly for ``Wrap2D``. However, |NumPyOverloader| supports
    single-dispatch on the calling type for the overload, so overloads can be
    customized for subclasses.

        >>> @W_FUNCS.implements(np.concatenate, Wrap2D)
        ... def concatenate2(w2ds):
        ...     print("using Wrap2D implementation...")
        ...     return Wrap2D(np.concatenate(tuple(w.x for w in w2ds)),
        ...                   np.concatenate(tuple(w.y for w in w2ds)))

    Checking this works:

        >>> w2d = Wrap2D(np.arange(3), np.arange(3, 6))
        >>> np.concatenate((w2d, w2d))
        using Wrap2D implementation...
        Wrap2D(x=array([0, 1, 2, 0, 1, 2]), y=array([3, 4, 5, 3, 4, 5]))

    Great! But rather than defining a new implementation for each subclass,
    let's see how we could write a more broadly applicable overload:

        >>> @W_FUNCS.implements(np.concatenate, Wrap1D)  # overriding both
        ... @W_FUNCS.implements(np.concatenate, Wrap2D)  # overriding both
        ... def concatenate_general(ws):
        ...     WT = type(ws[0])
        ...     return WT(*(np.concatenate(tuple(getattr(w, f.name) for w in ws))
        ...                 for f in fields(WT)))

    Checking this works:

        >>> np.concatenate((w2d, w2d))
        Wrap2D(x=array([0, 1, 2, 0, 1, 2]), y=array([3, 4, 5, 3, 4, 5]))

        >>> @dataclass
        ... class Wrap3D(Wrap2D):
        ...     '''A simple 3-array wrapper.'''
        ...     z: np.ndarray

        >>> w3d = Wrap3D(np.arange(2), np.arange(3, 5), np.arange(6, 8))
        >>> np.concatenate((w3d, w3d))
        Wrap3D(x=array([0, 1, 0, 1]), y=array([3, 4, 3, 4]), z=array([6, 7, 6, 7]))

    In the previous examples we wrote implementations for a single NumPy
    function. Overloading the full set of NumPy functions this way would take a
    long time.

    Wouldn't it be better if we could write many fewer, based on groups of NumPy
    functions?

        >>> stack_funcs = {np.vstack, np.hstack, np.dstack, np.column_stack, np.row_stack}
        >>> @W_FUNCS.assists(stack_funcs, types=Wrap1D, dispatch_on=Wrap1D)
        ... def stack_assists(cls, func, ws, *args, **kwargs):
        ...     return cls(*(func(tuple(getattr(v, f.name) for v in ws), *args, **kwargs)
        ...                     for f in fields(cls)))

    Checking this works:

        >>> np.vstack((w1d, w1d))
        Wrap1D(x=array([[0, 1, 2],
                          [0, 1, 2]]))

        >>> np.hstack((w1d, w1d))
        Wrap1D(x=array([0, 1, 2, 0, 1, 2]))
    """

    NP_OVERLOADS: ClassVar[NumPyOverloader]

    NP_FUNC_TYPES: ClassVar[frozenset[type | TypeConstraint] | None] = frozenset()

    def __array_function__(
        self, func: Callable[..., Any], types: Collection[type], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """|array_function|_.

        Parameters
        ----------
        func : Callable[..., Any]
            The overloaded :mod:`numpy` function.
        types : Collection[type]
            ``types`` is a collection collections.abc.Collection of unique
            argument types from the original NumPy function call that implement
            |array_function|_.
        args : tuple[Any, ...]
            The tuple args are directly passed on from the original call.
        kwargs : dict[str, Any]
            The dict kwargs are directly passed on from the original call.

        Returns
        -------
        Any
            The result of calling the overloaded functions.
        """
        # Check if can be dispatched.
        if not self.NP_OVERLOADS.__contains__(func):
            return NotImplemented

        # Get ImplementsFunc on function, from single-dispatch on type
        # of self. If a NotImplementedError is raised, it means there's no
        # implementation.
        try:
            fwrap = self.NP_OVERLOADS[func](self)
        except NotImplementedError:
            return NotImplemented

        # Needed only for type narrowing
        if not isinstance(fwrap, (ImplementsFunc, AssistsFunc)):
            return NotImplemented

        # Note: this allows subclasses that don't override `__array_function__`
        # to handle this object type.
        #
        # From https://numpy.org/devdocs/reference/arrays.classes.html:
        # "Implementations should not rely on the iteration order of types."
        # TODO: move this inside the fwrap.__call__?
        if not fwrap.validate_types(types):
            return NotImplemented

        # TODO! validation for args and kwargs.
        return fwrap(type(self), args, kwargs)  # Returns result or NotImplemented


##############################################################################


@mypyc_attr(allow_interpreted_subclasses=True)
@trait
class NPArrayUFuncOverloadMixin:
    """
    Mixin for adding |array_ufunc|_ to a class.

    This mixin adds the method ``__array_ufunc__``. Subclasses must define a
    class variable ``NP_OVERLOADS``.

    Attributes
    ----------
    NP_OVERLOADS : |NumPyOverloader|
        A class-attribute of an instance of |NumPyOverloader|.

    Notes
    -----
    When compiled this class is a ``mypyc`` :func:`~mypy_extensions.trait` and
    permits interpreted subclasses (see
    https://mypyc.readthedocs.io/en/latest/native_classes.html#inheritance).

    Examples
    --------
    First, some imports:

        >>> from dataclasses import dataclass, fields
        >>> from typing import ClassVar
        >>> import numpy as np
        >>> from overload_numpy import NumPyOverloader, NPArrayUFuncOverloadMixin

    Now we can define a |NumPyOverloader| instance:

        >>> W_FUNCS = NumPyOverloader()

    The overloads apply to an array wrapping class. Let's define one:

        >>> @dataclass
        ... class Wrap1D(NPArrayUFuncOverloadMixin):
        ...     '''A simple array wrapper.'''
        ...     x: np.ndarray
        ...     NP_OVERLOADS: ClassVar[NumPyOverloader] = W_FUNCS

        >>> w1d = Wrap1D(np.arange(3))

    Now :class:`numpy.ufunc` can be overloaded and registered for ``Wrap1D``.

        >>> @W_FUNCS.implements(np.add, Wrap1D)
        ... def add(w1, w2):
        ...     return Wrap1D(np.add(w1.x, w2.x))

    Time to check this works:

        >>> np.add(w1d, w1d)
        Wrap1D(x=array([0, 2, 4]))

    |ufunc| also have a number of methods: 'at', 'accumulate', etc. The function
    dispatch mechanism in `NEP13
    <https://numpy.org/neps/nep-0013-ufunc-overrides.html>`_ says that  "If one
    of the input or output arguments implements __array_ufunc__, it is executed
    instead of the ufunc." Currently the overloaded `numpy.add` does not work
    for any of the |ufunc| methods.

        >>> try: np.add.accumulate(w1d)
        ... except Exception: print("failed")
        failed

    |ufunc| method overloads can be registered on the wrapped ``add``
    implementation:

        >>> @add.register('accumulate')
        ... def add_accumulate(w1):
        ...     return Wrap1D(np.add.accumulate(w1.x))

        >>> np.add.accumulate(w1d)
        Wrap1D(x=array([0, 1, 3]))

    What if we defined a subclass of ``Wrap1D``?

        >>> @dataclass
        ... class Wrap2D(Wrap1D):
        ...     '''A simple 2-array wrapper.'''
        ...     y: np.ndarray

    The overload for :func:`numpy.concatenate` registered on ``Wrap1D`` will not
    work correctly for ``Wrap2D``. However, |NumPyOverloader| supports
    single-dispatch on the calling type for the overload, so overloads can be
    customized for subclasses.

        >>> @W_FUNCS.implements(np.add, Wrap2D)
        ... def add(w1, w2):
        ...     print("using Wrap2D implementation...")
        ...     return Wrap2D(np.add(w1.x, w2.x), np.add(w1.y, w2.y))

    Checking this works:

        >>> w2d = Wrap2D(np.arange(3), np.arange(3, 6))
        >>> np.add(w2d, w2d)
        using Wrap2D implementation...
        Wrap2D(x=array([0, 2, 4]), y=array([ 6, 8, 10]))

    Great! But rather than defining a new implementation for each subclass,
    let's see how we could write a more broadly applicable overload:

        >>> @W_FUNCS.implements(np.add, Wrap1D)  # overriding both
        ... @W_FUNCS.implements(np.add, Wrap2D)  # overriding both
        ... def add_general(w1, w2):
        ...     WT = type(w1)
        ...     return WT(*(np.add(getattr(w1, f.name), getattr(w2, f.name))
        ...                 for f in fields(WT)))

    Checking this works:

        >>> np.add(w2d, w2d)
        Wrap2D(x=array([0, 2, 4]), y=array([ 6, 8, 10]))

        >>> @dataclass
        ... class Wrap3D(Wrap2D):
        ...     '''A simple 3-array wrapper.'''
        ...     z: np.ndarray

        >>> w3d = Wrap3D(np.arange(2), np.arange(3, 5), np.arange(6, 8))
        >>> np.add(w3d, w3d)
        Wrap3D(x=array([0, 2]), y=array([6, 8]), z=array([12, 14]))

    In the previous examples we wrote implementations for a single NumPy
    function. Overloading the full set of NumPy functions this way would take a
    long time.

    Wouldn't it be better if we could write many fewer, based on groups of NumPy
    functions?

        >>> add_funcs = {np.add, np.subtract}
        >>> @W_FUNCS.assists(add_funcs, types=Wrap1D, dispatch_on=Wrap1D)
        ... def add_assists(cls, func, w1, w2, *args, **kwargs):
        ...     return cls(*(func(getattr(w1, f.name), getattr(w2, f.name), *args, **kwargs)
        ...                     for f in fields(cls)))

    Checking this works:

        >>> np.subtract(w2d, w2d)
        Wrap2D(x=array([0, 0, 0]), y=array([0, 0, 0]))

    We would also like to implement the ``accumulate`` method for all the
    ``add_funcs`` overloads:

        >>> @add_assists.register("accumulate")
        ... def add_accumulate_assists(cls, func, w1, *args, **kwargs):
        ...     return cls(*(func(getattr(w1, f.name), *args, **kwargs)
        ...                  for f in fields(cls)))

        >>> np.subtract.accumulate(w2d)
        Wrap2D(x=array([ 0, -1, -3]), y=array([ 3, -1, -6]))
    """

    NP_OVERLOADS: ClassVar[NumPyOverloader]

    def __array_ufunc__(self, ufunc: UFuncLike, method: UFMT, *inputs: Any, **kwargs: Any) -> Any:
        # Check if can be dispatched.
        if not self.NP_OVERLOADS.__contains__(ufunc):
            return NotImplemented

        # Get ImplementsUFunc on function, from single-dispatch on type
        # of self. If a NotImplementedError is raised, it means there's no
        # implementation.
        try:
            ufwrap = self.NP_OVERLOADS[ufunc](self)
        except NotImplementedError:
            return NotImplemented

        # Needed only for type narrowing
        if not isinstance(ufwrap, (ImplementsUFunc, AssistsUFunc)):
            return NotImplemented

        # No need to ``validate_method`` b/c each method does that internally.
        # TODO: handle kwarg ``out``.
        return ufwrap(method, type(self), inputs, kwargs)


##############################################################################
# Convenience mix of both func and ufunc overload mixins


@mypyc_attr(allow_interpreted_subclasses=True)
class NPArrayOverloadMixin(NPArrayFuncOverloadMixin, NPArrayUFuncOverloadMixin):
    """
    Mixin for adding |array_ufunc|_ and |array_function|_ to a class.

    This mixin adds the methods ``__array_ufunc__`` and ``__array_function__``.
    Subclasses must define a class variable ``NP_OVERLOADS`` and optionally
    ``NP_FUNC_TYPES``.

    Attributes
    ----------
    NP_OVERLOADS : |NumPyOverloader|
        A class-attribute of an instance of |NumPyOverloader|.
    NP_FUNC_TYPES : frozenset[type | TypeConstraint] | None, optional
        A class-attribute of `None` or a `frozenset` of `type` or
        |TypeConstraint|.

    Notes
    -----
    When compiled this class is permits interpreted subclasses (see
    https://mypyc.readthedocs.io/en/latest/native_classes.html#inheritance).

    Examples
    --------
    First, some imports:

        >>> from dataclasses import dataclass, fields
        >>> from typing import ClassVar
        >>> import numpy as np
        >>> from overload_numpy import NumPyOverloader, NPArrayOverloadMixin

    Now we can define a |NumPyOverloader| instance:

        >>> W_FUNCS = NumPyOverloader()

    The overloads apply to an array wrapping class. Let's define one:

        >>> @dataclass
        ... class Wrap1D(NPArrayOverloadMixin):
        ...     '''A simple array wrapper.'''
        ...     x: np.ndarray
        ...     NP_OVERLOADS: ClassVar[NumPyOverloader] = W_FUNCS

        >>> w1d = Wrap1D(np.arange(3))

    Now both :class:`numpy.ufunc` (e.g. :obj:`numpy.add`) and :mod:`numpy`
    functions (e.g. :func:`numpy.concatenate`) can be overloaded and registered
    for ``Wrap1D``.

        >>> @W_FUNCS.implements(np.add, Wrap1D)
        ... def add(w1, w2):
        ...     return Wrap1D(np.add(w1.x, w2.x))

        >>> @W_FUNCS.implements(np.concatenate, Wrap1D)
        ... def concatenate(w1ds):
        ...     return Wrap1D(np.concatenate(tuple(w.x for w in w1ds)))

    Time to check these work:

        >>> np.add(w1d, w1d)
        Wrap1D(x=array([0, 2, 4]))

        >>> np.concatenate((w1d, w1d))
        Wrap1D(x=array([0, 1, 2, 0, 1, 2]))

    |ufunc| also have a number of methods: 'at', 'accumulate', etc. The function
    dispatch mechanism in `NEP13
    <https://numpy.org/neps/nep-0013-ufunc-overrides.html>`_ says that  "If one
    of the input or output arguments implements __array_ufunc__, it is executed
    instead of the ufunc." Currently the overloaded `numpy.add` does not work
    for any of the |ufunc| methods.

        >>> try: np.add.accumulate(w1d)
        ... except Exception: print("failed")
        failed

    |ufunc| method overloads can be registered on the wrapped ``add``
    implementation:

        >>> @add.register('accumulate')
        ... def add_accumulate(w1):
        ...     return Wrap1D(np.add.accumulate(w1.x))

        >>> np.add.accumulate(w1d)
        Wrap1D(x=array([0, 1, 3]))

    What if we defined a subclass of ``Wrap1D``?

        >>> @dataclass
        ... class Wrap2D(Wrap1D):
        ...     '''A simple 2-array wrapper.'''
        ...     y: np.ndarray

    The overload for :func:`numpy.concatenate` registered on ``Wrap1D`` will not
    work correctly for ``Wrap2D``. However, |NumPyOverloader| supports
    single-dispatch on the calling type for the overload, so overloads can be
    customized for subclasses.

        >>> @W_FUNCS.implements(np.add, Wrap2D)
        ... def add(w1, w2):
        ...     print("using Wrap2D implementation...")
        ...     return Wrap2D(np.add(w1.x, w2.x),
        ...                   np.add(w1.y, w2.y))

        >>> @W_FUNCS.implements(np.concatenate, Wrap2D)
        ... def concatenate2(w2ds):
        ...     print("using Wrap2D implementation...")
        ...     return Wrap2D(np.concatenate(tuple(w.x for w in w2ds)),
        ...                   np.concatenate(tuple(w.y for w in w2ds)))

    Checking these work:

        >>> w2d = Wrap2D(np.arange(3), np.arange(3, 6))
        >>> np.add(w2d, w2d)
        using Wrap2D implementation...
        Wrap2D(x=array([0, 2, 4]), y=array([ 6, 8, 10]))

        >>> np.concatenate((w2d, w2d))
        using Wrap2D implementation...
        Wrap2D(x=array([0, 1, 2, 0, 1, 2]), y=array([3, 4, 5, 3, 4, 5]))

    Great! But rather than defining a new implementation for each subclass,
    let's see how we could write a more broadly applicable overload:

        >>> @W_FUNCS.implements(np.add, Wrap1D)  # overriding both
        ... @W_FUNCS.implements(np.add, Wrap2D)  # overriding both
        ... def add_general(w1, w2):
        ...     WT = type(w1)
        ...     return WT(*(np.add(getattr(w1, f.name), getattr(w2, f.name))
        ...                 for f in fields(WT)))

        >>> @W_FUNCS.implements(np.concatenate, Wrap1D)  # overriding both
        ... @W_FUNCS.implements(np.concatenate, Wrap2D)  # overriding both
        ... def concatenate_general(ws):
        ...     WT = type(ws[0])
        ...     return WT(*(np.concatenate(tuple(getattr(w, f.name) for w in ws))
        ...                 for f in fields(WT)))

    Checking these work:

        >>> np.add(w2d, w2d)
        Wrap2D(x=array([0, 2, 4]), y=array([ 6, 8, 10]))

        >>> np.concatenate((w2d, w2d))
        Wrap2D(x=array([0, 1, 2, 0, 1, 2]), y=array([3, 4, 5, 3, 4, 5]))

        >>> @dataclass
        ... class Wrap3D(Wrap2D):
        ...     '''A simple 3-array wrapper.'''
        ...     z: np.ndarray

        >>> w3d = Wrap3D(np.arange(2), np.arange(3, 5), np.arange(6, 8))
        >>> np.add(w3d, w3d)
        Wrap3D(x=array([0, 2]), y=array([6, 8]), z=array([12, 14]))
        >>> np.concatenate((w3d, w3d))
        Wrap3D(x=array([0, 1, 0, 1]), y=array([3, 4, 3, 4]), z=array([6, 7, 6, 7]))

    In the previous examples we wrote implementations for a single NumPy
    function. Overloading the full set of NumPy functions this way would take a
    long time.

    Wouldn't it be better if we could write many fewer, based on groups of NumPy
    functions?

        >>> add_funcs = {np.add, np.subtract}
        >>> @W_FUNCS.assists(add_funcs, types=Wrap1D, dispatch_on=Wrap1D)
        ... def add_assists(cls, func, w1, w2, *args, **kwargs):
        ...     return cls(*(func(getattr(w1, f.name), getattr(w2, f.name), *args, **kwargs)
        ...                     for f in fields(cls)))

        >>> stack_funcs = {np.vstack, np.hstack, np.dstack, np.column_stack, np.row_stack}
        >>> @W_FUNCS.assists(stack_funcs, types=Wrap1D, dispatch_on=Wrap1D)
        ... def stack_assists(cls, func, ws, *args, **kwargs):
        ...     return cls(*(func(tuple(getattr(v, f.name) for v in ws), *args, **kwargs)
        ...                     for f in fields(cls)))

    Checking these work:

        >>> np.subtract(w2d, w2d)
        Wrap2D(x=array([0, 0, 0]), y=array([0, 0, 0]))

        >>> np.vstack((w1d, w1d))
        Wrap1D(x=array([[0, 1, 2],
                          [0, 1, 2]]))

        >>> np.hstack((w1d, w1d))
        Wrap1D(x=array([0, 1, 2, 0, 1, 2]))

    We would also like to implement the ``accumulate`` method for all the
    ``add_funcs`` overloads:

        >>> @add_assists.register("accumulate")
        ... def add_accumulate_assists(cls, func, w1, *args, **kwargs):
        ...     return cls(*(func(getattr(w1, f.name), *args, **kwargs)
        ...                  for f in fields(cls)))

        >>> np.subtract.accumulate(w2d)
        Wrap2D(x=array([ 0, -1, -3]), y=array([ 3, -1, -6]))
    """
