"""Classes for defining type constraints in |array_function|_.

|array_function| has an argument ``types``, which is a "collection
:class:`collections.abc.Collection` of unique argument types from the original
NumPy function call that implement |array_function|. The purpose of ``types`` is
to allow implementations of |array_function| to check if all arguments of a type
that the overload knows how to handle. Normally this is implemented inside of
|array_function|, but :mod:`overload_numpy` gives overloading functions more
flexibility to set constrains on a per-overloaded function basis.

Examples
--------
First, some imports:

    >>> from dataclasses import dataclass
    >>> from typing import ClassVar
    >>> import numpy as np
    >>> from overload_numpy import NumPyOverloader, NPArrayFuncOverloadMixin

Now we can define a |NumPyOverloader| instance:

    >>> W_FUNCS = NumPyOverloader()

The overloads apply to an array wrapping class. Let's define one, and a subclass
that contains more information:

    >>> @dataclass
    ... class Wrap1D(NPArrayFuncOverloadMixin):
    ...     NP_OVERLOADS: ClassVar[NumPyOverloader] = W_FUNCS
    ...     NP_FUNC_TYPES: ClassVar[None] = None
    ...     x: np.ndarray

Note the ``NP_FUNC_TYPES``. Normally this would inherit from
:class:`~overload_numpy.NPArrayFuncOverloadMixin` and be an empty `frozenset`,
signalling that types are covariant with ``Wrap1D`` (includes subclasses).
Changing ``NP_FUNC_TYPES`` to `None` means each overload must explicitly define
the ``types`` argument. See :class:`~overload_numpy.NPArrayFuncOverloadMixin`
for further details.

Now :mod:`numpy` functions can be overloaded and registered for ``Wrap1D``. Here
is where we introduce the type constraints:

    >>> from overload_numpy.constraints import Invariant, Covariant
    >>> invariant = Invariant(Wrap1D)  # only works for Wrap1D

    >>> @W_FUNCS.implements(np.concatenate, Wrap1D, types=invariant)
    ... def concatenate(w1ds):
    ...     return Wrap1D(np.concatenate(tuple(w.x for w in w1ds)))

    >>> w1d = Wrap1D(np.arange(3))
    >>> np.concatenate((w1d, w1d))
    Wrap1D(x=array([0, 1, 2, 0, 1, 2]))

This implementation of `numpy.concatenate` is invariant on the type of
``Wrap1D``, so should fail if we use a subclass:

    >>> @dataclass
    ... class Wrap2D(Wrap1D):
    ...     '''A simple 2-array wrapper.'''
    ...     y: np.ndarray

    >>> w2d = Wrap2D(np.arange(3), np.arange(3, 6))
    >>> try: np.concatenate((w2d, w2d))
    >>> except Exception as e: print(e)
    there is no implementation

Normally overlaoded functions are made to be
:class:`~overload_numpy.constraints.Covariant`. As this is the default, just
passing the type is a convenience short-hand.

    >>> @W_FUNCS.implements(np.concatenate, Wrap1D, types=Wrap1D)
    ... def concatenate(w1ds):
    ...     return Wrap1D(*(np.concatenate(tuple(getattr(w, f.name) for w in w1ds))
                            for f in fields(wlds[0])))

    >>> np.concatenate((w2d, w2d))
    Wrap1D(x=array([0, 1, 2, 0, 1, 2]), y=array([3, 4, 5, 3, 4, 5]))

This module offers other types of constraints, so be sure to check them out.
Also, if you need something more specific, it's easy to make your own
constraint. There are currently two things you need to do:

    1. subclass :class:`overload_numpy.constraints.TypeConstraint`
    2. define a method ``validate_type``

As an example, let's define a constraint where the argument must be one of 2
types:

    >>> from overload_numpy.constraints import TypeConstraint
    >>> @dataclass(frozen=True)
    ... class ThisOrThat(TypeConstraint):
    ...     this: type
    ...     that: type
    ...     def validate_type(self, arg_type: type, /) -> bool:
    ...         return arg_type is self.this or arg_type is self.that

.. note::

    TypeConstraint will eventually be converted to a runtime-checkable
    :class:`typing.Protocol`. When that happens step 1 (subclassing
    TypeConstraint) will become optional.
"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

# THIRDPARTY
from mypy_extensions import mypyc_attr

__all__ = ["TypeConstraint", "Invariant", "Covariant", "Contravariant", "Between"]
__doctest_skip__ = ["*"]  # TODO! figure out weird dataclass error

##############################################################################
# CODE
##############################################################################


@mypyc_attr(allow_interpreted_subclasses=True)
class TypeConstraint(metaclass=ABCMeta):
    r"""ABC for constraining an argument type.

    .. warning::

        This class will be converted to a runtime-checkable `Protocol` when
        mypyc behaves nicely with runtime_checkable interpreted subclasses
        (see https://github.com/mypyc/mypyc/issues/909).

    Examples
    --------
    It's very easy to define a custom type constraint.

        >>> from dataclasses import dataclass
        >>> from overload_numpy.constraints import TypeConstraint

        >>> @dataclass(frozen=True)
        ... class ThisOrThat(TypeConstraint):
        ...     this: type
        ...     that: type
        ...     def validate_type(self, arg_type: type, /) -> bool:
        ...         return arg_type is self.this or arg_type is self.that
    """

    @abstractmethod
    def validate_type(self, arg_type: type, /) -> bool:
        """Validate the argument type.

        This is used in :class:`overload_numpy.mixin.NPArrayFuncOverloadMixin`
        and subclasses like :class:`overload_numpy.mixin.NPArrayOverloadMixin`
        to ensure that the input is of the correct set of types to work
        with the |array_function|_ override.

        Parameters
        ----------
        arg_type : type, positional-only
            The type of the argument that must fit the type constraint.

        Returns
        -------
        bool
            Whether the type is valid.

        Examples
        --------
        The simplest built-in type constraint is
        :class:`overload_numpy.constraints.Invariant`.

            >>> from overload_numpy.constraints import Invariant
            >>> constraint = Invariant(int)
            >>> constraint.validate_type(int)  # exact type
            True
            >>> constraint.validate_type(bool)  # subclass
            False
        """

    def validate_object(self, arg: object, /) -> bool:
        """Validate an argument.

        This is used in :class:`overload_numpy.mixin.NPArrayFuncOverloadMixin`
        and subclasses like :class:`overload_numpy.mixin.NPArrayOverloadMixin`
        to ensure that the input is of the correct set of types to work
        with the |array_function|_ override.

        Parameters
        ----------
        arg : object, positional-only
            The argument that's type must fit the type constraint.

        Returns
        -------
        bool
            Whether the type is valid.

        Examples
        --------
        The simplest built-in type constraint is
        :class:`overload_numpy.constraints.Invariant`.

            >>> from overload_numpy.constraints import Invariant
            >>> constraint = Invariant(int)
            >>> constraint.validate_type(int)  # exact type
            True
            >>> constraint.validate_type(bool)  # subclass
            False
        """
        return self.validate_type(type(arg))


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass(frozen=True)
class Invariant(TypeConstraint):
    r"""
    Type constraint for invariance -- the exact type.

    This is equivalent to ``arg_type is bound``.

    Parameters
    ----------
    bound : type
        The exact type of the argument.

    Notes
    -----
    When compiled this class permits interpreted subclasses, see
    https://mypyc.readthedocs.io/en/latest/native_classes.html.

    Examples
    --------
    Construct the constraint object:

        >>> constraint = Invariant(int)

    This can be used to validate argument types:

        >>> constraint.validate_type(int)  # exact type
        True
        >>> constraint.validate_type(bool)  # subclass
        False
        >>> constraint.validate_type(object)  # superclass
        False
    """

    bound: type

    def validate_type(self, arg_type: type, /) -> bool:  # noqa: D102
        return arg_type is self.bound


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass(frozen=True)
class Covariant(TypeConstraint):
    r"""
    A covariant constraint -- permitting subclasses.

    This is the most common constraint, equivalent to ``issubclass(arg_type,
    bound)``.

    Parameters
    ----------
    bound : type
        The parent type of the argument.

    Notes
    -----
    When compiled this class permits interpreted subclasses, see
    https://mypyc.readthedocs.io/en/latest/native_classes.html.

    Examples
    --------
    Construct the constraint object:

        >>> constraint = Covariant(int)

    This can be used to validate argument types:

        >>> constraint.validate_type(int)  # exact type
        True
        >>> constraint.validate_type(bool)  # subclass
        True
        >>> constraint.validate_type(object)  # superclass
        False
    """

    bound: type

    def validate_type(self, arg_type: type, /) -> bool:  # noqa: D102
        return issubclass(arg_type, self.bound)


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass(frozen=True)
class Contravariant(TypeConstraint):
    r"""
    A contravariant constraint -- permitting superclasses.

    An uncommon constraint. See examples for why.

    Parameters
    ----------
    bound : type
        The child type of the argument.

    Notes
    -----
    When compiled this class permits interpreted subclasses, see
    https://mypyc.readthedocs.io/en/latest/native_classes.html.

    Examples
    --------
    Construct the constraint object:

        >>> constraint = Contravariant(int)

    This can be used to validate argument types:

        >>> constraint.validate_type(int)  # exact type
        True
        >>> constraint.validate_type(bool)  # subclass
        False
        >>> constraint.validate_type(object)  # superclass
        True
    """

    bound: type

    def validate_type(self, arg_type: type, /) -> bool:  # noqa: D102
        return issubclass(self.bound, arg_type)


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass(frozen=True)
class Between(TypeConstraint):
    r"""
    Type constrained between two types.

    This combines the functionality of
    :class:`~overload_numpy.constraints.Covariant` and
    :class:`~overload_numpy.constraints.Contravariant`.

    Parameters
    ----------
    lower_bound : type
        The child type of the argument.
    upper_bound : type
        The parent type of the argument.

    Notes
    -----
    When compiled this class permits interpreted subclasses, see
    https://mypyc.readthedocs.io/en/latest/native_classes.html

    Examples
    --------
    For this example we need a type hierarchy:

        >>> class A: pass
        >>> class B(A): pass
        >>> class C(B): pass
        >>> class D(C): pass
        >>> class E(D): pass

    Construct the constraint object:

        >>> constraint = Between(D, B)

    This can be used to validate argument types:

        >>> constraint.validate_type(A)
        False
        >>> constraint.validate_type(B)
        True
        >>> constraint.validate_type(C)
        True
        >>> constraint.validate_type(D)
        True
        >>> constraint.validate_type(E)
        False
    """

    lower_bound: type
    upper_bound: type

    def validate_type(self, arg_type: type, /) -> bool:  # noqa: D102
        return issubclass(self.lower_bound, arg_type) & issubclass(arg_type, self.upper_bound)

    @property
    def bounds(self) -> tuple[type, type]:
        """Return tuple of (lower, upper) bounds.

        The lower bound is contravariant, the upper bound covariant.

        Examples
        --------
        For this example we need a type hierarchy:

            >>> class A: pass
            >>> class B(A): pass
            >>> class C(B): pass

            >>> constraint = Between(C, B)
            >>> constraint.bounds
            (C, B)
        """
        return (self.lower_bound, self.upper_bound)
