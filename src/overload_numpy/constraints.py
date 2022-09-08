##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

# THIRD PARTY
from mypy_extensions import mypyc_attr

__all__ = ["TypeConstraint", "Invariant", "Covariant", "Contravariant", "Between"]


##############################################################################
# CODE
##############################################################################


@mypyc_attr(allow_interpreted_subclasses=True)
class TypeConstraint(metaclass=ABCMeta):
    """ABC for constraining an argument type.

    .. warning::

        This class will be converted to a runtime-checkable `Protocol` when
        mypyc behaves nicely with runtime_checkable interpreted subclasses.
        Related to https://github.com/mypyc/mypyc/issues/909.
    """

    @abstractmethod
    def validate_type(self, arg_type: type) -> bool:
        """Validate the argument type.

        Parameters
        ----------
        arg_type : type
            The type of the argument that must fit the type con

        Returns
        -------
        bool
            Whether the type is valid.
        """
        ...


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass
class Invariant(TypeConstraint):
    """Type constraint for invariance -- the exact type.

    This is equivalent to ``arg_type is bound``.

    Parameters
    ----------
    bound : type
        The exact type of the argument.

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

    def validate_type(self, arg_type: type) -> bool:
        """Validate argument type given constraint.

        Parameters
        ----------
        arg_type : type

        Returns
        -------
        bool
        """
        return arg_type is self.bound


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass(frozen=True)
class Covariant(TypeConstraint):
    """A covariant constraint -- permitting subclasses.

    This is the most common constraint, equivalent to ``issubclass(arg_type,
    bound)``.

    Parameters
    ----------
    bound : type
        The parent type of the argument.

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

    def validate_type(self, arg_type: type) -> bool:
        return issubclass(arg_type, self.bound)


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass(frozen=True)
class Contravariant(TypeConstraint):
    """A contravariant constraint -- permitting superclasses.

    An uncommon constraint. See examples for why.

    Parameters
    ----------
    bound : type
        The child type of the argument.

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

    def validate_type(self, arg_type: type) -> bool:
        return issubclass(self.bound, arg_type)


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass(frozen=True)
class Between(TypeConstraint):
    """Type constrained between two types.

    Parameters
    ----------
    lower_bound : type
        The child type of the argument.
    upper_bound : type
        The parent type of the argument.

    Examples
    --------
    For this example we need a type heirarchy:

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

    def validate_type(self, arg_type: type) -> bool:
        return issubclass(self.lower_bound, arg_type) & issubclass(arg_type, self.upper_bound)

    @property
    def bounds(self) -> tuple[type, type]:
        """Return tuple of lower and upper bounds."""
        return (self.lower_bound, self.upper_bound)
