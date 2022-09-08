##############################################################################
# IMPORTS

# STDLIB
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Protocol

# THIRD PARTY
import pytest

# LOCAL
from overload_numpy.constraints import (
    Between,
    Contravariant,
    Covariant,
    Invariant,
    TypeConstraint,
)

##############################################################################
# IMPORTS


@dataclass
class A:
    x: int


@dataclass
class B(A):
    pass


@dataclass
class C(B):
    pass


##############################################################################
# TESTS
##############################################################################


@pytest.mark.xfail
def test_TypeConstraint_Protocol():

    assert issubclass(TypeConstraint, Protocol)
    assert hasattr(TypeConstraint, "validate_type")


class TypeConstraint_TestBase(metaclass=ABCMeta):
    @abstractmethod
    def constraint_cls(self) -> type:
        raise NotImplementedError

    @abstractmethod
    def constraint(self, constraint_cls) -> TypeConstraint:
        raise NotImplementedError

    @pytest.fixture
    def types(self):
        return (A, B, C)

    # ===============================================================
    # Method Tests

    @abstractmethod
    def test_validate_type(self, constraint, types) -> None:
        pass


class Test_Invariant(TypeConstraint_TestBase):
    @pytest.fixture
    def constraint_cls(self) -> type:
        return Invariant

    @pytest.fixture
    def constraint(self, constraint_cls) -> TypeConstraint:
        return constraint_cls(B)

    # ===============================================================
    # Method Tests

    def test_validate_type(self, constraint, types) -> None:
        A, B, C = types

        assert constraint.validate_type(A) is False
        assert constraint.validate_type(B) is True
        assert constraint.validate_type(C) is False


class Test_Covariant(TypeConstraint_TestBase):
    @pytest.fixture
    def constraint_cls(self) -> type:
        return Covariant

    @pytest.fixture
    def constraint(self, constraint_cls) -> TypeConstraint:
        return constraint_cls(B)

    # ===============================================================
    # Method Tests

    def test_validate_type(self, constraint, types) -> None:
        A, B, C = types

        assert constraint.validate_type(A) is False
        assert constraint.validate_type(B) is True
        assert constraint.validate_type(C) is True


class Test_Contravariant(TypeConstraint_TestBase):
    @pytest.fixture
    def constraint_cls(self) -> type:
        return Contravariant

    @pytest.fixture
    def constraint(self, constraint_cls) -> TypeConstraint:
        return constraint_cls(B)

    # ===============================================================
    # Method Tests

    def test_validate_type(self, constraint, types) -> None:
        A, B, C = types

        assert constraint.validate_type(A) is True
        assert constraint.validate_type(B) is True
        assert constraint.validate_type(C) is False


class Test_Between(TypeConstraint_TestBase):
    @pytest.fixture
    def constraint_cls(self) -> type:
        return Between

    @pytest.fixture
    def constraint(self, constraint_cls) -> TypeConstraint:
        return constraint_cls(B, A)

    # ===============================================================
    # Method Tests

    def test_validate_type(self, constraint, types) -> None:
        A, B, C = types

        assert constraint.validate_type(A) is True
        assert constraint.validate_type(B) is True
        assert constraint.validate_type(C) is False
