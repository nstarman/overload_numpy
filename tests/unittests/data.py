##############################################################################
# IMPORTS

# STDLIB
from dataclasses import dataclass

__all__ = ["A", "B", "C", "D", "E"]


##############################################################################
# CODE
##############################################################################


@dataclass
class A:
    x: int


@dataclass
class B(A):
    pass


@dataclass
class C(B):
    pass


@dataclass
class D(C):
    pass


@dataclass
class E(D):
    pass
