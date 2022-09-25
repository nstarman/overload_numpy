##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import itertools
from collections.abc import Set
from typing import Any, Callable, Iterator, TypeVar

# THIRDPARTY
import numpy as np

##############################################################################
# TYPING

T = TypeVar("T")


##############################################################################
# CODE
##############################################################################


class ChainSet(Set[T]):
    """Chain of sets.

    Parameters
    ----------
    *sets : set[T]
        set of sets.
    """

    def __init__(self, *sets: set[T]) -> None:
        """
        Initialize a ChainMap by setting *sets* to the given `set` objects.
        If no sets are provided, a single empty `set` is used.
        """
        self.sets: list[set[T]] = list(sets) or [set()]  # always at least one set

    def __contains__(self, key: Any) -> bool:
        return any(key in s for s in self.sets)

    def __iter__(self) -> Iterator[T]:
        return itertools.chain(*self.sets)

    def __len__(self) -> int:
        return sum(map(len, self.sets))


##############################################################################
# ALL NUMPY FUNCTIONS

# -- Array creation routines ------------------------

array_creation: set[Callable[..., Any]] = {
    np.empty,
    np.empty_like,
    np.eye,
    np.identity,
    np.ones,
    np.ones_like,
    np.zeros,
    np.zeros_like,
    np.full,
    np.full_like,
}

from_existing_data: set[Callable[..., Any]] = {
    np.array,
    np.asarray,
    np.asanyarray,
    np.ascontiguousarray,
    np.asmatrix,
    np.copy,
    np.frombuffer,
    # np.from_dlpack,
    np.fromfile,
    np.fromfunction,
    np.fromiter,
    np.fromstring,
    np.loadtxt,
}

creating_record_arrays: set[Callable[..., Any]] = {
    np.rec.array,
    np.rec.fromarrays,
    np.rec.fromrecords,
    np.rec.fromstring,
    np.rec.fromfile,
}

creating_character_arrays: set[Callable[..., Any]] = {np.char.array, np.char.asarray}  # np.char.chararray

numerical_ranges: set[Callable[..., Any]] = {
    np.arange,
    np.linspace,
    np.logspace,
    np.geomspace,
    np.meshgrid,
    # np.mgrid,  # FIXME!
    # np.ogrid,
}

building_matrices: set[Callable[..., Any]] = {np.diag, np.diagflat, np.tri, np.tril, np.triu, np.vander}

matrix_class: set[Callable[..., Any]] = {np.mat, np.bmat}

# -- Array manipulation routines ------------------------

basic_operations: set[Callable[..., Any]] = {np.copyto, np.shape}

changing_array_shape: set[Callable[..., Any]] = {np.reshape, np.ravel}

transpose_like_operations: set[Callable[..., Any]] = {np.moveaxis, np.rollaxis, np.swapaxes, np.transpose}

changing_number_of_dimensions: set[Callable[..., Any]] = {
    np.atleast_1d,
    np.atleast_2d,
    np.atleast_3d,
    np.broadcast,
    np.broadcast_to,
    np.broadcast_arrays,
    np.expand_dims,
    np.squeeze,
}

changing_kind_of_array: set[Callable[..., Any]] = {
    np.asarray,
    np.asanyarray,
    np.asmatrix,
    np.asfarray,
    np.asfortranarray,
    np.ascontiguousarray,
    np.asarray_chkfinite,
    np.require,
}

joining_arrays: set[Callable[..., Any]] = {
    np.concatenate,
    np.stack,
    np.block,
    np.vstack,
    np.hstack,
    np.dstack,
    np.column_stack,
    np.row_stack,
}

splitting_arrays: set[Callable[..., Any]] = {np.split, np.array_split, np.dsplit, np.hsplit, np.vsplit}

tiling_arrays: set[Callable[..., Any]] = {np.tile, np.repeat}

adding_and_removing_elements: set[Callable[..., Any]] = {
    np.delete,
    np.insert,
    np.append,
    np.resize,
    np.trim_zeros,
    np.unique,
}

rearranging_elements: set[Callable[..., Any]] = {np.flip, np.fliplr, np.flipud, np.reshape, np.roll, np.rot90}

# -- Binary operations ------------------------

elementwise_bit_operations: set[Callable[..., Any]] = {
    np.bitwise_and,
    np.bitwise_or,
    np.bitwise_xor,
    np.invert,
    np.left_shift,
    np.right_shift,
}

bit_packing: set[Callable[..., Any]] = {np.packbits, np.unpackbits}

output_formatting: set[Callable[..., Any]] = {np.binary_repr}


# -- String operations ------------------------

string_operations: set[Callable[..., Any]] = {
    np.char.add,
    np.char.multiply,
    np.char.mod,
    np.char.capitalize,
    np.char.center,
    np.char.decode,
    np.char.encode,
    np.char.expandtabs,
    np.char.join,
    np.char.ljust,
    np.char.lower,
    np.char.lstrip,
    np.char.partition,
    np.char.replace,
    np.char.rjust,
    np.char.rpartition,
    np.char.rsplit,
    np.char.rstrip,
    np.char.split,
    np.char.splitlines,
    np.char.strip,
    np.char.swapcase,
    np.char.title,
    np.char.translate,
    np.char.upper,
    np.char.zfill,
}

string_comparison: set[Callable[..., Any]] = {
    np.char.equal,
    np.char.not_equal,
    np.char.greater_equal,
    np.char.less_equal,
    np.char.greater,
    np.char.less,
    np.char.compare_chararrays,
}

string_information: set[Callable[..., Any]] = {
    np.char.count,
    np.char.endswith,
    np.char.find,
    np.char.index,
    np.char.isalpha,
    np.char.isalnum,
    np.char.isdecimal,
    np.char.isdigit,
    np.char.islower,
    np.char.isnumeric,
    np.char.isspace,
    np.char.istitle,
    np.char.isupper,
    np.char.rfind,
    np.char.rindex,
    np.char.startswith,
    np.char.str_len,
}

# -- emath ------------------------

emath: set[Callable[..., Any]] = {getattr(np.emath, n) for n in np.emath.__all__}


# -- FFTs ------------------------

# TODO!

# -- Functional programming ------------------------

# TODO!

# -- NumPy-specific help functions ------------------------

# TODO!

# -- NumPy-specific help functions ------------------------

# TODO!


##############################################################################

has_like: set[Callable[..., Any]] = {
    np.empty,
    np.eye,
    np.identity,
    np.ones,
    np.zeros,
    np.full,
    np.array,
    np.asarray,
    np.asanyarray,
    np.ascontiguousarray,
    np.frombuffer,
    np.fromfile,
    np.fromfunction,
    np.fromiter,
    np.fromstring,
}

as_like: set[Callable[..., Any]] = {np.empty_like, np.ones_like, np.zeros_like, np.full_like}

to_sort: set[Callable[..., Any]] = {
    np.asmatrix,
}
