##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import inspect
from itertools import permutations

# THIRDPARTY
import numpy as np
import pytest

# LOCAL
from overload_numpy.utils import UFMT, VALID_UFUNC_METHODS, _get_key, _parse_methods

##############################################################################
# TESTS
##############################################################################


class Test__parse_methods:
    @pytest.mark.parametrize("method", VALID_UFUNC_METHODS)
    def test_method(self, method):
        assert _parse_methods(method) == {method}

    @pytest.mark.parametrize("methods", [set(p) for p in permutations(VALID_UFUNC_METHODS, r=2)])
    def test_methods(self, methods: set[UFMT]):
        assert _parse_methods(methods) == methods

    @pytest.mark.parametrize("method", ["a", {"a", "b"}])
    def test_error(self, method):
        with pytest.raises(ValueError, match="methods must be an element or subset"):
            _parse_methods(method)


class Test__get_key:
    @pytest.mark.parametrize("key", ["a", "b"])
    def test_str(self, key):
        assert _get_key(key) == key

    @pytest.mark.parametrize("key", [f for f in (getattr(np, n) for n in dir(np)) if isinstance(f, np.ufunc)])
    def test_ufunc(self, key):
        assert _get_key(key) == f"{key.__class__.__module__}.{key.__name__}"

    def test_ufunc_examples(self):
        assert _get_key(np.add) == "numpy.add"
        assert _get_key(np.subtract) == "numpy.subtract"

    @pytest.mark.parametrize(
        "key",
        [
            f
            for f in (getattr(np, n) for n in dir(np))
            if callable(f)
            and not inspect.isclass(f)
            and (hasattr(f, "__module__") and hasattr(f, "__name__"))
            and not isinstance(f, np.ufunc)
        ],
    )
    def test_func(self, key):
        assert _get_key(key) == f"{key.__module__}.{key.__name__}"

    def test_func_examples(self):
        assert _get_key(np.concatenate) == "numpy.concatenate"
        assert _get_key(np.vstack) == "numpy.vstack"

    @pytest.mark.parametrize("key", [object(), 1])
    def test_error(self, key):
        with pytest.raises(ValueError, match=f"the key {key}"):
            _get_key(key)
