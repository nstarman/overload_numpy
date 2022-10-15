"""Configure Test Suite.

This file is used to configure the behavior of pytest when using the Astropy
test infrastructure. It needs to live inside the package in order for it to
get picked up when running the tests inside an interpreter using
packagename.test

"""

from __future__ import annotations

# STDLIB
import os
from typing import Any

# THIRDPARTY
import pytest
from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS


def pytest_configure(config: pytest.Config) -> None:
    """Configure Pytest with Astropy.

    Parameters
    ----------
    config : pytest configuration
    """
    config.option.astropy_header = True

    # Customize the following lines to add/remove entries from the list of
    # packages for which version numbers are displayed when running the
    # tests.
    PYTEST_HEADER_MODULES.pop("Pandas", None)

    # STDLIB
    from importlib.metadata import version

    packagename = os.path.basename(os.path.dirname(__file__))
    TESTED_VERSIONS[packagename] = version("overload_numpy")


@pytest.fixture(autouse=True)  # type: ignore
def add_numpy(doctest_namespace: dict[str, Any]) -> None:
    """Add NumPy to Pytest.

    Parameters
    ----------
    doctest_namespace : namespace

    """
    # THIRDPARTY
    import numpy

    # add to namespace
    doctest_namespace["np"] = numpy
