"""Configure Test Suite.

This file is used to configure the behavior of pytest when using the Astropy
test infrastructure. It needs to live inside the package in order for it to
get picked up when running the tests inside an interpreter using
packagename.test

"""

from __future__ import annotations

# STDLIB
import os

# THIRD PARTY
import pytest

ASTROPY_HEADER: bool
try:
    # THIRD PARTY
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES  # type: ignore
    from pytest_astropy_header.display import TESTED_VERSIONS  # type: ignore

    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False
else:
    assert isinstance(PYTEST_HEADER_MODULES, dict)


def pytest_configure(config: pytest.Config) -> None:
    """Configure Pytest with Astropy.

    Parameters
    ----------
    config : pytest configuration
    """
    if not ASTROPY_HEADER:
        return None

    config.option.astropy_header = True

    # Customize the following lines to add/remove entries from the list of
    # packages for which version numbers are displayed when running the
    # tests.
    PYTEST_HEADER_MODULES.pop("Pandas", None)

    # STDLIB
    # from stream import __version__  # type: ignore
    from importlib.metadata import version

    packagename = os.path.basename(os.path.dirname(__file__))
    TESTED_VERSIONS[packagename] = version("overload_numpy")
