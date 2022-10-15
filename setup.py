##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import os
import sys
from pathlib import Path

# THIRDPARTY
from mypyc.build import mypycify
from setuptools import setup

##############################################################################
# PARAMETERS

USE_MYPYC: bool = False

CURRENT_DIR = Path(__file__).parent
SRC = CURRENT_DIR / "src"

sys.path.insert(0, str(CURRENT_DIR))  # for setuptools.build_meta


##############################################################################
# CODE
##############################################################################


def find_python_files(base: Path, exclude: tuple[str, ...] = ("test_",)) -> list[Path]:
    """Recursively find python files in all subfolders of base.

    Parameters
    ----------
    base : Path
        Base path from which to search for python files.
    exclude : tuple[str, ...], optional
        Paths to exclude.

    Returns
    -------
    list[Path]
    """
    files: list[Path] = []

    for entry in base.iterdir():
        if entry.name.startswith(exclude):
            continue
        if entry.is_file() and entry.suffix == ".py":
            files.append(entry)
        elif entry.is_dir():
            files.extend(find_python_files(entry))

    return files


# To compile with mypyc, a mypyc checkout must be present on the PYTHONPATH
if len(sys.argv) > 1 and sys.argv[1] == "--use-mypyc":
    sys.argv.pop(1)
    USE_MYPYC = True
if os.getenv("NPOVERLOAD_USE_MYPYC", None) == "1":
    USE_MYPYC = True


if not USE_MYPYC:
    ext_modules = []

else:
    print("BUILDING `overload_numpy` WITH MYPYC")

    blocklist = [  # TODO!
        "overload_numpy/implementors/dispatch.py",  # https://github.com/python/mypy/issues/13613
        "overload_numpy/implementors/ufunc.py",  # FIXME: can't call ImplementsUFunc
        "overload_numpy/_typeutils.py",  # https://github.com/mypyc/mypyc/issues/909
    ]
    discovered: list[Path] = []
    discovered.extend(find_python_files(SRC / "overload_numpy"))
    mypyc_targets = [str(p) for p in discovered if p.relative_to(SRC).as_posix() not in blocklist]

    opt_level = os.getenv("MYPYC_OPT_LEVEL", "3")
    ext_modules = mypycify(mypyc_targets, opt_level=opt_level, verbose=True)


setup(name="overload_numpy", package_dir={"": "src"}, ext_modules=ext_modules)
