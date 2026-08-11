"""The command line: one command, and no analysis of its own.

``aidatasetkit audit`` reads a CSV, calls the public components in order, and
writes what they produced. Every judgement in the output came from a layer below
this one; the CLI's contribution is argument parsing, file handling, a readable
summary, and an exit code a CI job can branch on.
"""

from aidatasetkit.cli.main import EXIT_CODES, build_parser, main

#: Note for anyone reaching for the module rather than the function: re-exporting
#: ``main`` here binds that name in this namespace to the *function*, which
#: shadows the ``main`` submodule. So ``import aidatasetkit.cli.main`` hands back
#: the function, not the module. The console script and ``python -m`` both
#: resolve the module path correctly and are unaffected; code that genuinely
#: wants the module should ask for it by its full name:
#:
#:     import sys
#:     module = sys.modules["aidatasetkit.cli.main"]
#:
#: The convention (``cli/main.py`` holding ``main()``) is worth more than the
#: collision costs, and naming the collision here is cheaper than renaming either.
__all__ = ["EXIT_CODES", "build_parser", "main"]
