"""The command line: one command, and no analysis of its own.

``aidatasetkit audit`` reads a CSV, calls the public components in order, and
writes what they produced. Every judgement in the output came from a layer below
this one; the CLI's contribution is argument parsing, file handling, a readable
summary, and an exit code a CI job can branch on.
"""

from aidatasetkit.cli.main import EXIT_CODES, build_parser, main

__all__ = ["EXIT_CODES", "build_parser", "main"]
