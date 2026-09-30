"""G1-W1 adversarial check: each weakening of ingestion must break a test.

Usage: python scripts/ingestion_mutations.py <clean-tree-copy> <scratch-dir>

Every mutation is applied to a fresh copy of the tree, never to the working
tree, and the tests named beside it are run there. A mutation that leaves its
tests passing is a claim the suite does not really defend.
"""
import shutil, subprocess, sys, os
from pathlib import Path
SRC = Path(sys.argv[1]); WORK = Path(sys.argv[2])
M = [
 ("1 pd.read_csv default delimiter", "aidatasetkit/ingestion/loader.py", "sep=plan.delimiter", "sep=','", "tests/unit/test_ingestion.py tests/integration/test_cli_ingestion.py -k semicolon"),
 ("2 delimiter from extension only", "aidatasetkit/ingestion/delimited.py", "    qualifying = [d for d in SUPPORTED_DELIMITERS if shapes[d].consistent]\n", "    return ',', DelimiterSource.DETECTED, False\n", "tests/unit/test_ingestion.py -k tab_separated_file_named_csv"),
 ("3 silent ambiguity", "aidatasetkit/ingestion/delimited.py", "    if len(qualifying) > 1:\n", "    if len(qualifying) > 1:\n        return qualifying[0], DelimiterSource.DETECTED, False\n", "tests/unit/test_ingestion.py -k ambiguous"),
 ("4 pandas renames duplicates", "aidatasetkit/ingestion/delimited.py", "        if duplicated:\n", "        if False:\n", "tests/unit/test_ingestion.py -k duplicate_headers"),
 ("5 raw UnicodeDecodeError", "aidatasetkit/ingestion/delimited.py", "        raise _encoding_error(path, encoding, error) from None\n    return sample", "        raise\n    return sample", "tests/unit/test_ingestion.py -k wrong_encoding"),
 ("6 metadata dropped from evidence", "aidatasetkit/cli/main.py", "ingestion=loaded.metadata", "ingestion=None", "tests/integration/test_cli_ingestion.py -k \"semicolon_file or input_section\""),
 ("7 absolute path in artifact", "aidatasetkit/cli/main.py", "ingestion=loaded.metadata", "ingestion=__import__('dataclasses').replace(loaded.metadata, warnings=(str(path.resolve()),))", "tests/integration/test_cli_ingestion.py -k absolute_path"),
 ("8 partial output before ingestion", "aidatasetkit/cli/main.py", "    loaded = load_table(\n", "    (args.output / 'runs' / 'partial').mkdir(parents=True, exist_ok=True)\n    loaded = load_table(\n", "tests/integration/test_cli_ingestion.py -k \"ambiguous_file_is_refused or structured_refusals\""),
 ("9 bypass source byte limit", "aidatasetkit/ingestion/loader.py", "    if limits.max_source_bytes is not None:\n", "    if False:\n", "tests/unit/test_ingestion_limits.py -k file_bytes"),
 ("10 reject exact byte boundary", "aidatasetkit/ingestion/loader.py", "if size > limits.max_source_bytes:", "if size >= limits.max_source_bytes:", "tests/unit/test_ingestion_limits.py -k file_bytes"),
 ("11 skip dataframe cell limit", "aidatasetkit/ingestion/loader.py", "if limits.max_cells is not None and _cells_exceed(rows, columns, limits.max_cells):", "if False:", "tests/unit/test_ingestion_limits.py -k dataframe"),
 ("12 skip record count limit", "aidatasetkit/ingestion/loader.py", "if limits.max_records is not None and len(records) > limits.max_records:", "if False:", "tests/unit/test_ingestion_limits.py -k records"),
 ("13 make cli unlimited", "aidatasetkit/cli/main.py", "_DEFAULT_INGESTION_LIMITS = IngestionLimits()", "_DEFAULT_INGESTION_LIMITS = IngestionLimits(max_source_bytes=None, max_rows=None, max_columns=None, max_cells=None, max_field_length=None)", "tests/unit/test_ingestion_limits.py -k cli_defaults"),
]
for name, rel, old, new, tests in M:
    if WORK.exists(): shutil.rmtree(WORK)
    shutil.copytree(SRC, WORK, ignore=shutil.ignore_patterns(".git", ".pytest_cache", "__pycache__", "*.pyc"))
    f = WORK / rel; t = f.read_bytes().replace(b"\r\n", b"\n").decode()
    assert t.count(old) >= 1, (name, "anchor missing")
    f.write_bytes(t.replace(old, new, 1).encode())
    env = dict(os.environ, PYTHONPATH=str(WORK), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(f'"{sys.executable}" -m pytest -p no:cacheprovider -q {tests}', cwd=WORK, env=env, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    last = [l for l in r.stdout.splitlines() if " passed" in l or " failed" in l]
    print(f"{name:40s} exit={r.returncode} {last[-1] if last else r.stdout[-300:]}")
