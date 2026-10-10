#!/usr/bin/env bash
# The pre-release gate: build, install into a throwaway environment, and prove
# a new user's first five minutes work.
#
# Nothing here touches the development virtual environment, and nothing is
# published. Run it before every release; it is the only check that catches a
# package which imports fine in the source tree and is broken once installed.
#
#   ./scripts/release_smoke_test.sh
#
# Exit 0 means the distribution is installable and the CLI works from it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
fail() { printf '\033[31mFAILED: %s\033[0m\n' "$1" >&2; exit 1; }

say "Building wheel and sdist"
rm -rf "$ROOT/dist" "$ROOT/build"
python -m build --outdir "$ROOT/dist" "$ROOT" >/dev/null
WHEEL="$(ls "$ROOT"/dist/*.whl)"
SDIST="$(ls "$ROOT"/dist/*.tar.gz)"
ls -lh "$ROOT"/dist/

say "Validating package metadata"
python -m twine check "$ROOT"/dist/* || fail "twine check"

say "Checking the distributions carry nothing they should not"
python - "$WHEEL" <<'PY'
import sys, zipfile
bad = [n for n in zipfile.ZipFile(sys.argv[1]).namelist()
       if any(x in n for x in (".venv", ".git/", "__pycache__", ".pytest_cache"))]
raise SystemExit(f"unwanted files in wheel: {bad}" if bad else 0)
PY

# A fresh interpreter with nothing preinstalled: the only honest test of a
# package is one run somewhere the source tree cannot be found.
for KIND in wheel sdist; do
  say "Installing the $KIND into a clean environment"
  ENV="$WORK/$KIND"
  python -m venv "$ENV"
  # A venv keeps its executables in bin/ on POSIX and Scripts/ on Windows.
  BIN="$ENV/bin"
  [ -d "$ENV/Scripts" ] && BIN="$ENV/Scripts"
  case "$KIND" in
    wheel) "$BIN/pip" install -q "$WHEEL" ;;
    sdist) "$BIN/pip" install -q "$SDIST" ;;
  esac

  "$BIN/python" -c "import aidatasetkit" || fail "$KIND: import"
  "$BIN/aidatasetkit" --version || fail "$KIND: --version"
  "$BIN/aidatasetkit" audit --help >/dev/null || fail "$KIND: audit --help"

  # matplotlib is optional and must not be pulled in by a core install.
  "$BIN/python" - <<'PY' || exit 1
import importlib.util, sys
import aidatasetkit  # noqa: F401
if "matplotlib" in sys.modules:
    raise SystemExit("matplotlib was imported by a core install")
PY

  say "Auditing a synthetic dataset with the installed $KIND"
  "$BIN/python" - "$WORK/$KIND.csv" <<'PY'
import sys
import numpy as np, pandas as pd
i = np.arange(400); churn = (i % 5 == 0).astype(int)
age = (21 + i % 53).astype(float); age[i % 11 == 0] = np.nan
pd.DataFrame({
    "CustomerID": [f"C{v:06d}" for v in i],
    "Age": age,
    "Charges": (20.0 + (i % 67) * 1.3).round(2),
    "City": [["Riyadh", "Jeddah", "Dammam"][v % 3] for v in i],
    "Churn": churn,
    "Churn_Copy": churn,
}).to_csv(sys.argv[1], index=False)
PY

  set +e
  "$BIN/aidatasetkit" audit "$WORK/$KIND.csv" --target Churn \
      --task classification --output "$WORK/out-$KIND" >/dev/null
  CODE=$?
  set -e
  [ "$CODE" -eq 3 ] || fail "$KIND: expected exit 3 (blocked), got $CODE"

  # Through the published reader, from the installed package: CURRENT, the
  # manifest, and every file's size and digest, or a refusal.
  "$BIN/python" - "$WORK/out-$KIND" <<'PY' || exit 1
import sys
from aidatasetkit.evidence import ARTIFACT_SCHEMA_VERSION, read_current
run = read_current(sys.argv[1])
assert sorted(run.contents) == ["audit.json", "lineage.json", "report.html"], sorted(run.contents)
assert all(run.contents.values()), "an artifact is empty"
a = run.json("audit.json")
assert a["schema_version"] == ARTIFACT_SCHEMA_VERSION == "1.3", a["schema_version"]
assert a["verdict"] == "blocked", a["verdict"]
assert a["columns"] and a["findings"], "artifact is empty"
PY
  printf '  %s: install, CLI, audit, artifacts, exit code -- all good\n' "$KIND"
done

say "Release smoke test passed"
echo "Nothing was published. dist/ holds the artifacts."
