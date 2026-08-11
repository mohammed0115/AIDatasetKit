# Contributing

The project is in public alpha. The most useful contribution right now is
feedback on the audit artifact: what is missing, what is unclear, what you would
want to fail a build on.

## Setup

```bash
git clone <this repository>
cd aidatasetkit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Tests

```bash
pytest -q                                          # everything
pytest -q -W error                                 # warnings are failures
pytest tests/integration/test_architecture_boundaries.py -q
PYTHONHASHSEED=1 pytest tests/unit -q              # determinism
```

Three rules the suite enforces and which a change must not weaken:

- **`pytest -W error` must pass.** Never add a global warning filter to make it.
  If a fixture provokes a meaningless warning, fix the fixture.
- **Determinism.** Nothing may depend on hash ordering, wall-clock time, or an
  unseeded generator.
- **Architecture boundaries.** A package may import from its own layer or below,
  never above.

## Architecture

```
core → statistics → profiling → preprocessing / models / visualization → evidence → cli
```

`tests/integration/test_architecture_boundaries.py` parses every import in the
package and enforces this. Adding a package means adding it to `LAYERS` there —
a deliberate decision rather than an accident.

Two rules carry most of the weight:

- **Models declare what they need; they never inspect data to find out.** The
  models layer may not import profiling, statistics, or preprocessing.
- **Evidence aggregates and nothing depends on it.** No layer may import
  `evidence`, and evidence recomputes nothing.

## Adding a quality check

1. Write a function in `aidatasetkit/profiling/checks.py` taking
   `(frame, context)` and returning `list[QualityIssue]`.
2. Register it in the inspector's check list.
3. Choose the severity honestly. `ERROR` blocks a run; reserve it for something
   certain, not heuristic. Anything inferred should set `requires_review`.
4. Put the numbers in `details` — a count, a ratio, the threshold crossed. A
   finding reduced to prose cannot be acted on programmatically.
5. **Text in `details` is redacted by default.** You do not have to do anything
   to keep a data value out of an audit artifact — evidence hashes every string
   whose key is not on the short vocabulary list `_SAFE_TEXT_DETAIL_KEYS` in
   `aidatasetkit/evidence/builder.py`. If your check reports a *name* rather
   than data (a column, a detected kind, the name of a signal that fired), add
   that key to the list, and extend the test that pins the full set. Numbers and
   booleans are never touched.
6. Add tests for both the positive case and a case that must *not* fire.

## Adding a model strategy

1. Subclass `ModelStrategy`, declare `name` and `capabilities`, implement
   `build`.
2. **Verify every capability by running the estimator.** Do not fill them from
   memory — three of the nine current answers contradict what most engineers
   would assume. Record what you measured in the docstring.
3. Remember the capabilities describe **one joint contract**, not three
   independent facts. scikit-learn's tree family accepts sparse input, accepts
   NaN, and rejects sparse-carrying-NaN.
4. Register with `@register_model(aliases=(...))`. Choose aliases that a future
   regression twin can share.
5. The contract matrix in `tests/unit/test_model_contracts.py` is parametrised
   over the live registry, so your model is tested the moment it is registered.
   Pin its shipped defaults in `TestProductionDefaultsAreExercisedAndPinned`.

## Regression tests

Every confirmed defect gets a test that fails without the fix. Not a test that
exercises the area — a test that reproduces the specific failure. Several tests
in the suite carry a docstring explaining what went wrong and why the assertion
is shaped the way it is; that is the house style.

## Style

- Docstrings explain *why*, not what the code plainly says.
- Error messages name the column, the value, and a way out.
- No `print` in library code. The CLI prints; the library logs.
