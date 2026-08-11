# Safety model

## What the verdict means

Every audit ends in one of four words. Each says what was *found*, not what is
true of your data in general.

| Verdict | Meaning |
|---|---|
| `ready` | No finding above informational severity, nothing awaiting a person. |
| `ready_with_warnings` | Warnings were raised. Nothing blocks proceeding. |
| `review_required` | Something needs a human decision. |
| `blocked` | An error-severity finding, or the analysis could not complete. |

The exit code is not a property of the verdict — it is the answer to *"did this
meet the bar you set with `--fail-on`"*. Under the default (`--fail-on review`),
`ready` and `ready_with_warnings` exit `0`, `review_required` exits `2`, and
`blocked` exits `3`. Under `--fail-on warning`, `ready_with_warnings` exits `2`
as well. Under `--fail-on never` everything exits `0`. See
[getting-started.md](getting-started.md#exit-codes).

`ready` claims only that **these checks found no blocker**. That is a much
narrower statement than "safe", and the difference is the point.

## The policy

Applied in order, and the reasons are recorded in the artifact:

1. The analysis could not complete → `blocked`. Nothing else can be concluded
   from a run that stopped.
2. Any `error` finding → `blocked`. The quality layer reserves that severity for
   a condition it considers disqualifying; re-litigating it here would put two
   disagreeing opinions in one library.
3. Anything awaiting a person → `review_required`. That is a finding flagged
   `requires_review`, or a feature the planner held back rather than deciding
   for you.
4. Any `warning` → `ready_with_warnings`.
5. Otherwise `ready`.

The policy is deterministic and reads only from severities the quality layer
already assigned. It never re-examines your data.

## Language we do not use

No artifact, report, or exit code will ever tell you your data is:

> safe · compliant · certified · leakage-free · production-ready

Those are conclusions about a world this library cannot see. What it says
instead:

> no known blocker found · review required · possible leakage ·
> verified train-only fit · unsupported condition

If you need a compliance claim, an audit artifact is evidence you can bring to
that conversation. It is not the conclusion of it.

## Detect, explain, recommend

The library never silently modifies your data. Concretely:

- A column that looks like an identifier is **held back and reported**, not
  dropped. The heuristic can be wrong, and a dropped column is invisible.
- An ordinal order is **never inferred** from the alphabet. `high`, `low`,
  `medium` sorts wrong, and a model trained on that ordering is wrong in a way
  no metric reveals.
- Numbers stored as text are **never quietly parsed**. Values that fail to parse
  would become missing, and a column of sentinels would turn into a column of
  imputed medians with nobody deciding that.
- Class imbalance is **never automatically rebalanced**. No resampling, no
  automatic `class_weight`.
- No model is ever selected, ranked, or recommended.

## Leakage: what is and is not detectable

**Detectable, and reported as an error:** a feature exactly equal to the target.
That is certain, not heuristic.

**Detectable, and reported for review:** a feature with an association to the
target above the configured threshold, or a deterministic mapping onto it. Both
are heuristics. A legitimately strong predictor looks the same as a leak from the
numbers alone.

**Not detectable:** a feature that encodes the outcome for reasons the numbers do
not show — a value assigned after the event, a join that pulled in future
information, an identifier that correlates with collection order. Statistics
cannot see any of these. **This is the most important limitation in the library**
and it travels inside every artifact for that reason.

## Fit scope and leakage

Every learned transformation records where its statistics came from. A median, a
most-frequent category, a scaler's centre, a one-hot vocabulary, and the
categorical sentinel are all `training_only` — read off the rows the
preprocessor was fitted on, never from data it will later transform.

This is the difference between a defensible pipeline and one whose validation
score means nothing, and the artifact states it per feature rather than as a
blanket assurance.

## What an audit does not prove

An audit records what preprocessing **would** do. It does not prove a model was
trained on the data it describes, that the plan was executed, or that the model
in production matches the one that was planned. Connecting an artifact to a
trained model is future work.
