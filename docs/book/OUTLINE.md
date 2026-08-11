# Book outline (plan only)

A plan, not a draft. Nothing here is written yet, and the structure will change
once alpha feedback shows which parts people actually need.

The through-line: **most tabular ML failures are data failures, and most of them
are visible before training.** The book follows one dataset from raw file to
audited, model-ready evidence, and each part answers a question the previous part
raised.

## Part I — Why data safety matters

1. The failure nobody sees — a model that scores 0.99 and is worthless
2. Leakage, and why cross-validation does not catch it
3. Silent modification: the cost of tools that "just fix it"
4. What an audit is, and what it can never be

## Part II — Knowing what you have

5. Statistics that refuse to lie (degenerate cases, and why `NaN` is not an answer)
6. Profiling a column: kinds, cardinality, gaps, and dominance
7. Detecting the task from the target
8. Quality findings: severity, evidence, and the discipline of `requires_review`

## Part III — Seeing it

9. What should be visualised, independently of how
10. Ranking charts without ranking data
11. Rendering as a replaceable detail

## Part IV — Preparing it safely

12. Capability-driven preprocessing: why the model decides, not its name
13. The plan as a promise
14. Fit scope, train-only statistics, and what leaks
15. Encoding without guessing: ordinals, mappings, and unknown categories
16. Everything that can go wrong with a column of text that looks numeric

## Part V — Choosing a model

17. Capabilities as verified facts, not documentation
18. The nine classifiers, and the three answers that surprised us
19. When two capabilities are individually true and jointly false

## Part VI — Evidence and provenance

20. The artifact: what happened, and why
21. Identity: fingerprinting data you must not store
22. Privacy in an audit trail
23. Verdicts, policy, and failing a build

## Part VII — A complete case study

24. A churn dataset, end to end
25. What the audit caught, and what it could not
26. Living with the artifact: diffs, review, and CI

## Part VIII — Architecture and contributing

27. Layers, boundaries, and why the import graph is a test
28. Adding a check, a model, a renderer
29. Testing philosophy: contracts executed, not asserted
