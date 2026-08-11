# Decision record — project license

| | |
|---|---|
| **Decision** | Apache License 2.0 (SPDX: `Apache-2.0`) |
| **Status** | Accepted |
| **Date** | 2026-08-11 |
| **Decided by** | Repository owner |
| **Scope** | The AIDatasetKit open-source repository and published package |

## The decision

AIDatasetKit is licensed under the Apache License, Version 2.0. The repository
root carries the official, unmodified licence text in `LICENSE`, and
`pyproject.toml` declares it as:

```toml
license = "Apache-2.0"
license-files = ["LICENSE"]
```

This closes the only blocker that stood between the `0.1.0a1` preparation work
and an owner-authorised release.

## What was decided against

**MIT.** `pyproject.toml` declared `license = { text = "MIT" }` from the first
commit (`559d085`) through `bd5e254`, and no `LICENSE` file ever accompanied it.
That declaration was left untouched by every stage of automated work, because
choosing a licence — and removing one — are decisions for the owner. The owner
chose Apache-2.0 instead, and the MIT declaration was removed as part of applying
that choice.

**Declaring nothing.** A package with no declared licence is "all rights
reserved", which is a clear position but not the one wanted for an open-source
release.

## Why the spelling matters

The old declaration used a form setuptools has deprecated:

> `SetuptoolsDeprecationWarning: project.license as a TOML table is deprecated.`
> Please use a simple string containing a SPDX expression. **By 2027-Feb-18 you
> need to update your project.**

The new declaration uses the PEP 639 SPDX string, so the metadata is correct
today and will still build after that date. `python -m build` no longer emits
that warning.

One consequence is worth recording, because the first attempt at this change hit
it: with an SPDX expression, the old `License :: OSI Approved :: ...` trove
classifier is no longer optional-but-harmless — setuptools **refuses to build**
while both are present:

> `InvalidConfigError: License classifiers have been superseded by license
> expressions (see PEP 639). Please remove: License :: OSI Approved :: Apache
> Software License`

So the classifier list carries no licence entry. The expression is the single
declaration.

## NOTICE

**Not required by the current repository contents.** Apache-2.0 requires a
`NOTICE` file only when one is inherited from redistributed third-party material.
This repository vendors nothing: numpy, pandas, scipy, and scikit-learn are
*declared* dependencies installed from their own distributions, not copied into
this package. No file in `aidatasetkit/` carries a third-party copyright or SPDX
header.

If third-party code is ever vendored, that changes and a `NOTICE` becomes
required.

## Copyright holder

The `LICENSE` appendix retains the standard placeholders
(`Copyright [yyyy] [name of copyright owner]`). No copyright holder, legal
entity, or year was invented, because none is established anywhere in the
repository's metadata. Filling those in is a separate owner decision, and
Apache-2.0 does not require it for the licence to apply.

## History

This document previously recorded the decision as *pending*, and set out the
exact change each option would need. That content is superseded by the decision
above and is preserved in git history at `bd5e254` and earlier.
