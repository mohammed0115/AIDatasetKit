# License decision — required before release

**This is the only blocker holding `0.1.0a1`.**

## Current state

`pyproject.toml` has declared `license = { text = "MIT" }` since the first
commit (`559d085`). There is **no `LICENSE` file** in the repository.

That combination is not a neutral state. A package index will display "MIT" from
the metadata while the repository contains no licence text, which is worse than
either choosing one or declaring none: a user reading the metadata believes they
have permission that the repository does not grant in writing.

Neither the declaration nor its absence was changed by any automated work.
Choosing a licence and removing one are both legal decisions, and both belong to
the owner.

## Option A — MIT

Consistent with what the metadata has always said. Permissive: anyone may use,
modify, and redistribute, including commercially, provided the notice travels
with the code. No patent grant.

**Changes required:**

1. Add `LICENSE` containing the official MIT text, with the copyright line
   completed:

   ```
   MIT License

   Copyright (c) 2026 <COPYRIGHT HOLDER>

   Permission is hereby granted, free of charge, to any person obtaining a copy
   ... (the unmodified OSI text)
   ```

2. `pyproject.toml` — **no change needed**. `license = { text = "MIT" }` already
   matches.

3. Optionally add the trove classifier:

   ```toml
   classifiers = [
       ...,
       "License :: OSI Approved :: MIT License",
   ]
   ```

4. `MANIFEST.in` — no change; `LICENSE` is included in both distributions
   automatically by setuptools.

## Option B — Apache-2.0

Also permissive, and additionally grants patent rights and terminates them for a
party that starts patent litigation. Usually preferred where contributors or
users may hold patents.

**Changes required:**

1. Add `LICENSE` containing the full, unmodified Apache License 2.0 text.

2. `pyproject.toml` — change the declaration:

   ```diff
   -license = { text = "MIT" }
   +license = { text = "Apache-2.0" }
   ```

3. Optionally add the trove classifier:

   ```toml
   "License :: OSI Approved :: Apache Software License",
   ```

4. A `NOTICE` file is required **only if** the project distributes third-party
   material that carries its own NOTICE. It does not today: the runtime
   dependencies (numpy, pandas, scipy, scikit-learn) are declared, not vendored.
   So no `NOTICE` is needed unless that changes.

5. Apache-2.0 conventionally adds a short header to each source file. That is a
   separate mechanical change across roughly forty files and is not included in
   the diff above.

## Option C — something else, or nothing yet

If the owner wants a different licence, or wants to keep the code private for
now, the honest interim step is to **remove the MIT declaration** rather than
leave metadata asserting a grant that no file makes:

```diff
-license = { text = "MIT" }
```

A package with no declared licence is legally "all rights reserved", which is a
clear position. A package declaring MIT with no licence text is an unclear one.

## Recommendation

None. This is the one decision in this project that an automated process should
not make, and the reason it is written down rather than acted on.
