# Feature lineage

`lineage.json` answers one question for every column you started with: *what
happened to it?*

It is only populated when a preprocessing plan exists, which means when you pass
`--model` together with `--target`. Without a model there is nothing to plan for,
so the file is written with an empty `features` list and a `stage` of
`inspected` — an honest "nothing was decided yet" rather than a missing file.

```json
{
  "schema_version": "1.1",
  "dataset_fingerprint": "8135be14ca667643...",
  "stage": "prepared",
  "features": [
    {
      "source": { "name": "City", "label_type": "str" },
      "source_kind": "nominal",
      "action": "include",
      "steps": ["most_frequent_imputation", "onehot_encoding"],
      "outputs": ["City_Dammam", "City_Jeddah", "City_Riyadh"],
      "fit_scope": "training_only",
      "reason": "Nominal categorical feature with 3 levels.",
      "observed": true
    },
    {
      "source": { "name": "CustomerID", "label_type": "str" },
      "source_kind": "id_like",
      "action": "review",
      "steps": [],
      "outputs": [],
      "fit_scope": "not_applicable",
      "reason": "100.0% of values are distinct ... held back for review.",
      "observed": false
    }
  ]
}
```

## Where it comes from

The preprocessing layer is the authority. Lineage is **not** re-derived by
parsing output names, and the reason is concrete: `city_Riyadh` looks like it
decomposes into a column and a level, and a single category containing an
underscore makes that guess wrong. The preprocessing layer walks its own fitted
steps and reports the mapping; evidence serialises what it reports.

## `observed` versus intended

- `"observed": true` — a preprocessor was fitted and these are the columns it
  actually produced.
- `"observed": false` — only a plan exists. `steps` is what *would* happen;
  `outputs` is empty because nothing has been built.

A column that was excluded or held for review is always `false` with empty
outputs, and that is the honest answer rather than a gap.

## Columns that produce nothing

They still appear. An audit that silently omitted the identifier it held back
would be indistinguishable from one that never saw it, and "this column was
deliberately not used, for this reason" is often the most important line in the
file.

## Using it

```python
import json

from aidatasetkit.evidence import read_current

lineage = read_current("aidk-audit").json("lineage.json")
for feature in lineage["features"]:
    if feature["action"] != "include":
        print(f"{feature['source']['name']}: {feature['reason']}")
```

To find which original column a model feature came from, search the `outputs`
lists — never split the name on an underscore.
