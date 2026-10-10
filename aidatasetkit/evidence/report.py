"""The human rendering of an audit artifact.

One rule governs this module: it reads the canonical mapping and nothing else.
:func:`render_report` takes the same dictionary that becomes ``audit.json``, so
the page cannot say something the JSON does not, and a discrepancy between the
two would have to be a bug in this file rather than a divergence between two
independent implementations of the same idea.

No template engine and no framework. A static page with inline CSS has no
dependency to install, no server to run, and no version to keep compatible; the
document is small and its structure is fixed, so a template language would add a
dependency to save very little.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["render_report"]

#: Section order, chosen to answer a reader's questions in the order they occur:
#: what was looked at, what needs attention, what would be done to the data, what
#: happens to each column, and whether any of it can be reproduced.
_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 2rem 1.25rem 4rem; font: 15px/1.6 -apple-system,
  BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: #fbfbfc; color: #16181d; }
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
h2 { font-size: 1.05rem; margin: 2.5rem 0 .75rem; padding-bottom: .4rem;
  border-bottom: 1px solid #e3e5ea; letter-spacing: .01em; }
.sub { color: #61656e; margin: 0 0 1.5rem; }
.verdict { display: inline-block; padding: .35rem .8rem; border-radius: 999px;
  font-weight: 600; font-size: .85rem; letter-spacing: .02em; }
.ready { background: #e6f4ea; color: #16643b; }
.ready_with_warnings { background: #fdf3d8; color: #7a5b00; }
.review_required { background: #fde8d7; color: #8a3d00; }
.blocked { background: #fbe0e0; color: #8a1c1c; }
table { width: 100%; border-collapse: collapse; margin: .5rem 0 1rem; font-size: .88rem; }
th, td { text-align: left; padding: .5rem .6rem; border-bottom: 1px solid #eceef2;
  vertical-align: top; }
th { font-weight: 600; color: #4a4e57; font-size: .78rem; text-transform: uppercase;
  letter-spacing: .04em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85em; }
.sev { font-weight: 600; font-size: .75rem; letter-spacing: .03em; }
.sev-error { color: #a11; } .sev-warning { color: #8a5b00; } .sev-info { color: #4a4e57; }
ul { margin: .4rem 0 1rem; padding-left: 1.2rem; }
li { margin: .2rem 0; }
.cards { display: flex; flex-wrap: wrap; gap: .75rem; margin: 1rem 0 0; }
.card { flex: 1 1 8rem; background: #fff; border: 1px solid #e3e5ea; border-radius: 8px;
  padding: .7rem .85rem; }
.card .k { font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; color: #61656e; }
.card .v { font-size: 1.15rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.muted { color: #61656e; }
.arrow { color: #8b9098; padding: 0 .3rem; }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e3e5ea;
  color: #61656e; font-size: .82rem; }
@media (prefers-color-scheme: dark) {
  body { background: #14161a; color: #e6e8ec; }
  h2 { border-color: #2a2e36; } .sub, .muted, .card .k, th { color: #9aa0aa; }
  td, th { border-color: #23262d; } .card { background: #191c21; border-color: #2a2e36; }
  .ready { background: #113724; color: #7fd8a4; }
  .ready_with_warnings { background: #3a2f0d; color: #e6c65c; }
  .review_required { background: #402512; color: #f0a563; }
  .blocked { background: #3d1717; color: #f08a8a; }
  .sev-error { color: #f08a8a; } .sev-warning { color: #e6c65c; } .sev-info { color: #9aa0aa; }
}
"""


def render_report(artifact: Mapping[str, Any]) -> str:
    """Return a standalone HTML page for a canonical audit mapping.

    Args:
        artifact: The mapping produced by :meth:`AuditArtifact.to_dict`. Passing
            the artifact object itself is a mistake this signature makes obvious:
            the renderer must read what was written, not what is in memory.

    Returns:
        A complete HTML document with no external references.
    """
    dataset = artifact.get("dataset") or {}
    provenance = artifact.get("provenance") or {}
    verdict = str(artifact.get("verdict", "unknown"))
    name = dataset.get("name") or "dataset"

    sections = [
        _summary(artifact, dataset, verdict, name),
        _identity(artifact, dataset),
        _ingestion(artifact),
        _columns(artifact),
        _findings(artifact),
        _decisions(artifact),
        _lineage(artifact),
        _model(artifact),
        _limitations(artifact),
        _provenance(provenance, artifact),
    ]
    body = "\n".join(section for section in sections if section)
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>AIDatasetKit audit — {_e(name)}</title>\n"
        f"<style>{_STYLE}</style>\n</head>\n<body>\n<main>\n{body}\n</main>\n</body>\n</html>\n"
    )


# ---------------------------------------------------------------------- #
# Sections
# ---------------------------------------------------------------------- #


def _summary(artifact, dataset, verdict, name) -> str:
    counts = _severity_counts(artifact.get("findings") or ())
    target = artifact.get("target") or {}
    target_name = (target.get("name") or {}).get("name")
    cards = [
        ("rows", f"{dataset.get('row_count', 0):,}"),
        ("columns", f"{dataset.get('column_count', 0):,}"),
        ("errors", counts["error"]),
        ("warnings", counts["warning"]),
        ("info", counts["info"]),
        ("review items", len(_review_items(artifact))),
    ]
    task = target.get("task_type") or "not specified"
    return (
        f"<h1>AIDatasetKit audit</h1>\n"
        f"<p class=\"sub\">{_e(name)} <span class=\"arrow\">·</span> "
        f"task: {_e(task)}"
        + (
            f" <span class=\"arrow\">·</span> target: <code>{_e(target_name)}</code>"
            if target_name
            else ""
        )
        + f" <span class=\"arrow\">·</span> stage: {_e(artifact.get('stage', ''))}</p>\n"
        f"<p><span class=\"verdict {_e(verdict)}\">"
        f"{_e(verdict.replace('_', ' ').upper())}</span></p>\n"
        + _reasons(artifact)
        + "<div class=\"cards\">"
        + "".join(
            f"<div class=\"card\"><div class=\"k\">{_e(k)}</div>"
            f"<div class=\"v\">{_e(v)}</div></div>"
            for k, v in cards
        )
        + "</div>\n"
    )


def _reasons(artifact) -> str:
    reasons = artifact.get("verdict_reasons") or ()
    if not reasons:
        return ""
    return "<ul>" + "".join(f"<li>{_e(reason)}</li>" for reason in reasons) + "</ul>\n"


_DELIMITER_NAMES = {",": "comma (,)", ";": "semicolon (;)", "\t": "tab", "|": "pipe (|)"}


def _ingestion(artifact) -> str:
    """How the table was read. Absent when the artifact has no ingestion record."""
    record = artifact.get("ingestion")
    if not record:
        return ""
    delimiter = record.get("delimiter")
    rows = [
        ("source", record.get("source_kind")),
        ("format", record.get("format")),
        ("encoding", record.get("encoding")),
        (
            "delimiter",
            None
            if delimiter is None
            else f"{_DELIMITER_NAMES.get(delimiter, delimiter)}, {record.get('delimiter_source')}",
        ),
        ("header row", None if record.get("header") is None else ("yes" if record["header"] else "no")),
        ("table", (artifact.get("source_selector") or {}).get("name")),
        ("rows × columns", f"{record.get('row_count', 0):,} × {record.get('column_count', 0):,}"),
        (
            "memory",
            "not applicable"
            if record.get("memory_bytes") is None
            else f"{record['memory_bytes']:,} bytes",
        ),
    ]
    body = "".join(
        f"<tr><th>{_e(label)}</th><td>{_e(value)}</td></tr>" for label, value in rows if value is not None
    )
    notes = "".join(f"<li>{_e(note)}</li>" for note in record.get("warnings") or ())
    return f"<h2>Input</h2>\n<table>{body}</table>\n" + (f"<ul>{notes}</ul>\n" if notes else "")


def _identity(artifact, dataset) -> str:
    rows = [
        ("dataset fingerprint", dataset.get("fingerprint")),
        ("schema fingerprint", dataset.get("schema_fingerprint")),
        ("config fingerprint", (artifact.get("config") or {}).get("fingerprint")),
        ("plan fingerprint", artifact.get("plan_fingerprint")),
        (
            "evidence fingerprint",
            (artifact.get("provenance") or {}).get("semantic_fingerprint"),
        ),
        ("algorithm", dataset.get("algorithm")),
        ("duplicate rows", dataset.get("duplicate_row_count")),
        ("missing cells", dataset.get("total_missing_count")),
    ]
    body = "".join(
        f"<tr><th>{_e(label)}</th><td class=\"mono\">{_e(value)}</td></tr>"
        for label, value in rows
        if value is not None
    )
    return f"<h2>Dataset identity</h2>\n<table>{body}</table>\n"


def _columns(artifact) -> str:
    """What profiling measured, per column. No values, by design."""
    columns = artifact.get("columns") or ()
    if not columns:
        return ""
    rows = "".join(
        "<tr>"
        f"<td><code>{_e((c.get('name') or {}).get('name'))}</code></td>"
        f"<td>{_e(c.get('detected_kind'))}</td>"
        f"<td class=\"mono\">{_e(c.get('pandas_dtype'))}</td>"
        f"<td class=\"num\">{_e(c.get('missing_count'))}</td>"
        f"<td class=\"num\">{_ratio(c.get('missing_ratio'))}</td>"
        f"<td class=\"num\">{_e(c.get('unique_count'))}</td>"
        f"<td>{_e(', '.join(_flags(c)) or '—')}</td>"
        "</tr>"
        for c in columns
    )
    return (
        "<h2>Columns</h2>\n<table>"
        "<tr><th>column</th><th>kind</th><th>dtype</th><th>missing</th>"
        "<th>missing %</th><th>distinct</th><th>flags</th></tr>"
        f"{rows}</table>\n"
        "<p class=\"muted\">Counts and ratios only. The most frequent value of each "
        "column is recorded as a digest, never in plain text.</p>\n"
    )


def _flags(column: Mapping[str, Any]) -> list[str]:
    names = {
        "is_constant": "constant",
        "is_near_constant": "near-constant",
        "is_high_cardinality": "high cardinality",
        "is_id_like": "identifier-like",
    }
    return [label for key, label in names.items() if column.get(key)]


def _ratio(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _findings(artifact) -> str:
    findings = artifact.get("findings") or ()
    if not findings:
        return "<h2>Safety findings</h2>\n<p class=\"muted\">No findings were recorded.</p>\n"
    order = {"error": 0, "warning": 1, "info": 2}
    ranked = sorted(findings, key=lambda f: (order.get(f.get("severity"), 9), f.get("code", "")))
    rows = "".join(
        "<tr>"
        f"<td><span class=\"sev sev-{_e(f.get('severity'))}\">"
        f"{_e(str(f.get('severity','')).upper())}</span></td>"
        f"<td><code>{_e(f.get('code'))}</code></td>"
        f"<td>{_e((f.get('column') or {}).get('name') or '—')}</td>"
        f"<td>{_e(f.get('message'))}"
        + (
            f"<br><span class=\"muted\">{_e(f.get('recommendation'))}</span>"
            if f.get("recommendation")
            else ""
        )
        + "</td>"
        f"<td>{'yes' if f.get('requires_review') else ''}</td>"
        "</tr>"
        for f in ranked
    )
    return (
        "<h2>Safety findings</h2>\n<table>"
        "<tr><th>severity</th><th>code</th><th>column</th><th>finding</th><th>review</th></tr>"
        f"{rows}</table>\n"
    )


def _decisions(artifact) -> str:
    decisions = artifact.get("decisions") or ()
    if not decisions:
        return (
            "<h2>Preprocessing plan</h2>\n<p class=\"muted\">No preprocessing was "
            "planned for this run, so no decisions were recorded.</p>\n"
        )
    rows = "".join(
        "<tr>"
        f"<td><code>{_e((d.get('feature') or {}).get('name'))}</code></td>"
        f"<td>{_e(d.get('role'))}</td>"
        f"<td>{_e(d.get('action'))}</td>"
        f"<td>{_e(' → '.join(d.get('steps') or []) or '—')}</td>"
        f"<td>{_e(d.get('fit_scope'))}</td>"
        f"<td>{_e(d.get('reason'))}"
        + (
            f"<br><span class=\"muted\">{_e(d.get('model_requirement'))}</span>"
            if d.get("model_requirement")
            else ""
        )
        + "</td></tr>"
        for d in decisions
    )
    return (
        "<h2>Preprocessing plan</h2>\n<table>"
        "<tr><th>feature</th><th>role</th><th>action</th><th>steps</th>"
        "<th>fit scope</th><th>reason</th></tr>"
        f"{rows}</table>\n"
    )


def _lineage(artifact) -> str:
    lineage = artifact.get("lineage") or ()
    if not lineage:
        return ""
    rows = "".join(
        "<tr>"
        f"<td><code>{_e((entry.get('source') or {}).get('name'))}</code></td>"
        f"<td>{_e(entry.get('action'))}</td>"
        f"<td>{_e(' → '.join(entry.get('steps') or []) or '—')}</td>"
        f"<td>{_e(', '.join(entry.get('outputs') or []) or '—')}</td>"
        f"<td>{'observed' if entry.get('observed') else 'planned'}</td>"
        "</tr>"
        for entry in lineage
    )
    return (
        "<h2>Feature lineage</h2>\n<table>"
        "<tr><th>source</th><th>action</th><th>steps</th><th>outputs</th><th>basis</th></tr>"
        f"{rows}</table>\n"
    )


def _model(artifact) -> str:
    model = artifact.get("model")
    if not model:
        return (
            "<h2>Model compatibility</h2>\n<p class=\"muted\">No model context was "
            "supplied, so preprocessing evidence is model-independent.</p>\n"
        )
    capabilities = model.get("capabilities") or {}
    rows = "".join(
        f"<tr><th>{_e(key)}</th><td>{_e(value)}</td></tr>"
        for key, value in sorted(capabilities.items())
    )
    return (
        f"<h2>Model compatibility</h2>\n<p><code>{_e(model.get('canonical_name'))}</code> "
        f"<span class=\"muted\">({_e(model.get('backend'))}, profile "
        f"<code>{_e(model.get('preprocessing_profile'))}</code>)</span></p>\n"
        f"<table>{rows}</table>\n"
        "<p class=\"muted\">No estimator was constructed and nothing was trained. "
        "These are declared capabilities, recorded because they shaped the plan above.</p>\n"
    )


def _limitations(artifact) -> str:
    items = list(artifact.get("known_limitations") or ())
    warnings = list(artifact.get("warnings") or ())
    if not items and not warnings:
        return ""
    parts = ["<h2>Known limitations</h2>\n"]
    if warnings:
        parts.append("<ul>" + "".join(f"<li>{_e(w)}</li>" for w in warnings) + "</ul>\n")
    parts.append("<ul>" + "".join(f"<li>{_e(item)}</li>" for item in items) + "</ul>\n")
    return "".join(parts)


def _provenance(provenance, artifact) -> str:
    environment = provenance.get("environment") or {}
    rows = "".join(
        f"<tr><th>{_e(key)}</th><td class=\"mono\">{_e(value)}</td></tr>"
        for key, value in sorted(environment.items())
    )
    return (
        "<h2>Provenance</h2>\n"
        f"<table><tr><th>created at</th><td class=\"mono\">{_e(provenance.get('created_at'))}</td></tr>"
        f"{rows}</table>\n"
        f"<footer>Artifact schema {_e(artifact.get('schema_version'))}. "
        "This page is rendered from the same canonical record as "
        "<code>audit.json</code>; the JSON is the artifact.</footer>\n"
    )


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _severity_counts(findings: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"error": 0, "warning": 0, "info": 0}
    for finding in findings:
        severity = str(finding.get("severity", ""))
        if severity in counts:
            counts[severity] += 1
    return counts


def _review_items(artifact: Mapping[str, Any]) -> list[str]:
    findings = artifact.get("findings") or ()
    decisions = artifact.get("decisions") or ()
    return [item for item in (*findings, *decisions) if item.get("requires_review")]


def _e(value: Any) -> str:
    """Escape a value for HTML. Everything reaching the page goes through here."""
    return html.escape("" if value is None else str(value), quote=True)
