"""Check local Markdown links introduced for the comparative research study."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "docs" / "getting-started.md",
    ROOT / "docs" / "research" / "README.md",
    ROOT / "docs" / "research" / "AIDatasetKit_Research_Protocol.md",
    ROOT / "examples" / "research" / "README.md",
)
pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
missing: list[tuple[Path, str, Path]] = []
for document in DOCUMENTS:
    text = document.read_text(encoding="utf-8")
    for raw in pattern.findall(text):
        target = raw.split("#", 1)[0].strip()
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        resolved = (document.parent / target).resolve()
        if not resolved.exists():
            missing.append((document, raw, resolved))
if missing:
    rendered = "\n".join(f"{doc}: {raw} -> {path}" for doc, raw, path in missing)
    raise AssertionError(f"Broken local Markdown links:\n{rendered}")
print(f"Validated local Markdown links in {len(DOCUMENTS)} research-related documents.")
