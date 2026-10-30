#!/usr/bin/env python3
"""Generate docs/content/generated/spec.md from esm-jacobian-spec.md.

The draft specification lives at the repository root, next to the code that
implements it; the site page is built from it rather than copied, so the two
cannot drift. Output is not checked in.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "esm-jacobian-spec.md"
OUT = ROOT / "docs" / "content" / "generated" / "spec.md"

FRONT = """---
title: "Specification (draft)"
description: "The jacobians block format, per-operator derivative semantics and the factorization-info interface."
weight: 20
---

"""

text = SRC.read_text(encoding="utf-8")
lines = text.splitlines()
if lines and lines[0].startswith("# "):  # the page title comes from front matter
    lines = lines[1:]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(FRONT + "\n".join(lines).lstrip("\n") + "\n", encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)}")
