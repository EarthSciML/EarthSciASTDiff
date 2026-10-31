#!/usr/bin/env python3
"""Resolve the EarthSciAST releases each binding is tested against.

Each binding declares the EarthSciAST versions it supports in its manifest:
the Julia `[compat]` entry, the Cargo requirement and the PEP 508 dependency.
This script reads those three ranges, lists the releases on the registry each
binding installs from (General, crates.io, PyPI; yanked and pre-releases
excluded), and prints, per binding, the LOWEST and the LATEST release inside
the range: the two ends conformance.yml tests, so the declared floor is a
tested fact rather than a guess, and the latest is what a user installs
today. When the two ends are the same release it is listed once.

A range no release satisfies is an error: it means a manifest asks for an
EarthSciAST that has not been published, and a release made from it would be
uninstallable. Release EarthSciAST first, then raise the floor here.

Usage:
  scripts/earthsci-ast-versions.py                  # print a table
  scripts/earthsci-ast-versions.py --github-output  # also write the matrices
                                                    # to $GITHUB_OUTPUT

Python 3.11+ (tomllib); standard library only.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent

Version = tuple[int, int, int]
# A range is a union of intervals [lo, hi), hi = None for unbounded.
Interval = tuple[Version, "Version | None"]


def parse_version(s: str) -> Version:
    parts = s.strip().split(".")
    if not 1 <= len(parts) <= 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"not a plain release version: {s!r}")
    nums = [int(p) for p in parts] + [0] * (3 - len(parts))
    return (nums[0], nums[1], nums[2])


def show(v: Version) -> str:
    return ".".join(map(str, v))


def caret(s: str) -> Interval:
    """`^x.y.z` as Cargo and Julia both define it: the left-most non-zero
    component of the written version may not change."""
    parts = s.strip().split(".")
    lo = parse_version(s)
    major, minor, patch = lo
    if major > 0 or len(parts) == 1:
        hi = (major + 1, 0, 0)
    elif minor > 0 or len(parts) == 2:
        hi = (0, minor + 1, 0)
    else:
        hi = (0, 0, patch + 1)
    return lo, hi


def tilde(s: str) -> Interval:
    parts = s.strip().split(".")
    lo = parse_version(s)
    hi = (lo[0] + 1, 0, 0) if len(parts) == 1 else (lo[0], lo[1] + 1, 0)
    return lo, hi


def comparator(op: str, v: str) -> Interval:
    """One `op version` comparator as an interval (`!=` is not supported)."""
    x = parse_version(v)
    nxt = (x[0], x[1], x[2] + 1)
    return {
        ">=": (x, None),
        ">": (nxt, None),
        "<": ((0, 0, 0), x),
        "<=": ((0, 0, 0), nxt),
        "=": (x, nxt),
        "==": (x, nxt),
    }[op]


def intersect(a: Interval, b: Interval) -> Interval:
    lo = max(a[0], b[0])
    if a[1] is None or b[1] is None:
        hi = a[1] if b[1] is None else b[1]
    else:
        hi = min(a[1], b[1])
    return lo, hi


CMP = re.compile(r"^(>=|<=|==|=|>|<|\^|~=|~)?\s*([0-9][0-9.]*)$")


def cargo_range(req: str) -> list[Interval]:
    """A Cargo requirement: comma-separated comparators, all of which hold."""
    acc: Interval = ((0, 0, 0), None)
    for part in req.split(","):
        m = CMP.match(part.strip())
        if not m or m.group(1) == "~=":
            raise ValueError(f"unsupported Cargo requirement: {req!r}")
        op, v = m.group(1) or "^", m.group(2)
        iv = caret(v) if op == "^" else tilde(v) if op == "~" else comparator(op, v)
        acc = intersect(acc, iv)
    return [acc]


def pep440_range(spec: str) -> list[Interval]:
    """A PEP 440 specifier set: comma-separated clauses, all of which hold."""
    acc: Interval = ((0, 0, 0), None)
    for part in spec.split(","):
        m = CMP.match(part.strip())
        if not m or m.group(1) in (None, "^", "~", "="):
            raise ValueError(f"unsupported PEP 440 specifier: {spec!r}")
        op, v = m.group(1), m.group(2)
        if op == "~=":
            n = len(v.split("."))
            if n < 2:
                raise ValueError(f"`~=` needs two components: {spec!r}")
            lo = parse_version(v)
            hi = (lo[0] + 1, 0, 0) if n == 2 else (lo[0], lo[1] + 1, 0)
            iv: Interval = (lo, hi)
        else:
            iv = comparator(op, v)
        acc = intersect(acc, iv)
    return [acc]


def julia_range(compat: str) -> list[Interval]:
    """A Julia `[compat]` entry: comma-separated specifiers, any of which
    holds (bare = caret, `~`, `=`, `>=`, and `a - b` hyphen ranges)."""
    out: list[Interval] = []
    for part in compat.split(","):
        p = part.strip()
        if " - " in p:
            a, b = (x.strip() for x in p.split(" - "))
            hi = parse_version(b)
            n = len(b.split("."))
            hi = (
                (hi[0] + 1, 0, 0)
                if n == 1
                else (hi[0], hi[1] + 1, 0)
                if n == 2
                else (hi[0], hi[1], hi[2] + 1)
            )
            out.append((parse_version(a), hi))
            continue
        m = CMP.match(p)
        if not m or m.group(1) not in (None, "^", "~", "=", ">="):
            raise ValueError(f"unsupported Julia compat entry: {compat!r}")
        op, v = m.group(1) or "^", m.group(2)
        out.append(caret(v) if op == "^" else tilde(v) if op == "~" else comparator(op, v))
    return out


def contains(rng: list[Interval], v: Version) -> bool:
    return any(lo <= v and (hi is None or v < hi) for lo, hi in rng)


# ── the manifests ────────────────────────────────────────────────────────────


def declared() -> dict[str, tuple[str, list[Interval]]]:
    jl = tomllib.loads((ROOT / "pkg/EarthSciASTDiff.jl/Project.toml").read_text())
    rs = tomllib.loads((ROOT / "pkg/earthsci-astdiff-rs/Cargo.toml").read_text())
    py = tomllib.loads((ROOT / "pkg/earthsci-astdiff-py/pyproject.toml").read_text())
    jreq = jl["compat"]["EarthSciAST"]
    rreq = rs["dependencies"]["earthsci-ast"]
    if isinstance(rreq, dict):
        rreq = rreq["version"]
    preq = None
    for dep in py["project"]["dependencies"]:
        m = re.match(r"^earthsci-ast\s*(.*)$", dep.strip())
        if m:
            preq = m.group(1).strip()
    if not preq:
        raise SystemExit("pyproject.toml: earthsci-ast has no version bound")
    return {
        "julia": (jreq, julia_range(jreq)),
        "rust": (rreq, cargo_range(rreq)),
        "python": (preq, pep440_range(preq)),
    }


# ── the registries ───────────────────────────────────────────────────────────


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "EarthSciASTDiff-ci"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def plain(versions) -> list[Version]:
    out = []
    for s in versions:
        try:
            out.append(parse_version(s))
        except ValueError:  # pre-release or local version
            pass
    return sorted(out)


def released() -> dict[str, list[Version]]:
    general = tomllib.loads(
        fetch(
            "https://raw.githubusercontent.com/JuliaRegistries/General/master/E/EarthSciAST/Versions.toml"
        ).decode()
    )
    crates = json.loads(fetch("https://crates.io/api/v1/crates/earthsci-ast"))
    pypi = json.loads(fetch("https://pypi.org/pypi/earthsci-ast/json"))
    return {
        "julia": plain(v for v, info in general.items() if not info.get("yanked")),
        "rust": plain(v["num"] for v in crates["versions"] if not v["yanked"]),
        "python": plain(
            v
            for v, files in pypi["releases"].items()
            if files and not all(f["yanked"] for f in files)
        ),
    }


def main() -> None:
    decl = declared()
    rel = released()
    matrices: dict[str, list[dict]] = {}
    failed = False
    rows = []
    for binding, (req, rng) in decl.items():
        ok = [v for v in rel[binding] if contains(rng, v)]
        if not ok:
            print(
                f"error: {binding}: no published EarthSciAST release satisfies {req!r} "
                f"(published: {', '.join(map(show, rel[binding]))})",
                file=sys.stderr,
            )
            failed = True
            continue
        lo, hi = ok[0], ok[-1]
        legs = (
            [{"label": f"{show(lo)}, lowest = latest", "version": show(lo)}]
            if lo == hi
            else [
                {"label": f"{show(lo)}, lowest", "version": show(lo)},
                {"label": f"{show(hi)}, latest", "version": show(hi)},
            ]
        )
        matrices[binding] = legs
        rows.append(f"| {binding} | `{req}` | {show(lo)} | {show(hi)} |")
    table = "\n".join(["| binding | declared | lowest | latest |", "|---|---|---|---|", *rows])
    print(table)
    if failed:
        sys.exit(1)
    if "--github-output" in sys.argv:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            for binding, legs in matrices.items():
                f.write(f"{binding}={json.dumps(legs)}\n")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a") as f:
                f.write("### EarthSciAST releases under test\n\n" + table + "\n")


if __name__ == "__main__":
    main()
